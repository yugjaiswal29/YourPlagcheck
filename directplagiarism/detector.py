from __future__ import annotations

import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, TypedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class _MatchGroup(TypedDict):
    sub_start: int
    sub_end: int
    ref_start: int
    ref_end: int
    matched: int
    submission_indices: list[int]
    has_edits: bool


class DirectPlagiarismDetector:
    def __init__(
        self,
        min_passage_length: int = 5,
        near_exact_threshold: float = 0.8,
        max_edit_gap: int = 2,
    ) -> None:
        if min_passage_length < 1:
            raise ValueError("min_passage_length must be positive.")
        if not 0 < near_exact_threshold <= 1:
            raise ValueError("near_exact_threshold must be between 0 and 1.")
        if max_edit_gap < 0:
            raise ValueError("max_edit_gap cannot be negative.")

        self.min_passage_length = min_passage_length
        self.near_exact_threshold = near_exact_threshold
        self.max_edit_gap = max_edit_gap

    def _normalized_tokens(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        tokens: list[str] = []
        spans: list[tuple[int, int]] = []
        token_start: int | None = None
        for index, character in enumerate(text):
            category = unicodedata.category(character)
            is_token_character = category[0] in {"L", "N", "M"} or category == "Pc"
            if is_token_character and token_start is None:
                token_start = index
            elif not is_token_character and token_start is not None:
                tokens.append(unicodedata.normalize("NFKC", text[token_start:index]).casefold())
                spans.append((token_start, index))
                token_start = None
        if token_start is not None:
            tokens.append(unicodedata.normalize("NFKC", text[token_start:]).casefold())
            spans.append((token_start, len(text)))
        return tokens, spans

    def _sentence_index(self, text: str, position: int) -> int:
        return len(re.findall(r"[.!?](?:[\"')\]]*)\s+|\n+", text[:position]))

    def _document_similarity(self, submission_text: str, reference_text: str) -> float:
        if not submission_text.strip() or not reference_text.strip():
            return 0.0

        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform([submission_text, reference_text])
        except ValueError:
            # For example, both documents may contain only English stop words.
            return 0.0
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

    def _matching_groups(
        self,
        submission_tokens: list[str],
        reference_tokens: list[str],
    ) -> list[_MatchGroup]:
        matcher = SequenceMatcher(None, submission_tokens, reference_tokens, autojunk=False)
        groups: list[_MatchGroup] = []

        for tag, sub_start, sub_end, ref_start, ref_end in matcher.get_opcodes():
            if tag != "equal":
                continue

            if groups:
                previous = groups[-1]
                sub_gap = sub_start - int(previous["sub_end"])
                ref_gap = ref_start - int(previous["ref_end"])
                candidate_matched = int(previous["matched"]) + sub_end - sub_start
                candidate_length = max(
                    sub_end - int(previous["sub_start"]),
                    ref_end - int(previous["ref_start"]),
                )
                if (
                    sub_gap >= 0
                    and ref_gap >= 0
                    and max(sub_gap, ref_gap) <= self.max_edit_gap
                    and candidate_matched / candidate_length >= self.near_exact_threshold
                ):
                    previous["sub_end"] = sub_end
                    previous["ref_end"] = ref_end
                    previous["matched"] = int(previous["matched"]) + sub_end - sub_start
                    previous["submission_indices"].extend(range(sub_start, sub_end))
                    previous["has_edits"] = bool(previous["has_edits"]) or sub_gap > 0 or ref_gap > 0
                    continue

            groups.append(
                {
                    "sub_start": sub_start,
                    "sub_end": sub_end,
                    "ref_start": ref_start,
                    "ref_end": ref_end,
                    "matched": sub_end - sub_start,
                    "submission_indices": list(range(sub_start, sub_end)),
                    "has_edits": False,
                }
            )

        return groups

    def _find_matches(
        self,
        submission_text: str,
        reference_text: str,
        reference_name: str,
    ) -> list[dict]:
        submission_tokens, submission_spans = self._normalized_tokens(submission_text)
        reference_tokens, reference_spans = self._normalized_tokens(reference_text)
        if not submission_tokens or not reference_tokens:
            return []

        matches: list[dict] = []
        for group in self._matching_groups(submission_tokens, reference_tokens):
            sub_start = int(group["sub_start"])
            sub_end = int(group["sub_end"])
            ref_start = int(group["ref_start"])
            ref_end = int(group["ref_end"])
            matched_words = int(group["matched"])
            passage_length = sub_end - sub_start
            comparison_length = max(passage_length, ref_end - ref_start)

            if matched_words < self.min_passage_length:
                continue

            sequence_similarity = matched_words / comparison_length
            is_near_exact = bool(group["has_edits"])
            if is_near_exact and sequence_similarity < self.near_exact_threshold:
                # Do not bridge edits unless the surrounding ordered sequence
                # remains sufficiently close.
                continue

            sub_char_start, sub_char_end = (
                submission_spans[sub_start][0],
                submission_spans[sub_end - 1][1],
            )
            ref_char_start, ref_char_end = (
                reference_spans[ref_start][0],
                reference_spans[ref_end - 1][1],
            )
            while sub_char_end < len(submission_text) and submission_text[sub_char_end] in ".,!?;:)]}\"'":
                sub_char_end += 1
            while ref_char_end < len(reference_text) and reference_text[ref_char_end] in ".,!?;:)]}\"'":
                ref_char_end += 1
            matches.append(
                {
                    "match_type": "near_exact" if is_near_exact else "exact",
                    "submission_index": self._sentence_index(submission_text, sub_char_start),
                    "reference_index": self._sentence_index(reference_text, ref_char_start),
                    "submission_word_index": sub_start,
                    "reference_word_index": ref_start,
                    "length": passage_length,
                    "matched_word_count": matched_words,
                    "_submission_word_indices": group["submission_indices"],
                    "similarity": round(sequence_similarity, 4),
                    "submission_excerpt": submission_text[sub_char_start:sub_char_end],
                    "reference_excerpt": reference_text[ref_char_start:ref_char_end],
                    "reference_document": reference_name,
                }
            )

        return matches

    def _coverage(
        self,
        matches: list[dict],
        token_count: int,
        match_type: str | None = None,
    ) -> float:
        covered: set[int] = set()
        for match in matches:
            if match_type is not None and match["match_type"] != match_type:
                continue
            covered.update(match["_submission_word_indices"])
        return len(covered) / token_count if token_count else 0.0

    def compare_submission_to_reference(
        self,
        submission_text: str,
        reference_text: str,
        reference_name: str,
    ) -> dict:
        if not submission_text.strip() or not reference_text.strip():
            return {
                "reference_document": reference_name,
                "submission_word_count": len(self._normalized_tokens(submission_text)[0]),
                "reference_word_count": len(self._normalized_tokens(reference_text)[0]),
                "matches": [],
                "exact_matches": 0,
                "near_exact_matches": 0,
                "overall_similarity": 0.0,
                "score_breakdown": {"exact": 0.0, "near_exact": 0.0, "tfidf": 0.0},
                "status": "No usable text extracted",
            }

        submission_tokens, _ = self._normalized_tokens(submission_text)
        matches = self._find_matches(submission_text, reference_text, reference_name)
        exact_score = self._coverage(matches, len(submission_tokens), "exact")
        near_score = self._coverage(matches, len(submission_tokens), "near_exact")
        overall_similarity = self._coverage(matches, len(submission_tokens))
        tfidf_score = self._document_similarity(submission_text, reference_text)

        # Direct overlap coverage is the primary score. TF-IDF remains a
        # separate contextual signal and cannot inflate the direct-match score.
        for match in matches:
            match.pop("_submission_word_indices")
        return {
            "reference_document": reference_name,
            "submission_word_count": len(submission_tokens),
            "matches": matches,
            "exact_matches": sum(match["match_type"] == "exact" for match in matches),
            "near_exact_matches": sum(match["match_type"] == "near_exact" for match in matches),
            "overall_similarity": round(overall_similarity, 4),
            "score_breakdown": {
                "exact": round(exact_score, 4),
                "near_exact": round(near_score, 4),
                "tfidf": round(tfidf_score, 4),
            },
            "status": "Qualifying overlap detected" if matches else "No qualifying matches detected",
        }

    def compare_documents(self, submission_text: str, reference_documents: Iterable[dict]) -> dict:
        comparisons = [
            self.compare_submission_to_reference(
                submission_text,
                reference.get("text") or "",
                reference.get("filename") or reference.get("name") or "reference",
            )
            for reference in reference_documents
        ]
        best_comparison = max(comparisons, key=lambda result: result["overall_similarity"], default=None)
        overall_score = best_comparison["overall_similarity"] if best_comparison else 0.0
        score_breakdown = best_comparison["score_breakdown"] if best_comparison else {
            "exact": 0.0,
            "near_exact": 0.0,
            "tfidf": 0.0,
        }
        matches = [match for comparison in comparisons for match in comparison["matches"]]
        return {
            "analysis_id": hashlib.sha256(
                f"{len(comparisons)}:{overall_score}:{len(matches)}".encode("utf-8")
            ).hexdigest()[:12],
            "overall_similarity": round(overall_score, 4),
            "per_reference": comparisons,
            "matches": matches,
            "score_breakdown": score_breakdown,
            "status": "Review required" if matches else "No qualifying matches detected",
        }
