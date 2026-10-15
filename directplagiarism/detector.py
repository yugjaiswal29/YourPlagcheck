from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .extraction import build_excerpt, is_empty_or_weak_text, normalize_whitespace, split_sentences, tokenize_words


class DirectPlagiarismDetector:
    def __init__(
        self,
        ngram_size: int = 5,
        min_passage_length: int = 5,
        similarity_threshold: float = 0.2,
    ) -> None:
        self.ngram_size = ngram_size
        self.min_passage_length = min_passage_length
        self.similarity_threshold = similarity_threshold

    def _normalize_text(self, text: str) -> str:
        value = unicodedata.normalize("NFKC", text or "")
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[\u2019\u2018]", "'", value)
        value = re.sub(r"[\u201C\u201D]", '"', value)
        value = re.sub(r"\s+", " ", value)
        value = value.lower()
        value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
        return re.sub(r"\s+", " ", value).strip()

    def _sentences_from_text(self, text: str) -> list[str]:
        return [sentence.strip() for sentence in split_sentences(text, minimum_length=3) if sentence.strip()]

    def _document_similarity(self, submission_text: str, reference_text: str) -> float:
        submission_tokens = self._normalize_text(submission_text)
        reference_tokens = self._normalize_text(reference_text)
        if not submission_tokens or not reference_tokens:
            return 0.0

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), lowercase=True)
        matrix = vectorizer.fit_transform([submission_tokens, reference_tokens])
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(similarity)

    def _match_sentence_pairs(self, submission_sentences: list[str], reference_sentences: list[str]) -> list[dict]:
        matches: list[dict] = []
        seen: set[tuple[int, int, int]] = set()

        for i, sub_sentence in enumerate(submission_sentences):
            sub_norm = self._normalize_text(sub_sentence)
            if len(sub_norm.split()) < self.min_passage_length:
                continue
            for j, ref_sentence in enumerate(reference_sentences):
                ref_norm = self._normalize_text(ref_sentence)
                if not ref_norm:
                    continue
                if sub_norm == ref_norm:
                    key = (i, j, len(sub_norm.split()))
                    if key not in seen:
                        seen.add(key)
                        matches.append(
                            {
                                "match_type": "exact",
                                "submission_index": i,
                                "reference_index": j,
                                "length": len(sub_norm.split()),
                                "similarity": 1.0,
                                "submission_excerpt": sub_sentence,
                                "reference_excerpt": ref_sentence,
                            }
                        )
                else:
                    overlap = self._sequence_overlap(sub_norm, ref_norm)
                    if overlap >= 0.75:
                        length = max(len(sub_norm.split()), len(ref_norm.split()))
                        key = (i, j, length)
                        if key not in seen:
                            seen.add(key)
                            matches.append(
                                {
                                    "match_type": "near_exact",
                                    "submission_index": i,
                                    "reference_index": j,
                                    "length": length,
                                    "similarity": round(overlap, 4),
                                    "submission_excerpt": sub_sentence,
                                    "reference_excerpt": ref_sentence,
                                }
                            )
        return matches

    def _sequence_overlap(self, left: str, right: str) -> float:
        left_tokens = left.split()
        right_tokens = right.split()
        if not left_tokens or not right_tokens:
            return 0.0

        common_tokens = set(left_tokens) & set(right_tokens)
        if not common_tokens:
            return 0.0
        return len(common_tokens) / max(len(set(left_tokens) | set(right_tokens)), 1)

    def _token_ngram_matches(self, submission_text: str, reference_text: str) -> list[dict]:
        submission_tokens = tokenize_words(submission_text)
        reference_tokens = tokenize_words(reference_text)
        if len(submission_tokens) < self.min_passage_length or len(reference_tokens) < self.min_passage_length:
            return []

        submission_ngrams: dict[tuple[str, ...], list[int]] = defaultdict(list)
        reference_ngrams: dict[tuple[str, ...], list[int]] = defaultdict(list)

        for idx in range(len(submission_tokens) - self.ngram_size + 1):
            submission_ngrams[tuple(submission_tokens[idx: idx + self.ngram_size])].append(idx)

        for idx in range(len(reference_tokens) - self.ngram_size + 1):
            reference_ngrams[tuple(reference_tokens[idx: idx + self.ngram_size])].append(idx)

        matches: list[dict] = []
        seen_keys: set[tuple[int, int, int]] = set()

        for start_index, ngram in enumerate(submission_ngrams):
            ref_positions = reference_ngrams.get(ngram)
            if not ref_positions:
                continue

            gram_tokens = list(ngram)
            sub_offset = start_index
            ref_offset = ref_positions[0]

            while sub_offset > 0 and ref_offset > 0 and submission_tokens[sub_offset - 1] == reference_tokens[ref_offset - 1]:
                sub_offset -= 1
                ref_offset -= 1

            sub_end = start_index + len(gram_tokens)
            ref_end = ref_positions[0] + len(gram_tokens)
            while sub_end < len(submission_tokens) and ref_end < len(reference_tokens) and submission_tokens[sub_end] == reference_tokens[ref_end]:
                sub_end += 1
                ref_end += 1

            length = sub_end - sub_offset
            if length < self.min_passage_length:
                continue
            match_key = (sub_offset, ref_offset, length)
            if match_key in seen_keys:
                continue
            seen_keys.add(match_key)

            matches.append(
                {
                    "match_type": "near_exact",
                    "submission_index": sub_offset,
                    "reference_index": ref_offset,
                    "length": length,
                    "similarity": round(min(1.0, length / max(self.ngram_size, 1)), 4),
                    "submission_excerpt": " ".join(submission_tokens[sub_offset:sub_end]),
                    "reference_excerpt": " ".join(reference_tokens[ref_offset:ref_end]),
                }
            )

        return matches

    def compare_submission_to_reference(self, submission_text: str, reference_text: str, reference_name: str) -> dict:
        submission_clean = normalize_whitespace(submission_text or "")
        reference_clean = normalize_whitespace(reference_text or "")

        if not submission_clean or not reference_clean:
            return {
                "reference_document": reference_name,
                "matches": [],
                "exact_matches": 0,
                "near_exact_matches": 0,
                "overall_similarity": 0.0,
                "score_breakdown": {"exact": 0.0, "near_exact": 0.0, "tfidf": 0.0},
                "status": "No usable text extracted",
            }

        submission_sentences = self._sentences_from_text(submission_text)
        reference_sentences = self._sentences_from_text(reference_text)

        exact_and_near_matches = self._match_sentence_pairs(submission_sentences, reference_sentences)
        ngram_matches = self._token_ngram_matches(submission_text, reference_text)

        aggregated_matches: list[dict] = []
        for match in exact_and_near_matches:
            aggregated_matches.append(match)
        for match in ngram_matches:
            aggregated_matches.append(match)

        aggregated_matches = self._merge_overlapping_matches(aggregated_matches)
        filtered_matches = [match for match in aggregated_matches if match["similarity"] >= self.similarity_threshold]

        tfidf_score = self._document_similarity(submission_text, reference_text)
        exact_score = max(0.0, sum(match["similarity"] for match in filtered_matches if match["match_type"] == "exact") / max(len(submission_sentences) or 1, 1))
        near_score = max(0.0, sum(match["similarity"] for match in filtered_matches if match["match_type"] == "near_exact") / max(len(submission_sentences) or 1, 1))
        overall_similarity = min(1.0, 0.5 * exact_score + 0.4 * near_score + 0.2 * tfidf_score)

        result = {
            "reference_document": reference_name,
            "matches": filtered_matches,
            "exact_matches": len([m for m in filtered_matches if m["match_type"] == "exact"]),
            "near_exact_matches": len([m for m in filtered_matches if m["match_type"] == "near_exact"]),
            "overall_similarity": round(overall_similarity, 4),
            "score_breakdown": {
                "exact": round(exact_score, 4),
                "near_exact": round(near_score, 4),
                "tfidf": round(float(tfidf_score), 4),
            },
            "status": "Qualifying overlap detected" if filtered_matches else "No qualifying matches detected",
        }
        return result

    def _merge_overlapping_matches(self, matches: list[dict]) -> list[dict]:
        if not matches:
            return []

        ordered = sorted(matches, key=lambda item: (item.get("submission_index", 0), item.get("reference_index", 0), item.get("length", 0)), reverse=False)
        merged: list[dict] = []
        for match in ordered:
            if not merged:
                merged.append(dict(match))
                continue
            current = merged[-1]
            sub_overlap = match.get("submission_index", 0) <= current.get("submission_index", 0) + current.get("length", 0)
            ref_overlap = match.get("reference_index", 0) <= current.get("reference_index", 0) + current.get("length", 0)
            if sub_overlap and ref_overlap:
                current["length"] = max(current.get("length", 0), match.get("length", 0))
                current["similarity"] = max(current.get("similarity", 0.0), match.get("similarity", 0.0))
                current["submission_excerpt"] = build_excerpt(current.get("submission_excerpt", "") + " " + match.get("submission_excerpt", ""))
                current["reference_excerpt"] = build_excerpt(current.get("reference_excerpt", "") + " " + match.get("reference_excerpt", ""))
            else:
                merged.append(dict(match))
        return merged

    def compare_documents(self, submission_text: str, reference_documents: Iterable[dict]) -> dict:
        comparisons: list[dict] = []
        weighted_scores: list[float] = []

        for reference in reference_documents:
            name = reference.get("filename") or reference.get("name") or "reference"
            reference_text = reference.get("text") or ""
            comparison = self.compare_submission_to_reference(submission_text, reference_text, name)
            comparisons.append(comparison)
            weighted_scores.append(comparison["overall_similarity"])

        overall_score = max(weighted_scores) if weighted_scores else 0.0

        return {
            "analysis_id": hashlib.sha256(str(len(comparisons) + overall_score).encode("utf-8")).hexdigest()[:12],
            "overall_similarity": round(overall_score, 4),
            "per_reference": comparisons,
            "matches": [match for comparison in comparisons for match in comparison["matches"]],
            "score_breakdown": {
                "exact": round(max((comparison["score_breakdown"]["exact"] for comparison in comparisons), default=0.0), 4),
                "near_exact": round(max((comparison["score_breakdown"]["near_exact"] for comparison in comparisons), default=0.0), 4),
                "tfidf": round(max((comparison["score_breakdown"]["tfidf"] for comparison in comparisons), default=0.0), 4),
            },
            "status": "No qualifying matches detected" if overall_score < self.similarity_threshold else "Review required",
        }
