import unittest

from directplagiarism.detector import DirectPlagiarismDetector


class DirectPlagiarismDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = DirectPlagiarismDetector(min_passage_length=5)

    def compare(self, submission, reference):
        return self.detector.compare_submission_to_reference(submission, reference, "reference.txt")

    def test_identical_passage_is_reported_once_as_exact(self):
        passage = (
            "Researchers developed a reliable method for measuring urban air quality "
            "across different neighborhoods."
        )
        result = self.compare(passage, passage)

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["match_type"], "exact")
        self.assertEqual(result["matches"][0]["submission_excerpt"], passage)
        self.assertEqual(result["matches"][0]["reference_excerpt"], passage)
        self.assertEqual(result["overall_similarity"], 1.0)
        self.assertEqual(result["score_breakdown"]["tfidf"], 1.0)

    def test_contiguous_copied_passage_in_longer_submission_has_faithful_excerpt(self):
        copied = (
            "Researchers developed a reliable method for measuring urban air quality "
            "across different neighborhoods."
        )
        submission = f"Background information comes first. {copied} Additional analysis follows."
        result = self.compare(submission, copied)

        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["submission_excerpt"], copied)
        self.assertEqual(result["matches"][0]["reference_excerpt"], copied)
        self.assertEqual(result["matches"][0]["submission_index"], 1)
        self.assertGreater(result["overall_similarity"], 0.5)
        self.assertLess(result["overall_similarity"], 1.0)

    def test_one_word_edit_is_near_exact_not_exact(self):
        reference = "Careful researchers collected reliable measurements from remote coastal stations."
        submission = "Careful researchers collected accurate measurements from remote coastal stations."
        result = self.compare(submission, reference)

        self.assertEqual(result["exact_matches"], 0)
        self.assertEqual(result["near_exact_matches"], 1)
        self.assertEqual(result["matches"][0]["similarity"], 0.8889)
        self.assertIn("accurate", result["matches"][0]["submission_excerpt"])
        self.assertIn("reliable", result["matches"][0]["reference_excerpt"])

    def test_reordered_words_do_not_become_a_perfect_near_exact_match(self):
        reference = (
            "Researchers developed a reliable method for measuring urban air quality "
            "across different neighborhoods."
        )
        submission = (
            "Neighborhoods different across quality air urban measuring for method "
            "reliable a developed researchers."
        )
        result = self.compare(submission, reference)

        self.assertFalse(any(match["match_type"] == "near_exact" for match in result["matches"]))
        self.assertLess(result["overall_similarity"], 1.0)

    def test_common_vocabulary_without_a_long_ordered_passage_is_not_flagged(self):
        submission = "The study examines people and their work in a changing world of ideas."
        reference = "People work through ideas while the world changes and studies examine them."
        result = self.compare(submission, reference)

        self.assertEqual(result["matches"], [])
        self.assertEqual(result["overall_similarity"], 0.0)
        self.assertGreater(result["score_breakdown"]["tfidf"], 0.0)

    def test_stopword_only_documents_do_not_raise(self):
        result = self.compare("the and if but or the", "and the but if or")

        self.assertEqual(result["matches"], [])
        self.assertEqual(result["score_breakdown"]["tfidf"], 0.0)

    def test_matching_is_case_and_unicode_normalization_insensitive(self):
        result = self.compare(
            "The CAFÉ study uses a reliable method for testing coastal water quality.",
            "The cafe\u0301 study uses a reliable method for testing coastal water quality.",
        )

        self.assertEqual(result["exact_matches"], 1)
        self.assertEqual(result["matches"][0]["submission_excerpt"], "The CAFÉ study uses a reliable method for testing coastal water quality.")

    def test_large_unmatched_gap_does_not_merge_evidence(self):
        left = "Careful researchers collected reliable measurements from remote coastal stations."
        right = "The separate experiment evaluated long-term forest recovery in northern regions."
        result = self.compare(
            f"{left} This unrelated section contains many distinct words and observations. {right}",
            f"{left} The source then discusses a different topic entirely. {right}",
        )

        self.assertTrue(all(match["match_type"] == "exact" for match in result["matches"]))
        self.assertEqual(len(result["matches"]), 2)

    def test_overall_score_combines_nonoverlapping_exact_and_near_exact_coverage(self):
        exact = "A careful team collected accurate readings across several remote stations."
        reference_near = "Researchers documented seasonal changes throughout the northern valley ecosystem."
        submission_near = "Researchers recorded seasonal changes throughout the northern valley ecosystem."
        result = self.compare(
            f"{exact} Blue quartz shifts slowly beneath rugged desert cliffs today. {submission_near}",
            f"{exact} Silver granite drifts quietly toward frozen alpine caves tonight. {reference_near}",
        )

        self.assertEqual(result["exact_matches"], 1)
        self.assertEqual(result["near_exact_matches"], 1)
        self.assertAlmostEqual(result["overall_similarity"], 18 / 28, places=4)

    def test_multiple_references_report_source_for_each_match_and_best_score(self):
        copied = "A careful process preserves accurate records during long scientific investigations."
        result = self.detector.compare_documents(
            f"Opening note. {copied}",
            [
                {"filename": "unrelated.txt", "text": "Completely different words describe another subject."},
                {"filename": "copied.txt", "text": copied},
            ],
        )

        self.assertEqual(result["overall_similarity"], 0.8333)
        self.assertEqual(result["matches"][0]["reference_document"], "copied.txt")
        self.assertEqual(result["per_reference"][1]["exact_matches"], 1)


if __name__ == "__main__":
    unittest.main()
