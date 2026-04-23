from evals.metrics_gt import hit_at_k, recall_at_k, reciprocal_rank


class TestHitAtK:
    def test_all_relevant(self):
        assert hit_at_k([1, 2, 3], {1, 2}) == 1.0

    def test_no_overlap(self):
        assert hit_at_k([1, 2, 3], {99}) == 0.0

    def test_empty_retrieved(self):
        assert hit_at_k([], {1}) == 0.0

    def test_single_match_at_end(self):
        assert hit_at_k([5, 6, 7, 1], {1}) == 1.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k([1, 2, 3], {1, 2}) == 1.0

    def test_partial_recall(self):
        # 1 of 2 relevant pages retrieved
        assert recall_at_k([1, 99], {1, 2}) == 0.5

    def test_zero_recall(self):
        assert recall_at_k([10, 20], {1, 2}) == 0.0

    def test_no_relevant_set(self):
        assert recall_at_k([1, 2], set()) == 0.0

    def test_dedup_coverage(self):
        # Duplicate relevant retrievals don't inflate recall.
        assert recall_at_k([1, 1, 1], {1, 2}) == 0.5


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank([1, 2, 3], {1}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank([9, 1, 2], {1}) == 0.5

    def test_not_found(self):
        assert reciprocal_rank([9, 8, 7], {1}) == 0.0

    def test_empty(self):
        assert reciprocal_rank([], {1}) == 0.0
