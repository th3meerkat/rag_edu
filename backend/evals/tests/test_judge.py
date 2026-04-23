from evals.judge import parse_veredicto


class TestParseVeredicto:
    def test_positive(self):
        assert parse_veredicto("VEREDICTO: 1") == 1

    def test_negative(self):
        assert parse_veredicto("VEREDICTO: 0") == 0

    def test_whitespace_tolerant(self):
        assert parse_veredicto("   veredicto   :  1  ") == 1

    def test_preamble(self):
        assert parse_veredicto("Análisis del fragmento. VEREDICTO: 1\n") == 1

    def test_unparseable(self):
        assert parse_veredicto("no tengo opinión") is None

    def test_only_valid_values(self):
        # "VEREDICTO: 5" → the regex only matches 0/1, so 5 is None.
        assert parse_veredicto("VEREDICTO: 5") is None
