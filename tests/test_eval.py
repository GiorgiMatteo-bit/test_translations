from htr_bench.eval import EvalConfig, normalize_text, score_page, score_run


def test_normalize_collapses_whitespace():
    assert normalize_text("a  b\nc\t d") == "a b c d"


def test_perfect_match_is_zero_error():
    s = score_page("hello world", "hello world", EvalConfig())
    assert s.cer == 0.0
    assert s.wer == 0.0


def test_full_miss_is_unit_wer():
    s = score_page("hello world", "", EvalConfig())
    assert s.wer == 1.0
    assert s.cer == 1.0


def test_score_run_handles_missing_hyp():
    refs = {"p1": "hello world", "p2": "foo bar baz"}
    hyps = {"p1": "hello world"}
    lats = {"p1": 1.0, "p2": 1.0}
    r = score_run(refs, hyps, lats, EvalConfig())
    assert r.n_pages == 2
    # p2 missing → counted as empty hypothesis, so corpus errors are nonzero
    assert r.cer > 0
    assert r.wer > 0
