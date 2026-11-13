from discourse.preprocessing import preprocess


def test_negation_preserved():
    text = "ini bukan toxic sama sekali"
    assert "bukan" in preprocess(text, pipeline="P0")
    assert "bukan" in preprocess(text, pipeline="P1")


def test_url_placeholder():
    text = "cek https://example.com dong"
    out = preprocess(text, pipeline="P1")
    assert "URLTOKEN" in out
    assert "https://" not in out
