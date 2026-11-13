from discourse.targets import summarize_votes


def test_soft_label():
    x = summarize_votes([0, 1, 1])
    assert x["n"] == 3
    assert abs(x["q"] - 2 / 3) < 1e-8
    assert x["hard"] == 1


def test_tie():
    x = summarize_votes([0, 1])
    assert x["q"] == 0.5
    assert x["hard"] is None


def test_missing_target():
    x = summarize_votes([])
    assert x["mask"] is False
    assert x["q"] is None


def test_string_votes():
    x = summarize_votes(["0", "1"])
    assert x["n"] == 2
    assert x["hard"] is None
