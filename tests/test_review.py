from acr.config import Settings
from acr.review import Reviewer, outputs_match, parse_examples

CF = "Problem\n\n-----Examples-----\nInput\n3\n1 2 3\n\nOutput\n6\n\nInput\n1\n5\n\nOutput\n5\n\n-----Note-----\nnote"
AT = "Problem\n-----Sample Input-----\n3 4\n\n-----Sample Output-----\n7\n\nExplanation here.\n"


def test_parse_examples_both_formats():
    assert parse_examples(CF) == [("3\n1 2 3\n", "6"), ("1\n5\n", "5")]
    assert parse_examples(AT) == [("3 4\n", "7")]
    assert parse_examples("no examples here") == []


def test_outputs_match():
    assert outputs_match("YES\n", "yes")
    assert outputs_match("0.3333333", "0.33333333")
    assert outputs_match("1 2\n3", "1 2 3")
    assert not outputs_match("1 2", "1 3")


def test_reviewer_verdicts(tmp_path):
    r = Reviewer(Settings(cache_dir=tmp_path, review_timeout_s=2.0))
    ex = parse_examples(CF)
    ok = "input()\nprint(sum(map(int, input().split())))"
    assert r.verdict(ok, ex) == "pass"
    assert r.verdict("input()\nprint(0)", ex) == "wrong"
    assert r.verdict("print 1", ex) == "error"                 # Python 2 syntax under Python 3
    assert r.verdict("while True: pass", ex) == "timeout"
    assert Reviewer(Settings(cache_dir=tmp_path)).cache        # verdicts were saved to disk
