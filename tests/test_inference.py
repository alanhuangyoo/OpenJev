import pytest
from transformers import AutoTokenizer

from wev.api import SystemOneRequest
from wev.inference import WebDecide
from wev.model import ContextTooLong


@pytest.fixture(scope="module")
def small():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    return WebDecide(tok, model=None, meta={"max_state": 400, "max_branch": 400}, name="t")


def request(text_words, n_options, n_history):
    options = [{"index": f"1:{j}", "label": f"Country → Option number {j}", "value": f"c{j}"} for j in range(n_options)]
    state = {"page": {"url": "", "title": "t", "text": "word " * text_words},
             "elements": [{"role": "combobox", "index": "1", "label": "Country", "operations": ["SELECT"],
                           "options": options}],
             "recent_actions": [{"action": f"step {i}", "kind": "click"} for i in range(n_history)]}
    return SystemOneRequest(state=state, questions={"operation": {"type": "choice", "instructions": "next?",
                                                                  "criteria": {"CLICK": None, "DONE": None}}})


def test_page_text_is_shortened_first(small):
    enc, _ = small.encode(request(text_words=600, n_options=0, n_history=0))
    assert enc["n_state"] <= 400


def test_dropdown_options_are_dropped_when_text_is_not_enough(small):
    enc, _ = small.encode(request(text_words=10, n_options=80, n_history=0))
    assert enc["n_state"] <= 400


def test_huge_url_is_cut(small):
    req = request(text_words=10, n_options=0, n_history=0)
    req.state["page"]["url"] = "https://example.com/search?" + "&".join(f"utm_{i}=abcdef{i}" for i in range(400))
    enc, _ = small.encode(req)
    assert enc["n_state"] <= 400


def test_history_is_cut_last(small):
    enc, _ = small.encode(request(text_words=10, n_options=0, n_history=60))
    assert enc["n_state"] <= 400


def test_huge_element_label_is_cut(small):
    req = request(text_words=10, n_options=0, n_history=0)
    req.state["elements"][0]["label"] = "label " * 3000
    enc, _ = small.encode(req)
    assert enc["n_state"] <= 400


def test_many_elements_are_halved(small):
    req = request(text_words=10, n_options=0, n_history=0)
    req.state["elements"] = [{"index": str(i)} for i in range(300)]
    enc, _ = small.encode(req)
    assert enc["n_state"] <= 400


def test_huge_option_is_cut(small):
    req = request(text_words=10, n_options=0, n_history=0)
    req.questions["operation"].criteria["CLICK"] = "click " * 3000
    enc, _ = small.encode(req)
    assert len(enc["ids"]) - enc["n_state"] <= 400


def test_unshrinkable_state_is_rejected():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    tiny = WebDecide(tok, model=None, meta={"max_state": 20, "max_branch": 400}, name="t")
    with pytest.raises(ContextTooLong):
        tiny.encode(SystemOneRequest(state="word " * 200, questions={"q": {"type": "noul", "instructions": "ok?"}}))


class _UniformModel:
    """Stand-in for DecisionModel: uniform probabilities over each question's options, no GPU needed."""

    def probs(self, encs):
        return [[[1 / len(o)] * len(o) for o in e["opt_idx"]] for e in encs]


def test_predict_returns_the_system_one_shape(small):
    m = WebDecide(small.tokenizer, _UniformModel(), small.meta, name="t")
    out = m.predict({"page": {"url": "", "title": "t", "text": "hello"}},
                    {"operation": {"type": "choice", "instructions": "next?", "criteria": {"CLICK": None, "DONE": None}},
                     "ok": {"type": "noul", "instructions": "ok?"}})
    assert set(out) == {"model", "answers", "usage", "latency_ms"}
    assert out["answers"]["operation"]["choice"] in ("CLICK", "DONE")
    assert set(out["answers"]["operation"]["probabilities"]) == {"CLICK", "DONE"}
    assert out["answers"]["ok"]["noul"] == 0.5
