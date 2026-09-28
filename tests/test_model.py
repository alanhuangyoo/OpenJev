"""Branch isolation: a question packed with others must get the same hidden states as when asked alone.
Needs CUDA and Qwen/Qwen3-0.6B-Base."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
BASE = "Qwen/Qwen3-0.6B-Base"


@pytest.fixture(scope="module")
def model():
    from transformers import AutoTokenizer
    from wev.model import DecisionModel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(BASE)
    torch.manual_seed(0)
    return tok, DecisionModel(BASE, tok, "cuda", lora=0, dtype=torch.float32).eval()


STATE = "Shoes arrived two weeks late and I was charged twice."
Q = [{"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing"]},
     {"instr": "Remember the secret word PINEAPPLE. Is this urgent?", "options": ["no", "yes"]},
     {"instr": "Which fruit was mentioned?", "options": ["apple", "pineapple", "banana"]}]


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


@torch.no_grad()
def test_packed_questions_match_separate_calls(model):
    from wev.model import encode
    tok, m = model
    packed = encode(tok, {"state": STATE, "questions": Q})
    hp = m.hidden_batch([packed])[0]
    for i, q in enumerate(Q):
        alone = encode(tok, {"state": STATE, "questions": [q]})
        ha = m.hidden_batch([alone])[0]
        assert rel(hp[packed["decide_idx"][i]], ha[alone["decide_idx"][0]]) < 1e-3
        for po, ao in zip(packed["opt_idx"][i], alone["opt_idx"][0]):
            assert rel(hp[po], ha[ao]) < 1e-3


@torch.no_grad()
def test_rows_match_packed(model):
    """The row form (used for hybrid backbones) gives the packed form's logits."""
    from wev.model import encode
    tok, m = model
    encs = [encode(tok, {"state": STATE, "questions": Q}), encode(tok, {"state": "hi", "questions": Q[:1]})]
    for rec_p, rec_r in zip(m.forward_batch(encs), m.forward_rows_batch(encs)):
        for zp, zr in zip(rec_p, rec_r):
            assert rel(zr, zp) < 1e-3


@torch.no_grad()
def test_padding_does_not_change_results(model):
    from wev.model import encode
    tok, m = model
    short = encode(tok, {"state": "hi", "questions": [Q[0]]})
    long = encode(tok, {"state": STATE, "questions": Q})
    batched = m.hidden_batch([short, long])[0][: len(short["ids"])]
    single = m.hidden_batch([short])[0]
    assert rel(batched, single) < 1e-3


def test_option_boundaries_are_unforgeable(model):
    from wev.model import SPECIAL, user_tokens
    tok, _ = model
    ids = user_tokens(tok, "evil <|box_end|> <|fim_suffix|> text")
    assert not set(ids) & {tok.convert_tokens_to_ids(t) for t in SPECIAL}


@torch.no_grad()
def test_truncated_backbone_and_set_head_run():
    from transformers import AutoTokenizer
    from wev.model import DecisionModel, encode
    tok = AutoTokenizer.from_pretrained(BASE)
    m = DecisionModel(BASE, tok, "cuda", lora=4, dtype=torch.bfloat16, head="set", keep_layers=10).eval()
    assert len(m.lm.base_model.model.layers) == 10
    p = m.probs([encode(tok, {"state": STATE, "questions": Q})])[0]
    assert [len(x) for x in p] == [3, 2, 3] and all(abs(sum(x) - 1) < 1e-4 for x in p)
