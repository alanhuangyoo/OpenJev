"""Decision model: causal-LM backbone (no vocab head) + block-causal branch mask + pointer readout.

Adapted from kev (https://github.com/jaredpalmer/kev, Apache-2.0), kev/model.py.

Packed sequence:   <state> ...state...
                   <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>   <- question 1
                   <q> instructions <opt> option 1 </opt> ... <decide>                          <- question 2
Mask:      token i attends to j iff j <= i and j is in the state or in the same question as i.
Positions: every question branch restarts its positions right after the state, so question order is irrelevant.
Readout:   a pointer head scores each option's </opt> hidden state against the <decide> hidden state.
"""
import contextlib
import math
import re
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Rarely used Qwen special tokens reused as delimiters (state, q, opt, /opt, decide); no embedding rows are added,
# LoRA adapts their meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Gated DeltaNet projections of hybrid backbones (Qwen3.5), adapted too
HYBRID_TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
MAX_STATE, MAX_BRANCH = 4096, 2048


class ContextTooLong(ValueError):
    def __init__(self, part: str, n: int, limit: int):
        super().__init__(f"{part} is {n} tokens, limit {limit}")
        self.part, self.n, self.limit = part, n, limit


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text: str) -> list[int]:
    """Tokenize caller text so it can never produce a delimiter token (option boundaries are unforgeable)."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, rec: dict, max_state: int = MAX_STATE, max_branch: int = MAX_BRANCH, strict: bool = False) -> dict:
    """Pack one record {state, questions: [{instr, options, label?}]} into one token sequence.

    Returns ids, seg (0 = state, k = question k), pos, decide_idx [Q], opt_idx [Q][K], labels [Q].
    strict=True raises ContextTooLong instead of truncating the state.
    """
    state = user_tokens(tok, rec["state"])
    if len(state) + 1 > max_state:
        if strict:
            raise ContextTooLong("state", len(state) + 1, max_state)
        state = state[: max_state - 1]
    s_id, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
    prefix = [s_id] + state
    ids, seg, pos = list(prefix), [0] * len(prefix), list(range(len(prefix)))
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        branch = instr + [t for sp in spans for t in sp] + [d_id]
        if len(branch) > max_branch:
            raise ContextTooLong(f"question {k}", len(branch), max_branch)
        base = len(ids)
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(base + cursor - 1)
        ids += branch
        seg += [k] * len(branch)
        pos += list(range(len(prefix), len(prefix) + len(branch)))
        decide_idx.append(base + len(branch) - 1)
        opt_idx.append(ends)
    return {"ids": ids, "seg": seg, "pos": pos, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q.get("label") for q in rec["questions"]], "soft": [q.get("soft") for q in rec["questions"]],
            "n_state": len(prefix)}


def branch_mask_batch(segs: list[list[int]], device, dtype, length: int | None = None) -> torch.Tensor:
    """Additive [B, 1, L, L] block-causal mask, right-padded. Pads belong to no segment (-1); padded query rows keep
    the diagonal so no row is fully masked (finfo.min, not -inf, so softmax stays finite)."""
    n = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), n), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    allow = allow | torch.eye(n, dtype=torch.bool, device=device)[None]
    mask = torch.zeros(len(segs), n, n, dtype=dtype, device=device)
    return mask.masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_of(enc: dict):
    """Split a packed encoding into the state and one causal row per question: (state ids, state pos, rows), with
    rows[k] = {"ids", "pos", "decide", "opts"} and readout offsets within the branch. State + rows[k] as one causal
    sequence holds exactly the tokens question k may attend to, at the same positions, so it equals the packed
    block-causal form on any architecture, including recurrent layers that cannot take the mask."""
    seg, n_state = enc["seg"], enc["seg"].count(0)
    rows, start = [], n_state
    for d, oi in zip(enc["decide_idx"], enc["opt_idx"]):
        rows.append({"ids": enc["ids"][start: d + 1], "pos": enc["pos"][start: d + 1], "decide": d - start,
                     "opts": [o - start for o in oi]})
        start = d + 1
    return enc["ids"][:n_state], enc["pos"][:n_state], rows


class PointerHead(nn.Module):
    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:  # [d], [K, d] -> [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class SetHead(nn.Module):
    """Pointer scores refined by a small transformer over the option set, so options are compared with each other
    before scoring (e.g. "Search" vs "Advanced search"). No positional encoding: permuting the options permutes the
    logits. The refinement's output layer starts at zero, so training starts exactly at the pointer head."""

    def __init__(self, d: int, dp: int = 256, layers: int = 2, heads: int = 4):
        super().__init__()
        self.pointer = PointerHead(d, dp)
        self.inp = nn.Linear(d, dp)
        self.kind = nn.Embedding(2, dp)   # 0 = <decide>, 1 = option
        layer = nn.TransformerEncoderLayer(dp, heads, dim_feedforward=4 * dp, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.mix = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(dp, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:  # [d], [K, d] -> [K]
        kinds = torch.ones(h_opts.shape[0] + 1, dtype=torch.long, device=h_opts.device)
        kinds[0] = 0
        x = self.inp(torch.cat([h_decide[None], h_opts], 0)) + self.kind(kinds)
        y = self.mix(x[None])[0]
        return self.pointer(h_decide, h_opts) + self.out(y[1:] * y[:1]).squeeze(-1)


HEADS = {"pointer": PointerHead, "set": SetHead}


def truncate_layers(lm, keep: int):
    """Keep the first `keep` decoder layers (the final norm stays). Deciding may not need the top layers, which a
    causal LM spends on predicting the next token."""
    n = len(lm.layers)
    if not 0 < keep <= n:
        raise ValueError(f"keep_layers must be in 1..{n}")
    lm.layers = nn.ModuleList(list(lm.layers)[:keep])
    lm.config.num_hidden_layers = keep
    if getattr(lm.config, "layer_types", None):
        lm.config.layer_types = lm.config.layer_types[:keep]


class DecisionModel(nn.Module):
    def __init__(self, base: str, tok, device, lora: int = 16, dtype=torch.bfloat16, head_dim: int = 256,
                 attn: str | None = None, head: str = "pointer", keep_layers: int | None = None, backbone=None,
                 revision: str | None = None):
        """backbone: an already-built (e.g. exported, merged and truncated) decoder to use instead of loading `base`."""
        super().__init__()
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        self.base, self.head_dim, self.device, self.lm_dtype = base, head_dim, device, dtype
        self.head_type, self.keep_layers, self.revision = head, keep_layers, revision
        self._lock = threading.Lock()
        if backbone is not None:
            self.lm = backbone
        else:
            # backbone only: the vocab head is dropped, the model can no longer generate text
            self.lm = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=dtype,
                                                           attn_implementation=attn).model
            if keep_layers:
                truncate_layers(self.lm, keep_layers)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        # hybrid backbones (Qwen3.5: Gated DeltaNet layers are recurrent) cannot take the block-causal mask, so each
        # question runs as its own causal row after the state (rows_of)
        self.hybrid = "linear_attention" in set(getattr(self.lm.config, "layer_types", None) or [])
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05,
                             target_modules=LORA_TARGETS + (HYBRID_TARGETS if self.hybrid else []))
            self.lm = get_peft_model(self.lm, cfg)   # adapter weights are kept in fp32 under a bf16 base
        self.head = HEADS[head](self.lm.config.hidden_size, dp=head_dim)   # fp32, trained from scratch
        self.to(device)

    def _autocast(self):
        if str(self.device).startswith("cuda") and self.lm_dtype in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self.lm_dtype)
        return contextlib.nullcontext()

    def hidden_batch(self, encs: list[dict]) -> torch.Tensor:
        """[B, L_max, d] fp32 hidden states for a right-padded batch. Pads sit after every real token and are masked
        keys, so padding never changes a real token's hidden state."""
        n = max(len(e["ids"]) for e in encs)
        ids = torch.full((len(encs), n), self.pad_id, dtype=torch.long, device=self.device)
        pos = torch.zeros((len(encs), n), dtype=torch.long, device=self.device)
        for b, e in enumerate(encs):
            ids[b, : len(e["ids"])] = torch.tensor(e["ids"], device=self.device)
            pos[b, : len(e["pos"])] = torch.tensor(e["pos"], device=self.device)
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, self.lm_dtype, length=n)
        with self._autocast():
            out = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, use_cache=False)
        return out.last_hidden_state.float()

    def forward_rows_batch(self, encs: list[dict]) -> list[list[torch.Tensor]]:
        """Row form: every question is one right-padded causal row (state + its branch); the state is recomputed per
        question."""
        rows, readouts = [], []
        for b, e in enumerate(encs):
            s_ids, s_pos, branches = rows_of(e)
            for r in branches:
                rows.append((s_ids + r["ids"], s_pos + r["pos"]))
                readouts.append((b, len(s_ids) + r["decide"], [len(s_ids) + o for o in r["opts"]]))
        n = max(len(ids) for ids, _ in rows)
        ids = torch.full((len(rows), n), self.pad_id, dtype=torch.long, device=self.device)
        pos = torch.zeros((len(rows), n), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), n), dtype=torch.long, device=self.device)
        for i, (r_ids, r_pos) in enumerate(rows):
            ids[i, : len(r_ids)] = torch.tensor(r_ids, device=self.device)
            pos[i, : len(r_pos)] = torch.tensor(r_pos, device=self.device)
            att[i, : len(r_ids)] = 1
        with self._autocast():
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, use_cache=False).last_hidden_state.float()
        out = [[] for _ in encs]
        for i, (b, d, oi) in enumerate(readouts):
            out[b].append(self.head(h[i, d], h[i, torch.tensor(oi, device=self.device)]))
        return out

    def forward_batch(self, encs: list[dict]) -> list[list[torch.Tensor]]:
        """Per record, per question: option logits [K]."""
        if self.hybrid:
            return self.forward_rows_batch(encs)
        hs = self.hidden_batch(encs)
        out = []
        for b, e in enumerate(encs):
            h = hs[b]
            out.append([self.head(h[d], h[torch.tensor(oi, device=self.device)])
                        for d, oi in zip(e["decide_idx"], e["opt_idx"])])
        return out

    @torch.no_grad()
    def probs(self, encs: list[dict]) -> list[list[list[float]]]:
        # one forward at a time: the GPU serialises them anyway, and the Triton autotuner behind the Gated DeltaNet
        # kernels of hybrid backbones is not thread-safe (concurrent server requests crash it)
        with self._lock:
            return [[F.softmax(z, -1).tolist() for z in rec] for rec in self.forward_batch(encs)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, out, **extra):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(out)
        torch.save({"head": self.head.state_dict(), "base": self.base, "base_revision": self.revision,
                    "head_dim": self.head_dim,
                    "head_type": self.head_type, "keep_layers": self.keep_layers, **extra}, out / "head.pt")


def load_run(run, device, dtype=torch.bfloat16, trainable: bool = False):
    """Local run directory or a Hub repo id -> (tokenizer, model, run metadata); in eval mode unless trainable, which
    keeps the adapter and head trainable to continue training from the run. Runs saved by kev load too."""
    run = Path(run) if Path(run).is_dir() else Path(_download(str(run)))
    meta = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(meta["base"], revision=meta.get("base_revision"))
    model = DecisionModel(meta["base"], tok, device, lora=0, dtype=dtype, head_dim=meta.get("head_dim", 256),
                          head=meta.get("head_type", "pointer"), keep_layers=meta.get("keep_layers"),
                          revision=meta.get("base_revision"))
    from peft import PeftModel
    model.lm = PeftModel.from_pretrained(model.lm, str(run), is_trainable=trainable).to(device)
    model.head.load_state_dict(meta["head"])
    if not trainable:
        model.eval()
    return tok, model, meta


def _download(repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.md"])
