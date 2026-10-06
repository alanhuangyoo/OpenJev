"""Training run -> self-contained model directory (what gets uploaded to the Hub).

The LoRA adapter is merged into the backbone in fp32 (exact), then the backbone is cast to the export dtype.
The export is checked against the run on real requests before it is written as complete.

    python -m wev.export --run runs/wd-1.7b --out exports/wev-1.7b --check data/m2w-v2/dev.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .inference import FORMAT, load
from .model import MAX_BRANCH, MAX_STATE, load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="", help="model name reported in responses (default: output dir name)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="backbone dtype in the export")
    ap.add_argument("--check", help="labelled jsonl; compare run and export on its first --check_n requests")
    ap.add_argument("--check_n", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")

    tok, model, meta = load_run(a.run, "cpu", dtype=torch.float32)
    backbone = model.lm.merge_and_unload()
    backbone = backbone.to(torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    backbone.config.use_cache = False
    out.mkdir(parents=True)
    backbone.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    save_file({k: v.contiguous() for k, v in model.head.state_dict().items()}, str(out / "head.safetensors"))
    info = {"format": FORMAT, "version": 1, "name": a.name or out.name, "base": meta["base"],
            "head_type": meta.get("head_type", "pointer"), "head_dim": meta.get("head_dim", 256),
            "keep_layers": meta.get("keep_layers"), "num_layers": backbone.config.num_hidden_layers,
            # inference limits: a page with hundreds of targets (an open date picker) needs more per question than
            # the 2048 tokens training used, and answering beyond the training range beats rejecting the step
            "max_state": meta.get("max_state") or MAX_STATE, "max_branch": max(meta.get("max_branch") or MAX_BRANCH, 8192),
            "train_max_state": meta.get("max_state") or MAX_STATE, "train_max_branch": meta.get("max_branch") or MAX_BRANCH,
            "backbone_dtype": a.dtype,
            "train_args": meta.get("train_args")}
    (out / "wev.json").write_text(json.dumps(info, indent=2))
    print(f"exported {a.run} -> {out} ({backbone.config.num_hidden_layers} layers, {a.dtype})", flush=True)

    if a.check:
        from .data import read_jsonl
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ref = load(a.run, device=dev)
        exp = load(str(out), device=dev)
        worst, flips, n = 0.0, 0, 0
        for i, row in enumerate(read_jsonl(a.check)):
            if i >= a.check_n:
                break
            r1 = ref.predict(row["request"]["state"], row["request"]["questions"])["answers"]
            r2 = exp.predict(row["request"]["state"], row["request"]["questions"])["answers"]
            for q in r1:
                # choice and score answers carry a distribution; noul carries one probability
                p1 = r1[q].get("probabilities") or {"true": r1[q]["noul"], "false": 1 - r1[q]["noul"]}
                p2 = r2[q].get("probabilities") or {"true": r2[q]["noul"], "false": 1 - r2[q]["noul"]}
                worst = max(worst, max(abs(p1[k] - p2[k]) for k in p1))
                flips += max(p1, key=p1.get) != max(p2, key=p2.get)
                n += 1
        print(f"check: {n} questions, max |dp| {worst:.4f}, argmax flips {flips}", flush=True)


if __name__ == "__main__":
    main()
