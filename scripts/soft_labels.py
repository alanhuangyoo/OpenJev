"""Add a teacher decision model's probabilities to labelled rows as soft labels (distillation against forgetting).

Each labelled question gets soft_labels = mix * one-hot(gold) + (1 - mix) * teacher probabilities, in the format
wev.data reads: {option key: p} for choice, {"false": p, "true": p} for noul, [p per level] for score. Rows the
teacher cannot encode are written unchanged. dev.jsonl is copied as is. Killed runs resume: rows already written are
kept and skipped.

    python scripts/soft_labels.py --teacher jaredpalmer/kev-4b --data data/general/kev-v7 --out data/kd/kev-v7
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from wev.data import labelled_record, read_jsonl, split_file
from wev.inference import load
from wev.model import ContextTooLong, encode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, help="run dir or Hub repo of the teacher decision model")
    ap.add_argument("--data", required=True, help="directory with train.jsonl and dev.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", type=float, default=0.5, help="weight of the gold one-hot label")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    m = load(a.teacher, dtype=torch.bfloat16)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(split_file(a.data, "dev"), out / "dev.jsonl")
    rows = list(read_jsonl(Path(a.data) / "train.jsonl"))
    done = []
    if (out / "train.jsonl").exists():   # keep complete lines only: a kill can leave the last one cut short
        with open(out / "train.jsonl") as f:
            done = [line for line in f if line.endswith("\n")]
        with open(out / "train.jsonl", "w") as f:
            f.writelines(done)
    print(f"resuming after {len(done)} rows" if done else "starting", flush=True)
    n_soft = 0
    with open(out / "train.jsonl", "a") as f:
        for start in range(len(done), len(rows), a.batch):
            chunk, encs, metas = rows[start: start + a.batch], [], []
            for row in chunk:
                try:
                    rec, meta = labelled_record(row)
                    encs.append(encode(m.tokenizer, rec, m.max_state, m.max_branch, strict=True) if rec["questions"] else None)
                except (ContextTooLong, ValueError):
                    rec, meta = None, None
                    encs.append(None)
                metas.append(meta)
            live = [i for i, e in enumerate(encs) if e is not None]
            probs = m.model.probs([encs[i] for i in live]) if live else []
            for i, p in zip(live, probs):
                soft = {}
                for q_probs, qm, y in zip(p, metas[i], encs[i]["labels"]):
                    mixed = [(1 - a.mix) * t + a.mix * (k == y) for k, t in enumerate(q_probs)]
                    if qm["type"] == "choice":
                        soft[qm["id"]] = dict(zip(qm["keys"], mixed))
                    elif qm["type"] == "noul":
                        soft[qm["id"]] = {"false": mixed[0], "true": mixed[1]}
                    else:
                        soft[qm["id"]] = mixed
                chunk[i] = {**chunk[i], "soft_labels": soft}
                n_soft += 1
            f.writelines(json.dumps(row) + "\n" for row in chunk)
            f.flush()
    print(f"{a.data}: {n_soft}/{len(rows)} rows with soft labels -> {out}")


if __name__ == "__main__":
    main()
