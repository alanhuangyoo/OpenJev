"""Fit the serving temperature of an exported model on development rows and record it in its wev.json.

Logits are divided by one scalar T before the softmax; T minimises the negative log-likelihood of the gold options on
the dev splits (never test). The argmax, and so every accuracy, is unchanged; only the probabilities move.

    python scripts/calibrate.py --model exports/wev-4b --data data/m2w-v2 data/nnetnav-v3-clean data/general/kev-v7
"""
import argparse
import json
import math
from pathlib import Path

import torch

from wev.data import load_items, split_file
from wev.evaluate import ece
from wev.inference import load


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="exported model directory (wev.json is updated in place)")
    ap.add_argument("--data", nargs="+", required=True, help="directories whose dev split is used")
    ap.add_argument("--n", type=int, default=400, help="dev records per directory")
    a = ap.parse_args()
    m = load(a.model)
    m.model.temperature = 1.0
    logits, gold = [], []
    with torch.no_grad():
        for d in a.data:
            items, _ = load_items(m.tokenizer, split_file(d, "dev"), m.max_state, m.max_branch, limit=a.n)
            for it in items:
                enc = it["enc"]
                rec = m.model.forward_prefix(enc) if m.model.hybrid and len(enc["decide_idx"]) > 1 else m.model.forward_batch([enc])[0]
                for z, y in zip(rec, enc["labels"]):
                    logits.append(z.float().cpu())
                    gold.append(y)

    def nll(t):
        return sum(-torch.log_softmax(z / t, -1)[y].item() for z, y in zip(logits, gold)) / len(gold)

    lo, hi = 0.3, 5.0   # golden-section search on log T
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = nll(x1), nll(x2)
    for _ in range(30):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = nll(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = nll(x2)
    t = round((lo + hi) / 2, 4)

    def report(temp):
        conf, hit = [], []
        for z, y in zip(logits, gold):
            p = torch.softmax(z / temp, -1)
            conf.append(p.max().item())
            hit.append(int(p.argmax().item() == y))
        return {"nll": round(nll(temp), 4), "ece": round(ece(conf, hit), 4)}

    fit = {"temperature": t, "questions": len(gold), "data": a.data, "before": report(1.0), "after": report(t)}
    info_path = Path(a.model) / "wev.json"
    info = json.loads(info_path.read_text())
    info.update(temperature=t, temperature_fit=fit)
    info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()
