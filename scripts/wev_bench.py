"""wev-bench: score any POST /v1/systemone endpoint on the frozen browser-step test set (wev-data config `bench`).

    wev serve --model alanhuangya/OpenJev-4B --port 8009
    python scripts/wev_bench.py --url http://127.0.0.1:8009/v1/systemone --name wev-4b \
        --out results/wev-bench/wev-4b.json

Every row's `request` is sent exactly as stored. A step succeeds when the returned operation and, if the step has
one, its target both match the labels. Conventions follow JevBench: argmax accuracy; top-label ECE (10 equal-width
bins) on the probability the endpoint gives its own choice; multi-class Brier over the offered options; p50/p95 of
client-side wall-clock latency, linearly interpolated. A failed request or an answer that is not one of the offered
options counts as wrong and is reported separately. An answer without a distribution is scored at confidence 1
(one-hot), and the result says so. premature_done_rate is the share of steps whose gold operation is not DONE that
the endpoint answered DONE: an agent that stops early abandons its task.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "alanhuangya/OpenJev-Data"
BENCH_FILE = "bench/test.parquet"
VERSION = "1.0"
SUM_TOL = 0.02


def load_rows(data: str | None, revision: str | None):
    """-> (rows, source description). data: a parquet or JSONL file, a built dataset folder, or None (the Hub)."""
    if data is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO, BENCH_FILE, repo_type="dataset", revision=revision))
        source = {"repo": REPO, "revision": revision or "main", "file": BENCH_FILE}
    else:
        path = Path(data)
        if path.is_dir():
            path = path / BENCH_FILE if (path / BENCH_FILE).exists() else path / "test.parquet"
        source = {"file": str(path)}
    source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in open(path) if line.strip()]
    out = []
    for r in rows:
        req = r["request"] if isinstance(r["request"], dict) else json.loads(r["request"])
        labels = r["labels"] if isinstance(r["labels"], dict) else json.loads(r["labels"])
        out.append({"id": r.get("id"), "subset": r.get("bench_subset") or r.get("source") or "all",
                    "request": req, "labels": labels})
    return out, source


def post(url: str, body: dict, headers: dict, timeout: float):
    """-> (response dict or None, seconds, error or None)"""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
        return out, time.perf_counter() - t, None
    except Exception as e:   # HTTP 4xx/5xx, timeouts, malformed JSON
        return None, time.perf_counter() - t, f"{type(e).__name__}: {str(e)[:200]}"


def read_answer(answer, options: list[str]):
    """-> (choice or None, confidence, probabilities or None). probabilities are kept only when they are a valid
    distribution over the offered options."""
    if not isinstance(answer, dict):
        return None, 0.0, None
    choice = answer.get("choice")
    if choice not in options:
        return None, 0.0, None
    p = answer.get("probabilities")
    if isinstance(p, dict) and set(p) <= set(options) and all(isinstance(v, (int, float)) for v in p.values()) \
            and abs(sum(p.values()) - 1) <= SUM_TOL:
        return choice, float(p.get(choice, 0.0)), {k: float(p.get(k, 0.0)) for k in options}
    return choice, 1.0, None


def brier(p: dict, gold: str) -> float:
    return sum((v - (k == gold)) ** 2 for k, v in p.items())


def ece(pairs, bins: int = 10) -> float | None:
    if not pairs:
        return None
    acc = defaultdict(lambda: [0, 0.0, 0])
    for conf, ok in pairs:
        b = acc[min(int(min(max(conf, 0.0), 1.0) * bins), bins - 1)]
        b[0] += 1
        b[1] += conf
        b[2] += ok
    return sum(n / len(pairs) * abs(c / n - s / n) for n, s, c in acc.values())


def percentile(values, q: float):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] * (c - k) + v[c] * (k - f)


def score(row, resp, error):
    """-> per-step record."""
    labels, qs = row["labels"], row["request"]["questions"]
    answers = (resp or {}).get("answers") or {}
    rec = {"id": row["id"], "subset": row["subset"], "gold_operation": labels["operation"], "error": error,
           "questions": {}, "input_tokens": ((resp or {}).get("usage") or {}).get("input_tokens")}
    for qid, gold in labels.items():
        options = list(qs[qid]["criteria"])
        choice, conf, probs = read_answer(answers.get(qid), options)
        rec["questions"][qid] = {"gold": gold, "choice": choice, "confidence": conf, "correct": choice == gold,
                                 "valid": choice is not None, "distribution": probs is not None,
                                 "brier": brier(probs, gold) if probs else None}
    q = rec["questions"]
    rec["operation"] = q["operation"]["choice"]
    rec["step_success"] = all(x["correct"] for x in q.values())
    return rec


def summarise(recs) -> dict:
    n = len(recs)
    ok = [r for r in recs if r["error"] is None]
    valid = [r for r in ok if all(x["valid"] for x in r["questions"].values())]
    op = [r["questions"]["operation"] for r in recs]
    tg = [x for r in recs for qid, x in r["questions"].items() if qid != "operation"]
    not_done = [r for r in recs if r["gold_operation"] != "DONE"]
    by_gold = defaultdict(Counter)
    for r in recs:
        by_gold[r["gold_operation"]][r["operation"] or "<invalid>"] += 1
    briers = [x["brier"] for x in op if x["brier"] is not None]
    tokens = [r["input_tokens"] for r in recs if isinstance(r["input_tokens"], (int, float))]
    mean = lambda xs: sum(xs) / len(xs) if xs else None   # noqa: E731
    return {
        "n_items": n, "n_attempted": n, "n_ok": len(ok), "n_valid": len(valid),
        "coverage": len(ok) / n if n else None, "success_rate": len(valid) / n if n else None,
        "accuracy": mean([r["step_success"] for r in recs]),
        "step_success": mean([r["step_success"] for r in recs]),
        "operation_accuracy": mean([x["correct"] for x in op]),
        "target_accuracy": mean([x["correct"] for x in tg]), "n_targets": len(tg),
        "premature_done_rate": mean([r["operation"] == "DONE" for r in not_done]),
        "done_recall": mean([r["operation"] == "DONE" for r in recs if r["gold_operation"] == "DONE"]),
        "ece": ece([(x["confidence"], x["correct"]) for x in op + tg]),
        "ece_operation": ece([(x["confidence"], x["correct"]) for x in op]),
        "ece_target": ece([(x["confidence"], x["correct"]) for x in tg]),
        "brier_operation": mean(briers),
        "has_distribution": bool(op) and all(x["distribution"] for x in op + tg if x["valid"]),
        "mean_input_tokens": mean(tokens),
        "operation_by_gold": {g: {"n": sum(c.values()), "recall": c[g] / sum(c.values()), "predicted": dict(c)}
                              for g, c in sorted(by_gold.items())},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="the endpoint's POST /v1/systemone URL")
    ap.add_argument("--data", help=f"bench parquet/JSONL or a built dataset folder (default: {REPO} on the Hub)")
    ap.add_argument("--revision", help="Hub revision of the dataset to pin")
    ap.add_argument("--name", required=True, help="result key, e.g. wev-4b")
    ap.add_argument("--display", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--repo", default="", help="model or code link")
    ap.add_argument("--licence", default="")
    ap.add_argument("--underlying", default="", help="backbone and size")
    ap.add_argument("--open", default="yes", choices=["yes", "no", "partial"])
    ap.add_argument("--hardware", default="", help="where the endpoint runs, e.g. 'RTX 5090, bf16'")
    ap.add_argument("--measured_where", default="client on the same host as the endpoint")
    ap.add_argument("--note", default="")
    ap.add_argument("--model", default=None, help="value for the request's `model` field (default: as stored)")
    ap.add_argument("--api_key_env", help="name of an environment variable holding a bearer token")
    ap.add_argument("--concurrency", type=int, default=1, help=">1 is faster but inflates latency")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--warmup", type=int, default=3, help="requests sent first and not scored")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps_out", help="optional JSONL of per-step predictions")
    a = ap.parse_args()

    rows, source = load_rows(a.data, a.revision)
    if a.limit:
        rows = rows[: a.limit]
    headers = {"Authorization": f"Bearer {os.environ[a.api_key_env]}"} if a.api_key_env else {}

    def body(row):
        return {**row["request"], "model": a.model} if a.model else row["request"]

    for row in rows[: a.warmup]:
        post(a.url, body(row), headers, a.timeout)

    def run(row):
        resp, seconds, error = post(a.url, body(row), headers, a.timeout)
        return {**score(row, resp, error), "latency_s": seconds}

    with ThreadPoolExecutor(max(1, a.concurrency)) as ex:
        recs = []
        for i, rec in enumerate(ex.map(run, rows), start=1):
            recs.append(rec)
            if i % 200 == 0:
                print(f"{i}/{len(rows)}  step_success {sum(r['step_success'] for r in recs) / i:.3f}", flush=True)

    lat = [r["latency_s"] for r in recs if r["error"] is None]
    lane = summarise(recs)
    lane.update(latency_p50_s=percentile(lat, 0.5), latency_p95_s=percentile(lat, 0.95),
                latency_p50_ms=round(percentile(lat, 0.5) * 1000, 1) if lat else None,
                latency_p95_ms=round(percentile(lat, 0.95) * 1000, 1) if lat else None)
    families = defaultdict(list)
    for r in recs:
        families[r["subset"]].append(r)
    lane["by_family"] = {f: {k: v for k, v in summarise(rs).items() if k in (
        "n_items", "step_success", "operation_accuracy", "target_accuracy", "premature_done_rate", "ece")}
        for f, rs in sorted(families.items())}
    lane["errors"] = dict(Counter(r["error"].split(":")[0] for r in recs if r["error"]))
    result = {
        "benchmark": "wev-bench", "version": VERSION, "lane": "browser",
        "key": a.name, "display": a.display or a.name, "class": "browser-system-one", "open": a.open,
        "author": a.author, "repo": a.repo, "licence": a.licence, "underlying": a.underlying,
        "has_distribution": lane["has_distribution"],
        "probability_source": ["native"] if lane["has_distribution"] else ["onehot"],
        "browser": lane,
        "speed_block": {"measured_where": a.measured_where, "hardware": a.hardware or None, "n": len(lat),
                        "p50_s": lane["latency_p50_s"], "p95_s": lane["latency_p95_s"],
                        "run": f"{'serial' if a.concurrency <= 1 else f'{a.concurrency}-way concurrent'} "
                               f"{len(recs)}-step run, {a.warmup} warm-up requests excluded"},
        "endpoint_condition": a.hardware or None, "note": a.note,
        "data": {**source, "n": len(recs), "limit": a.limit or None},
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    if a.steps_out:
        with open(a.steps_out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({k: lane[k] for k in ("n_items", "step_success", "operation_accuracy", "target_accuracy",
                                           "premature_done_rate", "ece", "latency_p50_ms", "latency_p95_ms",
                                           "by_family", "errors")}, indent=1))


if __name__ == "__main__":
    main()
