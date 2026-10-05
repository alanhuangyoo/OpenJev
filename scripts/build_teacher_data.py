"""Teacher decision log + episode outcomes -> labelled training rows.

Kept:
  - every decision of an episode that ended DONE, when an LLM judge confirms, on the state of the final DONE decision,
    that the goal is achieved (the teacher's own DONE is not trusted blindly)
  - episodes stopped by the site (the teacher chose BLOCKED on a 403, CAPTCHA, Cloudflare or login wall page):
    every decision up to and including that BLOCKED
  - episodes cut short by browser infrastructure (daemon timeouts, a page navigating mid-read): every decision
    except the last one before the crash; the teacher's choices were sound, the harness failed
Dropped: loops caught by jev-ultrafast, action or model-call budget exhaustion (the teacher going in circles),
other crashes, and re-sent duplicate states within an episode.
Episodes are split into train/dev by task id hash, so no task is in both.

--onpolicy reads a dagger_server.py log, where the student or the teacher acted at each step and `labels` is always the
teacher's choice: an episode's ending is judged from the executed actions, the judge-confirmed final state is labelled
DONE, earlier teacher DONE labels (unverified) are dropped, and a BLOCKED ending is kept only when the teacher agrees.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge_done import SYSTEM, call, parse, user_message  # noqa: E402

INFRA = re.compile(r"IPCResponseTimeout|StalePage|timed out after|Document is navigating")
SITE_BLOCK = re.compile(r"403|captcha|unusual traffic|cloudflare|security verification|access denied|log ?in|sign ?in|"
                        r"robot|forbidden|blocked by", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev_frac", type=float, default=0.08)
    ap.add_argument("--onpolicy", action="store_true", help="the log comes from dagger_server.py")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    outcome = {e["episode"]: e for e in map(json.loads, open(a.episodes))}
    by = defaultdict(list)
    for r in map(json.loads, open(a.log)):
        by[r["session"]].append(r)
    base, key = os.environ["WEV_JUDGE_BASE_URL"], os.environ["WEV_JUDGE_API_KEY"]
    judge_model = os.environ.get("WEV_JUDGE_MODEL", "deepseek-v4.1-flash")
    stats, splits = Counter(), defaultdict(list)

    def acted(r):   # the operation that was carried out
        return (r["executed"] if a.onpolicy else r["labels"])["operation"]

    for ep, rows in by.items():
        e = outcome.get(ep)
        if e is None:
            stats["no_outcome"] += 1
            continue
        rows.sort(key=lambda r: r["t"])
        keep = None
        if e["outcome"] == "done":
            last_done = [r for r in rows if acted(r) == "DONE"]
            if not last_done:
                stats["done_without_done_decision"] += 1
                continue
            final = last_done[-1]
            final_request = dict(final["request"])
            resp = call(base, key, judge_model, [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": user_message(final_request)}])
            verdict = parse(resp["choices"][0]["message"]["content"])
            if not verdict or not verdict["done"]:
                stats["done_rejected_by_judge"] += 1
                continue
            keep = rows[: rows.index(final) + 1]
            if a.onpolicy:
                keep = [r for r in keep[:-1] if r["labels"]["operation"] != "DONE"] + [{**final, "labels": {"operation": "DONE"}}]
            stats["done_kept"] += 1
        elif e["outcome"] == "blocked":
            blocked = [r for r in rows if acted(r) == "BLOCKED" and r["labels"]["operation"] == "BLOCKED"
                       and SITE_BLOCK.search(r["reason"])]
            if not blocked:
                stats["blocked_loop_dropped"] += 1
                continue
            keep = rows[: rows.index(blocked[0]) + 1]
            stats["site_blocked_kept"] += 1
        elif e["outcome"] == "error" and INFRA.search(e.get("error", "")):
            if len(rows) < 3:
                stats["infra_error_too_short"] += 1
                continue
            keep = rows[:-1]
            stats["infra_error_prefix_kept"] += 1
        else:
            stats[f"{e['outcome']}_dropped"] += 1
            continue
        seen, dedup = set(), []
        for r in keep:
            sig = hashlib.sha256(json.dumps([r["request"], r["labels"]], sort_keys=True).encode()).hexdigest()
            if sig in seen:
                stats["duplicate_state"] += 1
                continue
            seen.add(sig)
            dedup.append(r)
        split = "dev" if int(hashlib.sha256(e["task"].encode()).hexdigest(), 16) % 1000 < a.dev_frac * 1000 else "train"
        for r in dedup:
            splits[split].append({"request": r["request"], "labels": r["labels"],
                                  "_meta": {"source": "teacher", "episode": ep, "task": e["task"], "task_source": e["source"],
                                            "teacher": r["teacher"], "reason": r["reason"],
                                            **({"actor": r["actor"]} if "actor" in r else {})}})
    for split, rows in splits.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ops = {s: dict(Counter(r["labels"]["operation"] for r in rows)) for s, rows in splits.items()}
    report = {"episodes": dict(stats), "rows": {s: len(r) for s, r in splits.items()}, "ops": ops}
    (out / "manifest.json").write_text(json.dumps({"doc": __doc__, **report}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
