"""JSONL subsets -> Hugging Face dataset folder of parquet configs (wev-data v2).

  mind2web          data/m2w-v2
  nnetnav_audited   data/nnetnav-v3-clean
  teacher_qwen3max  data/teacher-v1 + data/teacher-v2 (column `collection`)
  teacher_glm       data/teacher-v3-glm
  onpolicy_glm      data/dagger-g1 (column `actor`: who acted; the label is always the GLM teacher's choice)
  webchain          data/webchain-v1 (when present)
  live_tasks        tasks-v1.jsonl (train -> train, eval -> test)
  bench             frozen wev-bench test set: mind2web test + nnetnav_audited test + the teacher test slices
                    (minus requests the API would reject: a question with more than 255 options)

Teacher collections have train/dev only, split by a hash of the task id, so a task is dev in every collection.
A dev task becomes test when neither its task id nor its goal occurs in any train split of any config, and validation
otherwise (NNetNav-sourced teacher tasks take their goals from NNetNav train, so they always land in validation).
Every row keeps the full request as a JSON string (replayable as is) plus flat columns.
The 153 end-to-end eval tasks are checked against every train split (by task id and by goal); overlaps are reported
in manifest.json, never silently fixed.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq

from redact import redact_row, unloaded_done

SPLITS = {"train": "train", "dev": "validation", "test": "test"}
SOURCES = {
    "mind2web": ["m2w-v2"],
    "nnetnav_audited": ["nnetnav-v3-clean"],
    "teacher_qwen3max": ["teacher-v1", "teacher-v2"],
    "teacher_glm": ["teacher-v3-glm"],
    "onpolicy_glm": ["dagger-g1"],
    "webchain": ["webchain-v1"],
}
TEACHER = {"teacher_qwen3max", "teacher_glm", "onpolicy_glm"}
MAX_OPTIONS = 255   # the API's cap per question (wev.api); larger requests are rejected with 422
BENCH = ["mind2web", "nnetnav_audited", "teacher_qwen3max", "teacher_glm", "onpolicy_glm"]
STEP_SCHEMA = pa.schema([
    ("id", pa.string()), ("url", pa.string()), ("title", pa.string()), ("goal", pa.string()),
    ("n_elements", pa.int32()), ("n_recent_actions", pa.int32()), ("operation", pa.string()),
    ("target_question", pa.string()), ("target", pa.string()), ("target_element", pa.string()),
    ("source", pa.string()), ("collection", pa.string()), ("website", pa.string()), ("task_id", pa.string()),
    ("episode_id", pa.string()), ("actor", pa.string()), ("teacher", pa.string()), ("reason", pa.string()),
    ("request", pa.string()), ("labels", pa.string()), ("meta", pa.string()),
])
BENCH_SCHEMA = STEP_SCHEMA.append(pa.field("bench_subset", pa.string()))
TASK_SCHEMA = pa.schema([("id", pa.string()), ("url", pa.string()), ("goal", pa.string()), ("source", pa.string())])


def servable(row) -> bool:
    questions = json.loads(row["request"])["questions"].values()
    return all(1 <= len(q.get("criteria") or ()) <= MAX_OPTIONS for q in questions)


def norm_goal(g) -> str:
    return " ".join(str(g or "").split()).casefold().rstrip(".")


def goal_of(row) -> str:
    instr = row["request"]["questions"]["operation"]["instructions"]
    return instr.get("goal", "") if isinstance(instr, dict) else ""


def ids_of(config, m):
    """-> (task id, episode id, website) from a row's _meta."""
    if config == "mind2web":
        return m.get("annotation_id"), m.get("annotation_id"), m.get("website")
    if config == "webchain":
        return m.get("trace"), m.get("trace"), m.get("host")
    return m.get("task"), m.get("episode") or m.get("id"), None


def flat(config, collection, row) -> dict:
    req, labels, m = row["request"], row["labels"], row.get("_meta", {})
    state = req["state"]
    page = state.get("page") or {}
    op = labels.get("operation")
    tq = next((q for q in labels if q != "operation"), None)
    target = labels.get(tq) if tq else None
    crit = req["questions"].get(tq, {}).get("criteria", {}) if tq else {}
    task, episode, website = ids_of(config, m)
    blob = json.dumps([req, labels], sort_keys=True, ensure_ascii=False)
    return {"id": hashlib.sha1(blob.encode()).hexdigest()[:16], "url": page.get("url") or "",
            "title": page.get("title") or "", "goal": goal_of(row), "n_elements": len(state.get("elements") or []),
            "n_recent_actions": len(state.get("recent_actions") or []), "operation": op, "target_question": tq,
            "target": target, "target_element": (crit.get(target) or {}).get("element") if target else None,
            "source": m.get("source"), "collection": collection,
            "website": website or urlparse(page.get("url") or "").hostname, "task_id": task, "episode_id": episode,
            "actor": m.get("actor"), "teacher": m.get("teacher"), "reason": m.get("reason"),
            "request": json.dumps(req, ensure_ascii=False), "labels": json.dumps(labels, ensure_ascii=False),
            "meta": json.dumps(m, ensure_ascii=False)}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: Path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd", row_group_size=1000)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="directory holding the JSONL subsets")
    ap.add_argument("--tasks", default="tasks-v1.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")

    # config -> split -> flat rows
    built = defaultdict(lambda: defaultdict(list))
    removed = Counter()   # rows dropped by redaction (card / SSN) or as DONE on a page that had not loaded
    sources = {}
    for config, dirs in SOURCES.items():
        if not all((data / d / "manifest.json").exists() for d in dirs):   # absent or still being built
            print(f"skip {config}: no manifest in {dirs}")
            continue
        sources[config] = {d: json.loads((data / d / "manifest.json").read_text()).get("sha256") for d in dirs}
        for d in dirs:
            for raw, split in SPLITS.items():
                if (data / d / f"{raw}.jsonl").exists():
                    for r in read_jsonl(data / d / f"{raw}.jsonl"):
                        r = redact_row(r)
                        if r is None or unloaded_done(r):
                            removed[config, "pii" if r is None else "unloaded_done"] += 1
                            continue
                        row = flat(config, d, r)
                        built[config][raw if config in TEACHER and raw == "dev" else split].append(row)

    tasks = read_jsonl(a.tasks)
    eval_tasks, live_train = [t for t in tasks if t["split"] == "eval"], [t for t in tasks if t["split"] == "train"]
    eval_ids, eval_goals = {t["id"] for t in eval_tasks}, {norm_goal(t["goal"]): t["id"] for t in eval_tasks}
    train_ids, train_goals = set(), set()
    for config, splits in built.items():
        for r in splits["train"]:
            train_ids.add(r["task_id"])
            train_goals.add(norm_goal(r["goal"]))

    for config in TEACHER & built.keys():
        for r in built[config].pop("dev", []):
            seen = r["task_id"] in train_ids or norm_goal(r["goal"]) in train_goals
            built[config]["validation" if seen else "test"].append(r)

    overlap = {"train": {}, "other": {}}
    for config, splits in built.items():
        for split, rows in splits.items():
            hit = [r for r in rows if r["task_id"] in eval_ids or norm_goal(r["goal"]) in eval_goals]
            if hit:
                ids = sorted({r["task_id"] if r["task_id"] in eval_ids else eval_goals[norm_goal(r["goal"])]
                              for r in hit})
                kind = "train" if split == "train" else "other"
                overlap[kind][f"{config}/{split}"] = {"eval_tasks": ids, "rows": len(hit)}
    overlap["live_tasks/train"] = sorted(t["id"] for t in eval_tasks
                                         if norm_goal(t["goal"]) in {norm_goal(x["goal"]) for x in live_train})

    counts, sha = defaultdict(dict), {}
    for config, splits in built.items():
        for split in ("train", "validation", "test"):
            if splits.get(split):
                sha[f"{config}/{split}"] = write(out / config / f"{split}.parquet", splits[split], STEP_SCHEMA)
                counts[config][split] = len(splits[split])
    live = {"train": live_train, "test": eval_tasks}
    for split, rows in live.items():
        sha[f"live_tasks/{split}"] = write(out / "live_tasks" / f"{split}.parquet",
                                           [{k: t[k] for k in TASK_SCHEMA.names} for t in rows], TASK_SCHEMA)
        counts["live_tasks"][split] = len(rows)
    bench = [{**r, "bench_subset": c} for c in BENCH if c in built for r in built[c]["test"] if servable(r)]
    sha["bench/test"] = write(out / "bench" / "test.parquet", bench, BENCH_SCHEMA)
    counts["bench"]["test"] = len(bench)

    ops = {c: {s: dict(Counter(r["operation"] for r in rows)) for s, rows in sp.items()} for c, sp in built.items()}
    ops["bench"] = {"test": dict(Counter(r["operation"] for r in bench))}
    manifest = {"doc": __doc__, "rows": counts, "operations": ops,
                "removed": {f"{c}/{why}": n for (c, why), n in sorted(removed.items())},
                "bench_subsets": dict(Counter(r["bench_subset"] for r in bench)),
                "teacher_test_tasks": {c: len({r["task_id"] for r in built[c]["test"]})
                                       for c in TEACHER & built.keys()},
                "eval_task_overlap": overlap, "sources_sha256": sources, "sha256": sha}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    summary = {k: manifest[k] for k in ("rows", "bench_subsets", "teacher_test_tasks")}
    summary["eval_tasks_in_train"] = {**overlap["train"], "live_tasks/train": overlap["live_tasks/train"]}
    summary["eval_tasks_elsewhere"] = {k: {"tasks": len(v["eval_tasks"]), "rows": v["rows"]}
                                       for k, v in overlap["other"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
