"""Labelled rows {request, labels, _meta} -> encoded training/eval items.

A row's request is exactly what a client sends to /v1/systemone; labels map question id -> gold key
(choice), bool (noul) or level index (score). Rendering goes through api.to_record, the same path as serving.
"""
import json
from pathlib import Path

from pydantic import ValidationError

from .api import SystemOneRequest, to_record
from .model import ContextTooLong, encode


def read_jsonl(path):
    """Rows of a JSONL file, or of a wev-data parquet file (request, labels and meta stored as JSON strings)."""
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        for r in pq.read_table(path, columns=["request", "labels", "meta"]).to_pylist():
            yield {"request": json.loads(r["request"]), "labels": json.loads(r["labels"]),
                   "_meta": json.loads(r["meta"])}
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def labelled_record(row: dict):
    """-> (record with integer labels, question metadata). Questions without a label are dropped: branches are
    isolated, so a question's prediction does not depend on which other questions are packed with it."""
    req = SystemOneRequest.model_validate(row["request"])
    rec, meta = to_record(req)
    qs, ms = [], []
    for q, m in zip(rec["questions"], meta):
        if m["id"] not in row["labels"]:
            continue
        y = row["labels"][m["id"]]
        if m["type"] == "choice":
            idx = m["keys"].index(y)
        elif m["type"] == "noul":
            idx = int(bool(y))
        else:
            idx = int(y)
        soft = (row.get("soft_labels") or {}).get(m["id"])
        if soft is not None:
            # soft label: a probability per option, in the order the question renders its options
            if m["type"] == "choice":
                soft = [float(soft.get(k, 0.0)) for k in m["keys"]]
            elif m["type"] == "noul":
                soft = [float(soft.get("false", 0.0)), float(soft.get("true", 0.0))]
            else:
                soft = [float(x) for x in soft]
            total = sum(soft)
            soft = [x / total for x in soft] if total > 0 and len(soft) == len(q["options"]) else None
        qs.append({**q, "label": idx, "soft": soft})
        ms.append(m)
    return {"state": rec["state"], "questions": qs}, ms


def load_items(tok, path, max_state, max_branch, max_tokens=None, limit=None):
    """-> (items, dropped). Invalid requests and rows that do not fit the context are dropped, never truncated."""
    items, dropped = [], 0
    for row in read_jsonl(path):
        try:
            rec, meta = labelled_record(row)
        except ValidationError:   # e.g. a dropdown with more than 255 options: the API itself rejects it
            dropped += 1
            continue
        if not rec["questions"]:
            dropped += 1
            continue
        try:
            enc = encode(tok, rec, max_state, max_branch, strict=True)
        except ContextTooLong:
            dropped += 1
            continue
        if max_tokens and len(enc["ids"]) > max_tokens:
            dropped += 1
            continue
        items.append({"enc": enc, "meta": meta, "_meta": row.get("_meta", {})})
        if limit and len(items) >= limit:
            break
    return items, dropped


def split_file(directory, split: str) -> Path:
    """A split's file, JSONL or parquet; "dev" and "validation" name the same split (Hugging Face datasets use
    "validation")."""
    directory = Path(directory)
    names = [split] + {"dev": ["validation"], "validation": ["dev"]}.get(split, [])
    for name in names:
        for ext in (".jsonl", ".parquet"):
            if (directory / f"{name}{ext}").exists():
                return directory / f"{name}{ext}"
    return directory / f"{split}.jsonl"
