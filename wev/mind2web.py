"""Mind2Web -> labelled System One requests shaped exactly like jev-ultrafast's per-step request.

One Mind2Web action step becomes one request:
  state      {page: {url, title, text}, elements: [...], recent_actions: [...]}
  questions  operation (CLICK | TYPE_TEXT | SELECT | SCROLL_* | WAIT | DONE | BLOCKED) + the gold operation's target
  labels     {"operation": <gold op>, "<op>_target": <gold element index>}

Splits are by website, so dev/test measure transfer to sites never seen in training. Mind2Web's official test sets
are distributed separately to prevent leakage and are not used here.

Known gap: Mind2Web has no DONE / WAIT / SCROLL / BLOCKED steps. They are offered as options, as in real requests,
but a model trained only on this data never sees them as the answer.
"""
import argparse
import hashlib
import json
import random
import re
from bisect import bisect_left
from collections import Counter
from pathlib import Path

from lxml import etree

from .prompts import (CONTROL_LABELS, HISTORY_STEPS, NEXT_ACTION, OPERATION_LABELS, PAGE_TEXT_CHARS, TARGET,
                      TERMINAL_LABELS)

OP_MAP = {"CLICK": "CLICK", "HOVER": "CLICK", "ENTER": "CLICK", "TYPE": "TYPE_TEXT", "SELECT": "SELECT"}
KIND = {"CLICK": "click", "TYPE_TEXT": "fill", "SELECT": "select"}
INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "option", "label", "summary"}
INTERACTIVE_ROLES = {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "option", "tab",
                     "menuitem", "switch"}
TEXT_INPUT_TYPES = {"", "text", "search", "email", "tel", "url", "number", "password", "date", "datetime-local",
                    "month", "week", "time"}
ROLE_BY_TAG = {"a": "link", "button": "button", "select": "combobox", "textarea": "textbox", "option": "option",
               "img": "img", "svg": "img", "summary": "button", "label": "label"}
LABEL_CHARS = 100
MAX_SELECT_OPTIONS = 60   # the API caps a question at 255 options; some Mind2Web dropdowns have over 1,000
REPR = re.compile(r"^\[(?P<tag>[^\]]*)\]\s*(?P<label>.*?)\s*->\s*(?P<op>[A-Z]+)(?::\s*(?P<value>.*))?$")


def _norm(s):
    return " ".join((s or "").split())


def _clip(s, n=LABEL_CHARS):
    s = _norm(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _role(el):
    t = (el.get("type") or "").lower()
    if el.tag == "input":
        if t in ("checkbox", "radio"):
            return t
        if t in ("submit", "button", "reset", "image"):
            return "button"
        return "searchbox" if t == "search" else "textbox"
    return ROLE_BY_TAG.get(el.tag, el.tag)


def _visible(candidate) -> bool:
    """Drop zero-size boxes and page-sized containers (negatives only)."""
    try:
        attrs = json.loads(candidate["attributes"])
        _, _, w, h = (float(x) for x in attrs["bounding_box_rect"].split(","))
    except (KeyError, ValueError, TypeError):
        return True
    return w > 0 and h > 0 and h < 1500


class Page:
    """Parsed Mind2Web cleaned_html. It is XML-like (self-closing tags, <text> nodes), so it is parsed as XML."""

    def __init__(self, html: str):
        self.root = etree.fromstring(html.encode("utf-8"), etree.XMLParser(recover=True, huge_tree=True))
        if self.root is None:
            raise ValueError("unparseable html")
        self.order, self.nodes, self.texts, self.text_pos = {}, {}, [], []
        for i, el in enumerate(self.root.iter()):
            if not isinstance(el.tag, str):
                continue
            self.order[el] = i
            nid = el.get("backend_node_id")
            if nid:
                self.nodes[nid] = el
            if el.tag == "text":
                t = _norm(el.text)
                if t:
                    self.texts.append(t)
                    self.text_pos.append(i)

    def page_text(self, limit: int) -> str:
        out, last = [], None
        for t in self.texts:
            if t != last:
                out.append(t)
                last = t
        return "\n".join(out)[:limit]

    def text_of(self, el) -> str:
        return _norm(" ".join(el.itertext()))

    def preceding_text(self, el) -> str:
        i = bisect_left(self.text_pos, self.order.get(el, 0)) - 1
        return self.texts[i] if i >= 0 else ""

    def describe(self, el) -> dict:
        a, tag = el.attrib, el.tag
        role = a.get("role") or _role(el)
        label = a.get("aria_label") or a.get("placeholder") or a.get("title") or a.get("alt")
        if not label and tag == "input" and role == "button":
            label = a.get("value")
        if not label:
            label = self.preceding_text(el) if tag in ("input", "select", "textarea") else self.text_of(el)
        if not label:
            label = a.get("value") or a.get("name") or a.get("id") or role
        d = {"role": role, "label": _clip(label)}
        if tag == "select":
            chosen = next((o for o in el.iter("option") if o.get("option_selected") == "true"), None)
            d["value"] = _clip(self.text_of(chosen) or chosen.get("value", "")) if chosen is not None else ""
        elif tag == "textarea":
            d["value"] = _clip(self.text_of(el))
        elif tag == "input" and role in ("textbox", "searchbox"):
            d["value"] = _clip(a.get("value", ""))
        return d

    def operations(self, el, d) -> list[str]:
        if el.tag == "select":
            return ["SELECT"]
        editable = (el.tag == "textarea"
                    or (el.tag == "input" and (el.get("type") or "").lower() in TEXT_INPUT_TYPES)
                    or d["role"] in ("textbox", "searchbox", "combobox")
                    or el.get("contenteditable") in ("true", ""))
        return ["TYPE_TEXT", "CLICK"] if editable else ["CLICK"]

    def select_options(self, el) -> list[tuple[str, str]]:
        out = []
        for o in el.iter("option"):
            label = _clip(self.text_of(o) or o.get("value") or "")
            if label:
                out.append((label, o.get("value") or label))
        return out


def recent_actions(reprs: list[str]) -> list[dict]:
    """Mind2Web action_reprs -> jev-ultrafast history entries {action, kind, text, page_changed}."""
    out = []
    for r in reprs[-HISTORY_STEPS:]:
        m = REPR.match(_norm(r))
        if not m:
            continue
        op = OP_MAP.get(m["op"], "CLICK")
        label = _clip(m["label"] or m["tag"])
        value = m["value"]
        if op == "SELECT" and value:
            label = f"{label} → {_clip(value)}"
        out.append({"action": label, "kind": KIND[op], "text": value if op == "TYPE_TEXT" else None,
                    "page_changed": None})
    return out


def _weighted_sample(items, weights, n, rng):
    """Without replacement (Efraimidis-Spirakis)."""
    keyed = sorted(((rng.random() ** (1.0 / w), i) for i, w in enumerate(weights)), reverse=True)
    return [items[i] for _, i in keyed[:n]]


def _matches(label, value, wanted) -> bool:
    w = _norm(wanted).casefold()
    return bool(w) and w in (_norm(label).casefold(), _norm(value).casefold())


def convert_step(task, i, page: Page, rng, k_min, k_max, text_min, text_max):
    """-> (row, None) or (None, skip_reason)."""
    step = task["actions"][i]
    op = OP_MAP.get(step["operation"]["op"])
    if op is None:
        return None, f"op:{step['operation']['op']}"
    gold = next((page.nodes[c["backend_node_id"]] for c in step["pos_candidates"]
                 if c["backend_node_id"] in page.nodes), None)
    if gold is None:
        return None, "no_positive" if not step["pos_candidates"] else "positive_not_in_html"
    if op == "SELECT" and gold.tag != "select":
        return None, "select_not_a_select"
    gold_d = page.describe(gold)

    # Ancestors/descendants of the gold element (and other positives) would also be correct clicks: never negatives.
    related = set(gold.iterancestors()) | set(gold.iterdescendants())
    related |= {page.nodes[c["backend_node_id"]] for c in step["pos_candidates"] if c["backend_node_id"] in page.nodes}
    seen = {gold_d["label"].casefold()}
    negatives = []
    for c in step["neg_candidates"]:
        el = page.nodes.get(c["backend_node_id"])
        if el is None or el in related or not _visible(c):
            continue
        d = page.describe(el)
        key = d["label"].casefold()
        if key in seen:   # identical labels make the target ambiguous
            continue
        seen.add(key)
        negatives.append((el, d))

    goal = task["confirmed_task"]
    questions, labels, state_elements, n, gold_target = build_questions(
        page, gold, gold_d, negatives, op, step["operation"]["value"], goal, rng, k_min, k_max)
    state = {"page": {"url": "", "title": task["website"],
                      "text": page.page_text(rng.randint(text_min, min(text_max, PAGE_TEXT_CHARS)))},
             "elements": state_elements,
             "recent_actions": recent_actions(task["action_reprs"][:i])}
    meta = {"source": "mind2web", "annotation_id": task["annotation_id"], "action_uid": step["action_uid"],
            "website": task["website"], "domain": task["domain"], "subdomain": task["subdomain"], "step": i,
            "op": op, "k": n, "has_target": gold_target is not None}
    return {"request": {"model": "wev-latest", "state": state, "questions": questions},
            "labels": labels, "_meta": meta}, None


def build_questions(page, gold, gold_d, negatives, op, value, goal, rng, k_min, k_max):
    """Gold + sampled negatives -> (questions, labels, state elements, K, gold target key or None).

    `page` provides describe(), operations() and select_options(); `value` is the requested SELECT option."""
    k = rng.randint(k_min, k_max)
    weights = [3.0 if (el.tag in INTERACTIVE_TAGS or d["role"] in INTERACTIVE_ROLES) else 1.0 for el, d in negatives]
    elements = [(gold, gold_d)] + _weighted_sample(negatives, weights, max(0, k - 1), rng)
    rng.shuffle(elements)

    targets = {"CLICK": {}, "TYPE_TEXT": {}, "SELECT": {}}
    state_elements, gold_target = [], None
    for idx, (el, d) in enumerate(elements, start=1):
        index = str(idx)
        ops = page.operations(el, d)
        if el is gold and op not in ops:
            ops = [op] + ops if op == "TYPE_TEXT" else ops + [op]
        e = {k: d[k] for k in ("role", "value") if k in d}
        e.update(index=index, label=d["label"], operations=ops)
        current = d.get("value", "")
        if "SELECT" in ops:
            e["value"], e["options"] = current, []
            options = list(enumerate(page.select_options(el), start=1))
            if len(options) > MAX_SELECT_OPTIONS:   # keep the requested option, sample the rest, keep page order
                keep = {j for j, (olab, oval) in options if el is gold and _matches(olab, oval, value)}
                rest = [j for j, _ in options if j not in keep]
                keep |= set(rng.sample(rest, MAX_SELECT_OPTIONS - len(keep)))
                options = [(j, o) for j, o in options if j in keep]
            for j, (olab, oval) in options:
                key = f"{index}:{j}"
                e["options"].append({"index": key, "label": f"{d['label']} → {olab}", "value": oval})
                targets["SELECT"][key] = {"element": f"[{index}] {d['label']} → {olab}", "current_value": current,
                                          "role": d["role"]}
                if el is gold and op == "SELECT" and gold_target is None and _matches(olab, oval, value):
                    gold_target = key
        for o in ("TYPE_TEXT", "CLICK"):
            if o in ops:
                # jev-ultrafast labels the click action of an editable field "Open <label>"
                label = f"Open {d['label']}" if o == "CLICK" and "TYPE_TEXT" in ops else d["label"]
                targets[o][index] = {"element": f"[{index}] {label}", "current_value": current, "role": d["role"]}
                if el is gold and op == o:
                    gold_target = index
        state_elements.append(e)

    present = [o for o in ("CLICK", "TYPE_TEXT", "SELECT") if targets[o]]
    rng.shuffle(present)   # in real requests this order follows the page's first element of each kind
    operations = {o: OPERATION_LABELS[o] for o in present}
    if rng.random() < 0.8:
        operations["SCROLL_DOWN"] = CONTROL_LABELS["SCROLL_DOWN"]
    if rng.random() < 0.3:
        operations["SCROLL_UP"] = CONTROL_LABELS["SCROLL_UP"]
    operations["WAIT"] = CONTROL_LABELS["WAIT"]
    operations.update(TERMINAL_LABELS)

    questions = {"operation": {"type": "choice", "criteria": operations,
                               "instructions": {"goal": goal, "rules": NEXT_ACTION}}}
    labels = {"operation": op}
    if gold_target is not None:
        qid = op.lower() + "_target"
        questions[qid] = {"type": "choice", "criteria": targets[op],
                          "instructions": {"goal": goal, "operation": op, "rules": [NEXT_ACTION, TARGET]}}
        labels[qid] = gold_target
    return questions, labels, state_elements, len(elements), gold_target


def split_of(website: str, seed: int, dev_frac: float, test_frac: float) -> str:
    h = int(hashlib.sha256(f"{seed}:{website}".encode()).hexdigest(), 16) / 16 ** 64
    return "test" if h < test_frac else "dev" if h < test_frac + dev_frac else "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_min", type=int, default=8, help="min elements offered per step (gold + negatives)")
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--text_min", type=int, default=1500, help="min page-text characters")
    ap.add_argument("--text_max", type=int, default=PAGE_TEXT_CHARS)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--train_variants", type=int, default=1, help="independent samplings per training step")
    ap.add_argument("--limit_tasks", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    from datasets import load_dataset
    ds = load_dataset("osunlp/Mind2Web", split="train")
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "dev", "test")}
    counts, skips, ops, websites, ks = Counter(), Counter(), Counter(), {s: set() for s in files}, Counter()
    n_tasks = len(ds) if not a.limit_tasks else min(a.limit_tasks, len(ds))
    for t in range(n_tasks):
        task = ds[t]
        split = split_of(task["website"], a.seed, a.dev_frac, a.test_frac)
        websites[split].add(task["website"])
        for i, step in enumerate(task["actions"]):
            try:
                page = Page(step["cleaned_html"])
            except ValueError:
                skips["unparseable_html"] += 1
                continue
            for v in range(a.train_variants if split == "train" else 1):
                rng = random.Random(f"{a.seed}:{step['action_uid']}:{v}")
                row, why = convert_step(task, i, page, rng, a.k_min, a.k_max, a.text_min, a.text_max)
                if row is None:
                    skips[why] += 1
                    break
                files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[split] += 1
                ops[(split, row["_meta"]["op"])] += 1
                ks[split] += row["_meta"]["k"]
        if (t + 1) % 100 == 0:
            print(f"{t + 1}/{n_tasks} tasks  {dict(counts)}  skipped {sum(skips.values())}", flush=True)
    for f in files.values():
        f.close()
    manifest = {"source": "osunlp/Mind2Web (train split)", "args": vars(a), "rows": dict(counts),
                "websites": {s: len(w) for s, w in websites.items()},
                "website_lists": {s: sorted(w) for s, w in websites.items()},
                "ops": {f"{s}:{o}": n for (s, o), n in sorted(ops.items())},
                "mean_k": {s: round(ks[s] / counts[s], 1) for s in counts if counts[s]},
                "skipped": dict(skips),
                "sha256": {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in files}}
    try:
        from huggingface_hub import HfApi
        manifest["dataset_revision"] = HfApi().dataset_info("osunlp/Mind2Web").sha
    except Exception as e:  # mirror without the API, offline, ...
        manifest["dataset_revision"] = f"unknown ({type(e).__name__})"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("rows", "websites", "mean_k", "skipped", "ops")}, indent=2))


if __name__ == "__main__":
    main()
