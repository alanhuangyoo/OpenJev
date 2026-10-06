"""WebChain (webagentlab/webchain, CC BY 4.0) -> labelled System One requests, like wev.mind2web.

WebChain v2 ships screenshots and multimodal SFT conversations. Its action metadata links, for most steps, a
pre-action HTML snapshot (html_dom_url) and an accessibility tree (ax_tree_url) on data.imean.tech, together with the
CSS selector of the element acted on. One step becomes:
  state      page url, title and visible text; 8-40 candidates drawn from the snapshot's interactive elements
  questions  operation + the gold operation's target, rendered exactly as in wev.mind2web
  labels     {"operation": <gold op>, "<op>_target": <gold element index>}
The gold element is the unique match of the recorded selector (climbing to the nearest interactive ancestor, e.g.
from a <span> to its <button>). WebChain has no candidate lists, so negatives are the snapshot's other visible
interactive elements, never the gold's ancestors or descendants.

Kept actions: click, hover, double_click -> CLICK; type, paste -> TYPE_TEXT; select on a <select> -> SELECT.
Dropped: drag, right_click, copy, press_enter, back, launchApp; steps without a snapshot; selectors that do not
resolve to exactly one element; non-English pages and goals. Like Mind2Web, there are no DONE / WAIT / SCROLL /
BLOCKED labels. Splits are by host. Snapshots are cached under --cache, so a rerun downloads nothing.
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lxml.html

from .mind2web import (INTERACTIVE_ROLES, INTERACTIVE_TAGS, TEXT_INPUT_TYPES, _clip, _norm, _role, build_questions,
                       split_of)
from .prompts import HISTORY_STEPS, PAGE_TEXT_CHARS

REPO = "webagentlab/webchain"
OP_MAP = {"click": "CLICK", "hover": "CLICK", "double_click": "CLICK", "type": "TYPE_TEXT", "paste": "TYPE_TEXT",
          "select": "SELECT"}
KIND = {"CLICK": "click", "TYPE_TEXT": "fill", "SELECT": "select"}
SKIP_TAGS = {"script", "style", "noscript", "template", "head", "svg", "path", "iframe"}
CJK = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
NO_TEXT = "no input text"
ACTION_COLS = ["trace_uid", "source_step_index", "action_type", "included_in_sft", "html_dom_url", "selector", "href",
               "value", "input_text", "attributes"]
TRACE_COLS = ["uid", "query", "primary_host", "web_type", "intent_type"]


def clean_goal(q: str) -> str:
    """Drop the annotation prefix some queries carry ('Task 4: "..."') and wrapping quotes."""
    q = re.sub(r"^\s*-?\s*(task\s*\d+\s*:)?\s*", "", _norm(q), flags=re.I)
    return q.strip("\"'“”").strip()


def _cjk_share(s: str) -> float:
    return len(CJK.findall(s)) / max(1, len(s))


def _hidden(el) -> bool:
    style = (el.get("style") or "").replace(" ", "").lower()
    return (el.get("hidden") is not None or el.get("aria-hidden") == "true" or "display:none" in style
            or "visibility:hidden" in style or (el.tag == "input" and (el.get("type") or "").lower() == "hidden"))


class WebPage:
    """A WebChain HTML snapshot, with the describe()/operations()/select_options() interface of mind2web.Page."""

    def __init__(self, html: bytes):
        self.root = lxml.html.fromstring(html)
        self.hidden, self.order, self.texts, self.text_pos, self.labels_for = set(), {}, [], [], {}
        self._walk(self.root, False)
        for lab in self.root.iter("label"):
            if lab.get("for"):
                self.labels_for[lab.get("for")] = self.text_of(lab)

    def _walk(self, el, hidden):
        if not isinstance(el.tag, str):
            return
        hidden = hidden or el.tag in SKIP_TAGS or _hidden(el)
        self.order[el] = len(self.order)
        if hidden:
            self.hidden.add(el)
        elif el.text and _norm(el.text):
            self.texts.append(_norm(el.text))
            self.text_pos.append(self.order[el])
        for c in el:
            self._walk(c, hidden)
            if not hidden and isinstance(c.tag, str) and c.tail and _norm(c.tail):
                self.texts.append(_norm(c.tail))
                self.text_pos.append(self.order[c])

    def title(self) -> str:
        t = self.root.find(".//title")
        return _clip(t.text_content()) if t is not None else ""

    def lang(self) -> str:
        return (self.root.get("lang") or "").lower()

    def page_text(self, limit: int) -> str:
        out, last = [], None
        for t in self.texts:
            if t != last:
                out.append(t)
                last = t
        return "\n".join(out)[:limit]

    def text_of(self, el) -> str:
        return _norm(" ".join(t for t in el.itertext() if t))

    def preceding_text(self, el) -> str:
        pos = self.order.get(el, 0)
        prev = [t for t, p in zip(self.texts, self.text_pos) if p < pos]
        return prev[-1] if prev else ""

    def describe(self, el) -> dict:
        a, tag = el.attrib, el.tag
        role = a.get("role") or _role(el)
        label = a.get("aria-label") or a.get("placeholder") or a.get("title") or a.get("alt")
        if not label and tag == "input" and role == "button":
            label = a.get("value")
        if not label and tag in ("input", "select", "textarea"):
            label = self.labels_for.get(a.get("id") or "\0") or self.preceding_text(el)
        if not label:
            label = self.text_of(el)
        if not label:
            label = a.get("value") or a.get("name") or a.get("id") or role
        d = {"role": role, "label": _clip(label)}
        if tag == "select":
            opts = list(el.iter("option"))
            chosen = next((o for o in opts if o.get("selected") is not None), opts[0] if opts else None)
            d["value"] = _clip(self.text_of(chosen) or chosen.get("value", "")) if chosen is not None else ""
        elif tag == "textarea":
            d["value"] = _clip(self.text_of(el))
        elif tag == "input" and role in ("textbox", "searchbox", "combobox"):
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

    def interactive(self, el) -> bool:
        return isinstance(el.tag, str) and el not in self.hidden and (
            el.tag in INTERACTIVE_TAGS or el.get("role") in INTERACTIVE_ROLES or el.get("onclick") is not None
            or el.get("contenteditable") in ("true", ""))

    def candidates(self):
        return [el for el in self.root.iter() if self.interactive(el) and el.tag != "option"
                and not any(p.tag == "select" for p in el.iterancestors())]

    def resolve(self, selector: str):
        """Unique element for a recorded CSS selector, else None."""
        from cssselect import GenericTranslator
        try:
            hits = self.root.xpath(GenericTranslator().css_to_xpath(selector.replace(">", " > ")))
        except Exception:   # cssselect rejects some recorded selectors (ids starting with a digit, ...)
            return None
        return hits[0] if len(hits) == 1 else None


def gold_element(page: WebPage, el, op):
    """The element a System One would be offered for this action."""
    if op == "TYPE_TEXT" and el.tag not in ("input", "textarea") and el.get("contenteditable") is None:
        fields = [d for d in el.iterdescendants("input", "textarea") if d not in page.hidden]
        if len(fields) == 1:
            return fields[0]
    cur = el
    for _ in range(4):
        if page.interactive(cur):
            return cur
        cur = cur.getparent()
        if cur is None:
            break
    return el


def history_entry(op, label, text):
    return {"action": label if op != "SELECT" or not text else f"{label} → {_clip(text)}", "kind": KIND[op],
            "text": text if op == "TYPE_TEXT" else None, "page_changed": None}


def _fallback_label(a) -> str:
    try:
        aria = json.loads(a["attributes"] or "{}").get("data", {}).get("aria-label", "")
    except ValueError:
        aria = ""
    return _clip(a["value"] or aria or "element")


def convert_trace(trace, actions, fetch, rng_seed, k_min, k_max, text_min, text_max):
    """-> (rows, skip counts) for one trajectory, in step order."""
    rows, skips, history = [], Counter(), []
    for a in sorted(actions, key=lambda r: r["source_step_index"]):
        op = OP_MAP.get(a["action_type"])
        text = None if (a["input_text"] or NO_TEXT) == NO_TEXT else a["input_text"]
        if op is None or not a["included_in_sft"]:
            skips[f"action:{a['action_type']}"] += 1
            continue
        label, row = _fallback_label(a), None
        if not a["html_dom_url"] or not a["selector"]:
            skips["no_snapshot"] += 1
        else:
            html = fetch(a["html_dom_url"])
            row, why = (None, "fetch_failed") if html is None else convert_step(
                trace, a, op, text, html, history, random.Random(f"{rng_seed}:{trace['uid']}:{a['source_step_index']}"),
                k_min, k_max, text_min, text_max)
            if row is None:
                skips[why] += 1
            else:
                rows.append(row)
                label = row["_meta"]["target_label"]
        history.append(history_entry(op, label, text))
    return rows, skips


def convert_step(trace, a, op, text, html, history, rng, k_min, k_max, text_min, text_max):
    """-> (row, None) or (None, skip_reason)."""
    try:
        page = WebPage(html)
    except Exception:
        return None, "unparseable_html"
    if page.lang() and not page.lang().startswith("en"):
        return None, "not_english"
    hit = page.resolve(a["selector"])
    if hit is None:
        return None, "selector_unresolved"
    gold = gold_element(page, hit, op)
    if op == "SELECT" and gold.tag != "select":
        return None, "select_not_a_select"
    if gold in page.hidden:
        return None, "gold_hidden"
    gold_d = page.describe(gold)
    related = set(gold.iterancestors()) | set(gold.iterdescendants()) | {hit}
    seen, negatives = {gold_d["label"].casefold()}, []
    for el in page.candidates():
        if el is gold or el in related:
            continue
        d = page.describe(el)
        key = d["label"].casefold()
        if key in seen:   # identical labels make the target ambiguous
            continue
        seen.add(key)
        negatives.append((el, d))
    if len(negatives) < k_min - 1:
        return None, "too_few_candidates"
    goal = clean_goal(trace["query"])
    questions, labels, elements, n, gold_target = build_questions(page, gold, gold_d, negatives, op, text or a["value"],
                                                                  goal, rng, k_min, k_max)
    if gold_target is None:
        return None, "select_option_not_found" if op == "SELECT" else "no_target"
    body = page.page_text(rng.randint(text_min, min(text_max, PAGE_TEXT_CHARS)))
    if _cjk_share(body) > 0.05:
        return None, "not_english"
    state = {"page": {"url": a["href"] or "", "title": page.title(), "text": body},
             "elements": elements, "recent_actions": history[-HISTORY_STEPS:]}
    meta = {"source": "webchain", "trace": trace["uid"], "step": a["source_step_index"], "host": trace["primary_host"],
            "web_type": trace["web_type"], "intent_type": trace["intent_type"], "action": a["action_type"], "op": op,
            "target_label": gold_d["label"], "k": n, "has_target": True}
    return {"request": {"model": "wev-latest", "state": state, "questions": questions},
            "labels": labels, "_meta": meta}, None


def make_fetch(cache: Path, retries: int = 2):
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(url):
        path = cache / (hashlib.sha1(url.encode()).hexdigest() + ".html.gz")
        if path.exists():
            return gzip.decompress(path.read_bytes()) or None
        for i in range(retries + 1):
            try:
                body = urllib.request.urlopen(url, timeout=30).read()
                path.write_bytes(gzip.compress(body))
                return body
            except Exception:
                time.sleep(1 + i)
        path.write_bytes(gzip.compress(b""))   # remember dead links
        return None
    return fetch


def load_meta(meta_dir):
    import pyarrow.parquet as pq
    if meta_dir is None:
        from huggingface_hub import hf_hub_download
        files = {n: hf_hub_download(REPO, f"data/seed_sft/metadata/{n}.parquet", repo_type="dataset")
                 for n in ("traces", "actions")}
    else:
        files = {n: f"{meta_dir}/{n}.parquet" for n in ("traces", "actions")}
    traces = pq.read_table(files["traces"], columns=TRACE_COLS).to_pylist()
    actions = defaultdict(list)
    for r in pq.read_table(files["actions"], columns=ACTION_COLS).to_pylist():
        actions[r["trace_uid"]].append(r)
    return traces, actions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", help="directory with WebChain's seed_sft/metadata traces.parquet and actions.parquet "
                                   "(default: download from the Hub)")
    ap.add_argument("--cache", default="data/.webchain-html", help="snapshot cache")
    ap.add_argument("--max_traces", type=int, default=3000, help="English trajectories to convert (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_min", type=int, default=8)
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--text_min", type=int, default=1500)
    ap.add_argument("--text_max", type=int, default=PAGE_TEXT_CHARS)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        ap.error(f"refusing to overwrite {out}")
    traces, actions = load_meta(a.meta)
    english = [t for t in traces if t["query"] and _cjk_share(t["query"]) == 0 and t["uid"] in actions]
    english.sort(key=lambda t: hashlib.sha256(f"{a.seed}:{t['uid']}".encode()).hexdigest())
    chosen = english[: a.max_traces] if a.max_traces else english
    fetch = make_fetch(Path(a.cache))

    def work(t):
        return t, *convert_trace(t, actions[t["uid"]], fetch, a.seed, a.k_min, a.k_max, a.text_min, a.text_max)

    out.mkdir(parents=True)
    files = {s: open(out / f"{s}.jsonl", "w") for s in ("train", "dev", "test")}
    counts, skips, ops, hosts, ks, done = Counter(), Counter(), Counter(), {s: set() for s in files}, Counter(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for t, rows, sk in ex.map(work, chosen):
            split = split_of(t["primary_host"], a.seed, a.dev_frac, a.test_frac)
            skips.update(sk)
            if rows:
                hosts[split].add(t["primary_host"])
            for r in rows:
                files[split].write(json.dumps(r, ensure_ascii=False) + "\n")
                counts[split] += 1
                ops[(split, r["_meta"]["op"])] += 1
                ks[split] += r["_meta"]["k"]
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(chosen)} traces  {dict(counts)}  skipped {sum(skips.values())}", flush=True)
    for f in files.values():
        f.close()
    manifest = {"source": f"{REPO} (seed_sft metadata + per-step HTML snapshots)", "args": vars(a),
                "traces": {"total": len(traces), "english": len(english), "converted": len(chosen)},
                "rows": dict(counts), "hosts": {s: len(h) for s, h in hosts.items()},
                "host_lists": {s: sorted(h) for s, h in hosts.items()},
                "ops": {f"{s}:{o}": n for (s, o), n in sorted(ops.items())},
                "mean_k": {s: round(ks[s] / counts[s], 1) for s in counts if counts[s]},
                "skipped": dict(skips.most_common()),
                "sha256": {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in files}}
    try:
        from huggingface_hub import HfApi
        manifest["dataset_revision"] = HfApi().dataset_info(REPO).sha
    except Exception as e:
        manifest["dataset_revision"] = f"unknown ({type(e).__name__})"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("traces", "rows", "hosts", "mean_k", "skipped", "ops")}, indent=2))


if __name__ == "__main__":
    main()
