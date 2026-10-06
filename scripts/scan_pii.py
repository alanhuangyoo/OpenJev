"""PII and secret scan of a built wev-data folder (scripts/build_hf_dataset.py output). Reports, never rewrites.

Fields scanned per step row:
  typed       text the agent typed: recent_actions[].text and the current values of editable elements
  url         page URL (query strings carry tokens and e-mail addresses)
  page_text   visible page text
  labels      element labels, action history labels, teacher reason
  goal        the task goal
Categories: email, phone, card (Luhn-valid card or account number), ssn, secret (API-key, JWT and private-key
shapes; credential-like URL parameters), password_value (a non-empty value in a password field) and name (a typed
"First Last" whose first token is a common first name). A typed value that also occurs in the goal was dictated by
the task (Mind2Web and NNetNav goals carry made-up contact details), not leaked by a user; it is counted separately
as <category>_from_goal.

The report lists counts per config, field and category, redacted examples, the row ids per category, and the
redaction rule proposed for them (see REDACT).
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
PHONE = re.compile(r"(?<![\w+.])(?:\+?1[\s.-])?(?:\(\d{3}\)\s?\d{3}[-.]\d{4}|\d{3}([-.])\d{3}\1\d{4})(?![\d.])"
                   r"|(?<![\w+])\+\d{1,3}(?:[\s.-]?\d{2,4}){3,5}(?![\d.])")
CARD = re.compile(r"(?<![\d.,-])(?:\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}|\d{13,19})(?![\d.,])")
SSN = re.compile(r"(?<![\d.-])\d{3}-\d{2}-\d{4}(?![\d-])")
NOT_SSN = re.compile(r"DPCI|item|model|part|sku|order", re.I)
CARD_CONTEXT = re.compile(r"card|account|member|family id|visa|mastercard|amex|discover|iban|loyalty", re.I)
YEAR = re.compile(r"(19|20)\d\d")
SECRET = re.compile(r"AKIA[0-9A-Z]{16}|\bsk-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{36}|\bhf_[A-Za-z0-9]{30,}"
                    r"|\bxox[abpr]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{35}"
                    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
                    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")
URL_SECRET = re.compile(r"[?&#](?:access_token|id_token|auth|token|api_?key|apikey|secret|password|passwd|pwd|"
                        r"session_?id|"
                        r"sid|sig|signature|code)=([^&#\s]{12,})", re.I)
FIRST_NAMES = set("""james john robert michael william david richard joseph thomas charles christopher daniel matthew
anthony mark donald steven paul andrew joshua kenneth kevin brian george timothy ronald edward jason jeffrey ryan jacob
gary nicholas eric jonathan stephen larry justin scott brandon benjamin samuel gregory alexander frank patrick raymond
jack dennis jerry tyler aaron jose adam nathan henry douglas zachary peter kyle noah ethan jeremy walter christian keith
roger terry austin sean gerald carl harold dylan arthur lawrence jordan jesse bryan billy bruce gabriel joe logan alan
juan albert willie elijah wayne randy vincent mason roy ralph bobby russell bradley philip eugene mary patricia jennifer
linda elizabeth barbara susan jessica sarah karen lisa nancy betty sandra margaret ashley kimberly emily donna michelle
carol amanda melissa deborah stephanie dorothy rebecca sharon laura cynthia amy kathleen angela shirley brenda emma anna
pamela nicole samantha katherine christine helen debra rachel carolyn janet maria catherine heather diane olivia julie
joyce victoria ruth virginia lauren kelly christina joan evelyn judith andrea hannah megan cheryl jacqueline martha
madison teresa gloria sara janice ann kathryn abigail sophia frances jean alice judy isabella julia grace amber denise
danielle marilyn beverly charlotte natalie theresa diana brittany doris kayla alexis lori marie alex sam chris jane
tom tim mike jim bob bill ben dan max leo lucas liam oliver ella mia ava chloe lily zoe""".split())
NAME = re.compile(r"^([A-Z][a-z]+)(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?$")
REDACT = {
    "email": "typed/url/labels: replace with user@example.com; page_text: replace with <email>",
    "phone": "typed: replace with 555-0100; url: drop the parameter; page_text and labels: keep (published business "
             "numbers on public pages) unless the row is also flagged for a typed value",
    "card": "drop the row",
    "ssn": "drop the row",
    "secret": "replace the value with <redacted>",
    "password_value": "replace the value with <redacted>",
    "name": "typed: replace with a fixed placeholder name (e.g. Alex Doe) in recent_actions and element values",
    "*_from_goal": "keep: the value is part of the task's own goal",
}


def luhn(digits: str) -> bool:
    d = [int(c) for c in digits][::-1]
    return sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2) for i, x in enumerate(d)) % 10 == 0


def mask(s: str) -> str:
    """Keep the shape, hide the content: letters -> x, digits -> 0 (except the last two), keep @ . - + space."""
    s = s[:80]
    out = [("x" if c.isalpha() else "0" if c.isdigit() else c) for c in s]
    digits = [i for i, c in enumerate(s) if c.isdigit()]
    for i in digits[-2:]:
        out[i] = s[i]
    return "".join(out)


def context(text: str, start: int, end: int, width: int = 30) -> str:
    left, right = text[max(0, start - width):start], text[end:end + width]
    return " ".join(f"{left}[{mask(text[start:end])}]{right}".split())


def find(text: str, field: str):
    """-> [(category, start, end)]"""
    hits = [("email", m.start(), m.end()) for m in EMAIL.finditer(text)]
    hits += [("phone", m.start(), m.end()) for m in PHONE.finditer(text)]
    hits += [("ssn", m.start(), m.end()) for m in SSN.finditer(text)
             if not NOT_SSN.search(text[max(0, m.start() - 30):m.start()])]
    if field != "url":   # long numeric ids in URLs are not card numbers
        hits += [("card", m.start(), m.end()) for m in CARD.finditer(text)
                 if 13 <= len(d := re.sub(r"\D", "", m.group())) <= 19 and luhn(d) and len(set(d)) > 1
                 and not all(YEAR.fullmatch(g) for g in re.split(r"[ -]", m.group()))
                 and (m.group(1) or CARD_CONTEXT.search(text[max(0, m.start() - 40):m.start()]))]
    hits += [("secret", m.start(), m.end()) for m in SECRET.finditer(text)]
    if field == "url":
        hits += [("secret", m.start(1), m.end(1)) for m in URL_SECRET.finditer(text)]
    return hits


def fields_of(row):
    """-> [(field, text, is_typed_value, element label)]"""
    req = json.loads(row["request"])
    st, out = req["state"], []
    page = st.get("page") or {}
    out.append(("url", page.get("url") or "", False, ""))
    out.append(("page_text", page.get("text") or "", False, ""))
    for a in st.get("recent_actions") or []:
        if a.get("text"):
            out.append(("typed", str(a["text"]), True, str(a.get("action") or "")))
        out.append(("labels", str(a.get("action") or ""), False, ""))
    for e in st.get("elements") or []:
        out.append(("labels", str(e.get("label") or ""), False, ""))
        if e.get("value") and "SELECT" not in (e.get("operations") or []):
            out.append(("typed", str(e["value"]), True, str(e.get("label") or "")))
    out.append(("goal", row.get("goal") or "", False, ""))
    if row.get("reason"):
        out.append(("labels", row["reason"], False, ""))
    return out


def scan_row(row):
    goal = (row.get("goal") or "").casefold()
    for field, text, typed, label in fields_of(row):
        for cat, s, e in find(text, field):
            yield field, f"{cat}_from_goal" if typed and text[s:e].casefold() in goal else cat, context(text, s, e)
        if typed and re.search(r"pass(word|code)|pwd|\bpin\b", label, re.I) and text.strip():
            yield field, "password_value", f"{label[:40]}: [{mask(text)}]"
        if typed and (m := NAME.match(text.strip())) and m.group(1).lower() in FIRST_NAMES:
            cat = "name_from_goal" if text.strip().casefold() in goal else "name"
            yield field, cat, f"{label[:40]}: [{mask(text)}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="build_hf_dataset.py output folder")
    ap.add_argument("--out", required=True, help="report JSON")
    ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()
    counts = defaultdict(Counter)        # config -> "field:category" -> matches
    rows_hit = defaultdict(Counter)      # config -> category -> rows
    examples = defaultdict(list)         # "field:category" -> redacted examples
    ids = defaultdict(set)               # category -> row ids
    total = Counter()
    for path in sorted(Path(a.data).glob("*/*.parquet")):
        config, split = path.parent.name, path.stem
        if config == "bench":   # a copy of test rows of other configs
            continue
        for row in pq.read_table(path).to_pylist():
            total[config] += 1
            if config == "live_tasks":
                row = {"id": row["id"], "goal": row["goal"],
                       "request": json.dumps({"state": {"page": {"url": row["url"]}}})}
            cats = set()
            for field, cat, ex in scan_row(row):
                counts[config][f"{field}:{cat}"] += 1
                cats.add(cat)
                if len(examples[f"{field}:{cat}"]) < a.examples:
                    examples[f"{field}:{cat}"].append({"config": config, "split": split, "id": row["id"],
                                                       "context": ex})
                ids[cat].add(f"{config}/{split}/{row['id']}")
            for c in cats:
                rows_hit[config][c] += 1
    report = {"rows_scanned": dict(total), "matches": {c: dict(v.most_common()) for c, v in counts.items()},
              "rows_flagged": {c: dict(v.most_common()) for c, v in rows_hit.items()},
              "examples_redacted": dict(sorted(examples.items())), "proposed_redaction": REDACT,
              "row_ids": {c: sorted(v) for c, v in ids.items()}}
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("rows_scanned", "rows_flagged")}, indent=2))


if __name__ == "__main__":
    main()
