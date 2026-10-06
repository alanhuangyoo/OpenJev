"""Redaction applied to every row before it is published (scripts/build_hf_dataset.py), using scan_pii.py's detectors.

- a card or social security number that the task goal does not dictate: the row is dropped
- secrets (API keys, JWTs, private keys) anywhere, and credential-like URL parameters: replaced with <redacted>
- a value typed into a password or PIN field: <redacted>
- a person's name, e-mail or phone typed by the agent but not given in the goal: Alex Doe, user@example.com, 555-0100
- e-mail addresses elsewhere (page text, labels, URLs): <email>; published phone numbers on pages are kept
- a step labelled DONE on a page with no elements and almost no text (a page that had not loaded): dropped
"""
import re

from scan_pii import EMAIL, FIRST_NAMES, NAME, PHONE, SECRET, URL_SECRET, find

PASSWORD_LABEL = re.compile(r"pass(word|code)|pwd|\bpin\b", re.I)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _rewrite(obj, fn):
    """Apply fn to every string value (dict keys, i.e. option keys, stay as they are)."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _rewrite(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v, fn) for v in obj]
    return obj


def unloaded_done(row) -> bool:
    st = row["request"]["state"]
    if not isinstance(st, dict) or row["labels"].get("operation") != "DONE":
        return False
    return not st.get("elements") and len((st.get("page") or {}).get("text") or "") < 200


def redact_row(row):
    """-> the row with sensitive values replaced, or None when it must be dropped."""
    req = row["request"]
    instr = (req.get("questions", {}).get("operation") or {}).get("instructions")
    goal = (instr.get("goal", "") if isinstance(instr, dict) else str(instr or "")).casefold()
    for text in _strings(req):
        if any(cat in ("card", "ssn") and text[s:e].casefold() not in goal for cat, s, e in find(text, "page_text")):
            return None
    st = req.get("state") if isinstance(req.get("state"), dict) else {}
    typed = [(str(a["text"]), str(a.get("action") or "")) for a in st.get("recent_actions") or [] if a.get("text")]
    typed += [(str(e["value"]), str(e.get("label") or "")) for e in st.get("elements") or []
              if e.get("value") and "SELECT" not in (e.get("operations") or [])]
    exact = {}
    for value, label in typed:
        v = value.strip()
        if not v or v.casefold() in goal:
            continue
        if PASSWORD_LABEL.search(label):
            exact[value] = "<redacted>"
        elif (m := NAME.match(v)) and m.group(1).lower() in FIRST_NAMES:
            exact[value] = "Alex Doe"
        else:
            for m in EMAIL.finditer(value):
                if m.group().casefold() not in goal:
                    exact[m.group()] = "user@example.com"
            for m in PHONE.finditer(value):
                if m.group().casefold() not in goal:
                    exact[m.group()] = "555-0100"
    order = sorted(exact, key=len, reverse=True)

    def fix(text):
        for old in order:
            text = text.replace(old, exact[old])
        text = SECRET.sub("<redacted>", text)
        text = URL_SECRET.sub(lambda m: m.group(0)[: m.start(1) - m.start(0)] + "<redacted>", text)
        return EMAIL.sub(lambda m: m.group() if m.group().casefold() in goal or m.group() == "user@example.com"
                         else "<email>", text)

    out = {**row, "request": _rewrite(req, fix)}
    if "_meta" in row:
        out["_meta"] = _rewrite(row["_meta"], fix)
    return out
