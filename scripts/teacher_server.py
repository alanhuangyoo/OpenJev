"""A System One compatible server backed by an LLM teacher, which logs every decision as a labelled training row.

jev-ultrafast (or any System One client) points TYPESAFE_BASE_URL here. For each request the teacher picks one
operation and, if that operation has a target question, one target. The answers returned put 0.97 on the teacher's
choice, so the client acts on it. Each request is logged with its labels, keyed by the client's bearer token, which
the collector sets to a per-episode id; episodes are filtered for success afterwards.

Credentials come from the environment only: WEV_JUDGE_BASE_URL, WEV_JUDGE_API_KEY; TEACHER_MODEL (default qwen3-max);
TEACHER_EFFORT (low | high | max) for models that cannot switch thinking off.

    python scripts/teacher_server.py --log teacher.jsonl --port 8010
"""
import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from fastapi import FastAPI, Request

app = FastAPI()
LOCK = threading.Lock()
CFG: dict = {}

SAFETY = """Safety rules, which override the goal:
- Never sign in, register, buy, book, pay, post, comment, send messages, or submit personal information.
- If the next step would need any of those, a CAPTCHA, an "unusual traffic" page, or a login wall: choose BLOCKED.
- If the goal is already visibly satisfied: choose DONE. Do not keep browsing after that."""


def render(body: dict) -> str:
    st, qs = body["state"], body["questions"]
    op_q = qs["operation"]
    goal = op_q["instructions"]["goal"] if isinstance(op_q["instructions"], dict) else op_q["instructions"]
    rules = op_q["instructions"].get("rules", "") if isinstance(op_q["instructions"], dict) else ""
    page = st.get("page", {})
    els = []
    for e in st.get("elements", []):
        extra = "".join(f" {k}={e[k]!r}" for k in ("value", "checked", "selected", "expanded") if e.get(k) not in (None, ""))
        els.append(f"[{e.get('index')}] {e.get('role')} {e.get('label')!r}{extra} ops={','.join(e.get('operations', []))}")
        for o in e.get("options", []) or []:
            els.append(f"    option [{o['index']}] {o['label']!r}")
    hist = [f"- {h.get('kind')}: {h.get('action')}" + (f" = {h['text']!r}" if h.get("text") else "") +
            ("" if h.get("page_changed") is None else f" (page changed: {h['page_changed']})")
            for h in st.get("recent_actions", [])]
    ops = [f"- {k}: {v}" for k, v in op_q["criteria"].items()]
    targets = []
    for qid, q in qs.items():
        if qid.endswith("_target"):
            keys = list(q["criteria"])
            targets.append(f"- {qid[:-7].upper()} target keys: {', '.join(keys[:300])}")
    return (f"GOAL: {goal}\n\nURL: {page.get('url', '')[:300]}\nTITLE: {page.get('title', '')}\n\n"
            f"VISIBLE PAGE TEXT:\n{page.get('text', '')[:6000]}\n\nELEMENTS:\n" + "\n".join(els) +
            "\n\nRECENT ACTIONS:\n" + ("\n".join(hist) or "- (none)") +
            "\n\nOPERATIONS:\n" + "\n".join(ops) + "\n\nVALID TARGETS PER OPERATION:\n" + ("\n".join(targets) or "- (none)") +
            f"\n\nRULES:\n{rules}\n\n{SAFETY}\n\n"
            'Reply with a JSON object only: {"operation": "<one OPERATIONS key>", '
            '"target": "<a valid target key for that operation, or null>", "reason": "<one short sentence>"}')


def llm(messages):
    body = {"model": CFG["model"], "messages": messages, "temperature": 0, "max_tokens": 300,
            "response_format": {"type": "json_object"}, "enable_thinking": False}
    if os.environ.get("TEACHER_EFFORT"):   # models that always think (GLM) take a reasoning level, not enable_thinking
        body.pop("enable_thinking")
        body.update(max_tokens=2000, reasoning_effort=os.environ["TEACHER_EFFORT"])
    body = json.dumps(body).encode()
    for attempt in range(8):
        req = urllib.request.Request(CFG["base"].rstrip("/") + "/chat/completions", data=body,
                                     headers={"content-type": "application/json", "authorization": f"Bearer {CFG['key']}"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=90))["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30) if not isinstance(e, urllib.error.HTTPError) or e.code in (429, 500, 502, 503) else 1)


def decide(body: dict):
    qs = body["questions"]
    ops = list(qs["operation"]["criteria"])
    messages = [{"role": "system", "content": "You control a web browser to accomplish the user's goal, one step at a time."},
                {"role": "user", "content": render(body)}]
    for _ in range(3):
        text = llm(messages)
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        op, target = d.get("operation"), d.get("target")
        tq = f"{str(op).lower()}_target"
        if op in ops and (tq not in qs or str(target) in qs[tq]["criteria"]):
            return op, (str(target) if tq in qs else None), str(d.get("reason", ""))[:300]
        problem = (f"operation must be one of {ops}" if op not in ops else
                   f"target must be one of the valid keys for {op}, got {target!r}")
        messages += [{"role": "assistant", "content": text or ""}, {"role": "user", "content": f"Invalid: {problem}. Reply again."}]
    return ("BLOCKED" if "BLOCKED" in ops else ops[0]), None, "teacher gave no valid answer"


def one_hot(keys, chosen):
    k = len(keys)
    if k == 1:
        return {keys[0]: 1.0}
    rest = round(0.03 / (k - 1), 6)
    probs = {key: (round(1 - rest * (k - 1), 6) if key == chosen else rest) for key in keys}
    return probs


@app.get("/v1/models")
def models():
    return {"data": [{"id": "teacher-" + CFG["model"]}]}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    session = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or "anon"
    t = time.time()
    import asyncio
    op, target, reason = await asyncio.to_thread(decide, body)
    answers = {}
    for qid, q in body["questions"].items():
        keys = list(q["criteria"])
        chosen = op if qid == "operation" else (target if qid == f"{op.lower()}_target" else keys[0])
        probs = one_hot(keys, chosen)
        k = len(keys)
        conf = 1.0 if k == 1 else round((max(probs.values()) - 1 / k) / (1 - 1 / k), 6)
        answers[qid] = {"type": "choice", "choice": chosen, "confidence": conf, "probabilities": probs}
    labels = {"operation": op}
    if target is not None:
        labels[f"{op.lower()}_target"] = target
    row = {"session": session, "t": t, "request": body, "labels": labels, "reason": reason, "teacher": CFG["model"],
           "latency_ms": round((time.time() - t) * 1000)}
    with LOCK, open(CFG["log"], "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"model": "teacher-" + CFG["model"], "answers": answers, "usage": {}, "latency_ms": row["latency_ms"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    CFG.update(base=os.environ["WEV_JUDGE_BASE_URL"], key=os.environ["WEV_JUDGE_API_KEY"],
               model=os.environ.get("TEACHER_MODEL", "qwen3-max"), log=a.log)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
