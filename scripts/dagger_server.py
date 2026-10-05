"""On-policy collection server: the student and the LLM teacher answer every request; one of them acts, the teacher's
answer is logged as the label (DAgger with a mixed policy).

Per request the student (a wev System One endpoint) acts with probability --beta and the teacher otherwise, so
episodes visit the states the student reaches while the teacher keeps them on track. Every row records the teacher's
choice as `labels` and the acted choice as `executed`; build_teacher_data.py --onpolicy turns the log into training
rows. Teacher credentials and model as in teacher_server.py (WEV_JUDGE_BASE_URL, WEV_JUDGE_API_KEY, TEACHER_MODEL,
TEACHER_EFFORT).

    python scripts/dagger_server.py --student http://127.0.0.1:8014 --log dagger.jsonl --port 8018 --beta 0.5
"""
import argparse
import asyncio
import json
import os
import random
import time
import urllib.request

from fastapi import FastAPI, Request

import teacher_server as ts

app = FastAPI()


def student_answers(body: dict) -> dict:
    req = urllib.request.Request(ts.CFG["student"].rstrip("/") + "/v1/systemone", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["answers"]


def teacher_answers(body: dict, op: str, target: str | None) -> dict:
    answers = {}
    for qid, q in body["questions"].items():
        keys = list(q["criteria"])
        chosen = op if qid == "operation" else (target if qid == f"{op.lower()}_target" else keys[0])
        probs = ts.one_hot(keys, chosen)
        k = len(keys)
        conf = 1.0 if k == 1 else round((max(probs.values()) - 1 / k) / (1 - 1 / k), 6)
        answers[qid] = {"type": "choice", "choice": chosen, "confidence": conf, "probabilities": probs}
    return answers


def choice_of(answers: dict) -> dict:
    op = answers["operation"]["choice"]
    target = answers.get(f"{op.lower()}_target", {}).get("choice")
    return {"operation": op, **({f"{op.lower()}_target": target} if target is not None else {})}


@app.get("/v1/models")
def models():
    return {"data": [{"id": "dagger-" + ts.CFG["model"]}]}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    session = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or "anon"
    t = time.time()
    (op, target, reason), student = await asyncio.gather(asyncio.to_thread(ts.decide, body),
                                                         asyncio.to_thread(student_answers, body))
    labels = {"operation": op, **({f"{op.lower()}_target": target} if target is not None else {})}
    actor = "student" if random.random() < ts.CFG["beta"] else "teacher"
    answers = student if actor == "student" else teacher_answers(body, op, target)
    row = {"session": session, "t": t, "request": body, "labels": labels, "reason": reason, "teacher": ts.CFG["model"],
           "actor": actor, "executed": choice_of(answers), "student": choice_of(student),
           "latency_ms": round((time.time() - t) * 1000)}
    with ts.LOCK, open(ts.CFG["log"], "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"model": "dagger-" + ts.CFG["model"], "answers": answers, "usage": {}, "latency_ms": row["latency_ms"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="base URL of the student's System One endpoint")
    ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--beta", type=float, default=0.5, help="probability that the student acts on a request")
    a = ap.parse_args()
    ts.CFG.update(base=os.environ["WEV_JUDGE_BASE_URL"], key=os.environ["WEV_JUDGE_API_KEY"],
                  model=os.environ.get("TEACHER_MODEL", "qwen3-max"), log=a.log, student=a.student, beta=a.beta)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
