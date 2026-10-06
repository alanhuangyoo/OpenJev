<div align="center">

# wev

**OpenJev: the open-source, local alternative to Jev.** Models on the Hub as `OpenJev-1.7B / 4B / 8B`.<br>
**Local System-One decision models for agents.**<br>
Typed questions in, calibrated probabilities out, in one forward pass. No text generation, no API key.

[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-1.7B%20%7C%204B%20%7C%208B-ffcc4d)](https://huggingface.co/collections/alanhuangya/wev-6ab4eb5d872c9ae9fd68faa6)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-OpenJev--Data-ffcc4d)](https://huggingface.co/datasets/alanhuangya/OpenJev-Data)
[![pip](https://img.shields.io/badge/pip%20install-wev--ai-3775a9?logo=pypi&logoColor=white)](https://pypi.org/project/wev-ai/)
[![Project page](https://img.shields.io/badge/%F0%9F%A4%97%20Project-page-ffcc4d)](https://huggingface.co/spaces/alanhuangya/wev)
[![Paper](https://img.shields.io/badge/Paper-Zenodo-b31b1b)](https://zenodo.org/records/22941164)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22941164.svg)](https://zenodo.org/records/22941164)
[![License](https://img.shields.io/badge/License-Apache--2.0-2ea44f)](https://github.com/alanhuangyoo/wev/blob/main/LICENSE)

[Quickstart](#quickstart) · [Results](#results) · [How it works](#how-it-works) · [Train your own](#train-your-own) · [Citation](#citation)

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/alanhuangyoo/wev/main/assets/overview.png" width="88%" alt="Interface distillation: a browser agent's typed requests are answered by a teacher LLM on live websites, an LLM judge keeps the verified episodes, and the wev decision model trained on them serves the same interface locally.">
</p>

`wev` answers the `POST /v1/systemone` request shape: a free-form state plus any number of questions, each a
**choice**, a **yes/no** or a **score**. It handles general decisions such as triage, routing, policy checks and
agent monitoring, and it handles browser-agent steps: *which operation?* and *which element?* It runs on your own
GPU, or on a laptop.

**New in v0.2 (OpenJev-4B):** live-website tasks completed rise from 30 to 39–41 of 153, out-of-domain general accuracy
from 73.8 to 81.4, and browser steps on unseen websites from 75.9 to 78.8; probabilities are calibrated (ECE 0.01–0.04)
and a browser step takes 184 ms. OpenJev-8B and OpenJev-1.7B are still v0.1.

## Highlights

- 🌐 **Browser steps on unseen websites: 79% right**, where open decision models trained on general data reach at
  most 21%.
- 🧠 **General decisions stay strong.** OpenJev-4B scores 88.1 on Kev decision-v7 and 81.4 out of domain, and wev beats
  Laya on every general benchmark.
- 🎓 **Better than its teacher, locally.** As the System One of an open browser agent, OpenJev-4B completes 39–41 of 153
  held-out live tasks, against 27 for the qwen3-max teacher.
- ⚡ **Fast.** 10–40 ms for a general decision and 91–322 ms for a full browser page on one RTX 5090; 77 ms on a
  laptop.
- 📏 **Calibrated.** Probabilities you can threshold: raise the bar for DONE and early stops drop sharply.

## Models

| Model | Size | Best for | General decision | Browser step |
|---|---|---|---|---|
| [**OpenJev-4B**](https://huggingface.co/alanhuangya/OpenJev-4B) (v0.2) | 8.4 GB | the default: one consumer GPU, best on live websites and out of domain | 40 ms | 184 ms |
| [OpenJev-8B](https://huggingface.co/alanhuangya/OpenJev-8B) (v0.1) | 15.2 GB | a larger v0.1 model | 21 ms | 322 ms |
| [OpenJev-1.7B](https://huggingface.co/alanhuangya/OpenJev-1.7B) (v0.1) | 3.5 GB | laptops and small GPUs; the fastest | 10 ms | 91 ms |

*Median latency on one RTX 5090 (bf16), one request at a time. OpenJev-4B v0.1 stays available as the `v0.1` revision:
`wev.load("alanhuangya/OpenJev-4B", revision="v0.1")`. OpenJev-4B v0.2 runs on a Qwen3.5 backbone and needs transformers 5;
`pip install "wev-ai[serve,fast]"` adds its fast kernels on CUDA.*

## Quickstart

```bash
pip install "wev-ai[serve]"
```

```python
import wev

m = wev.load("alanhuangya/OpenJev-4B")   # downloads once from the Hugging Face Hub
out = m.predict(
    state="Refund request: order #4411 arrived damaged, customer attached photos, first refund this year.",
    questions={
        "action": {"type": "choice", "instructions": "What should support do?",
                   "criteria": {"refund": "Refund the order.", "replace": "Ship a replacement.",
                                "escalate": "Send to a human agent."}},
        "fraud_risk": {"type": "noul", "instructions": "This request looks fraudulent.",
                       "criteria": {"true": "Likely fraud.", "false": "No sign of fraud."}},
    },
)
out["answers"]
# {'action': {'type': 'choice', 'choice': 'replace', 'probabilities': {'refund': 0.38, 'replace': 0.51, ...}},
#  'fraud_risk': {'type': 'noul', 'noul': 0.02}}
```

**Serve it over HTTP** with the same request and response shapes as a System One API:

```bash
wev serve --model alanhuangya/OpenJev-4B --port 8009
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{"state": "...", "questions": {...}}'
```

**Use it in a browser agent.** An agent written for a System One API, such as
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast), can call `http://127.0.0.1:8009/v1/systemone`
instead of the hosted endpoint. jev-ultrafast hard-codes that URL in `jev_ultrafast/model.py` (`SYSTEM_ONE_URL`), so
change that one line.

## Results

<p align="center"><img src="https://raw.githubusercontent.com/alanhuangyoo/wev/main/assets/teaser.png" width="52%" alt="Browser step success against out-of-domain general accuracy for wev, Kev and Laya."></p>

Every model below receives the same requests and is scored the same way: per-question accuracy, with each model's
most probable option taken as its answer. All numbers are on held-out test splits.

**General typed decisions**

| Model | Kev decision-v7 | Kev transfer-v4 | typed-decisions |
|---|:---:|:---:|:---:|
| **OpenJev-4B** (v0.2) | 88.1 | 81.4 | 78.8 |
| **OpenJev-8B** | 82.4 | 77.2 | 79.1 |
| **OpenJev-1.7B** | 81.1 | 65.5 | **79.5** |
| Kev-4B | **88.2** | **82.1** | 65.1 |
| Kev-8B | 88.1 | 76.8 | 62.7 |
| Laya (typed-decisions) | 65.7 | 62.8 | 76.8 |
| Laya | 64.3 | 63.7 | 36.2 |

**Browser steps.** Mind2Web test split: 873 requests on websites unseen in training. A step counts when both the
operation and the target element are right.

| Model | Step success | Operation |
|---|:---:|:---:|
| **OpenJev-4B** (v0.2) | **78.8** | **92.0** |
| **OpenJev-8B** | 75.5 | 90.3 |
| **OpenJev-1.7B** | 68.2 | 88.1 |
| Kev-4B | 21.2 | 35.7 |
| Kev-8B | 19.0 | 73.3 |
| Laya (typed-decisions) | 0.7 | 13.1 |
| Laya | 0.0 | 2.5 |

**End to end on live websites.** jev-ultrafast ran 153 held-out tasks with each model as its System One. A task
succeeds when the agent says DONE and an LLM judge, reading the final page, agrees.

| System One | Tasks completed |
|---|:---:|
| OpenJev-4B (v0.2) | 39–41 / 153 |
| OpenJev-4B (v0.1) | 30–31 / 153 |
| OpenJev-8B | 28 / 153 |
| qwen3-max, prompted (teacher of v0.1) | 27 / 153 |
| GLM-5.3-Flash, prompted (teacher added in v0.2) | 47 / 153 |

*Ranges are two passes of the LLM judge over the same episodes.*

<details>
<summary><b>Notes on the comparison</b></summary>

- *Kev decision-v7* is Kev's own training suite; wev also trains on its train split, and Kev leads there by 6–8
  points. *Kev transfer-v4* is out of domain for every model in the table.
- wev and Laya (typed-decisions) train on 80% of the typed-decisions train split; Kev and plain Laya do not, so on
  that column they are generalists.
- Kev and Laya ran as published, without web training data. wev is evaluated with a 4,096-token state; the 11
  Mind2Web requests beyond it count as wrong for wev.
- Live sites change from run to run, so treat end-to-end gaps of a few tasks as noise. OpenJev-4B's gain from its second
  teacher collection held on a paired comparison (10 tasks gained, 1 lost).
- Test splits were held out from training and model selection, with one exception: OpenJev-4B v0.1 and OpenJev-8B each had
  two candidates, and both were read on test. OpenJev-4B v0.2 was read on test once (and its calibrated export once more;
  the calibration is fitted on development rows and changes no answer).
- OpenJev-4B v0.2 continues training from the Kev-4B checkpoint (Apache-2.0); see its model card.
- Raw results are in [`results/`](https://github.com/alanhuangyoo/wev/tree/main/results).

</details>

<details>
<summary><b>Calibration: choosing when to stop</b></summary>

<p align="center"><img src="https://raw.githubusercontent.com/alanhuangyoo/wev/main/assets/stopping.png" width="55%" alt="DONE recall against premature DONE rate as the threshold on the DONE probability is swept."></p>

The hardest browser decision is when to stop. Accepting DONE only above a probability threshold trades missed stops
for early ones: at 0.8, OpenJev-4B v0.1 stops early on 3.8% of unfinished steps (8.5% at 0.5), and OpenJev-8B on 1.7%.
OpenJev-4B v0.2 stops early on 3.3% of wev-bench's unfinished steps at its default argmax.

</details>

## wev-bench

A frozen browser lane for decision models: 2,346 steps (Mind2Web test on unseen websites, audited NNetNav test with
its stopping and giving-up steps, and teacher steps on tasks in no train split), stored as the `bench` config of
[`alanhuangya/OpenJev-Data`](https://huggingface.co/datasets/alanhuangya/OpenJev-Data). `scripts/wev_bench.py` sends each
stored request to any `POST /v1/systemone` endpoint and reports step success, operation and target accuracy, the
premature-DONE rate, ECE and p50/p95 latency, overall and per subset. It needs only the standard library and
`pyarrow`.

```bash
wev serve --model alanhuangya/OpenJev-4B --port 8009          # or any System One server
python scripts/wev_bench.py --url http://127.0.0.1:8009/v1/systemone --name wev-4b \
  --hardware "RTX 5090, bf16" --out results/wev-bench/wev-4b-v0.2.json
```

Add `--revision <commit>` to pin the dataset, `--api_key_env NAME` to send a bearer token from an environment
variable, and `--steps_out steps.jsonl` to keep every prediction. Scoring follows
[JevBench](https://github.com/fstandhartinger/jevbench) (argmax accuracy, top-label ECE over 10 bins, multi-class
Brier, interpolated latency percentiles; failed or malformed answers count as wrong), and the output file uses its
per-system layout with a `browser` lane block, so it can be submitted to a JevBench-style leaderboard as is.

| System One | Step success | Operation | Target | Premature DONE | ECE | p50 / p95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **OpenJev-4B v0.2** | **73.1** | **84.7** | **81.4** | **3.3** | **0.013** | **153 / 353 ms** |
| OpenJev-4B v0.1 | 68.0 | 81.3 | 79.0 | 5.0 | 0.051 | 166 / 409 ms |

One RTX 5090 (bf16), queried serially from the same host;
[`results/wev-bench/`](https://github.com/alanhuangyoo/wev/tree/main/results/wev-bench) has the full results.

## How it works

**Interface distillation.** An agent talks to its System One through typed requests. We serve a prompted LLM behind
that interface while the agent works on live websites, so every logged request and answer is already a training
example in the decision model's own format. An LLM judge reads each episode's final page and keeps only the episodes
whose outcome it confirms. It rejected 36% of the teacher's own DONE claims.

**Data.** We combine those episodes with Mind2Web and NNetNav steps converted to the same request format, and with
general typed-decision corpora. An LLM-judge audit of NNetNav's stopping labels overturned 31% of its DONE labels.

**Model.** A Qwen3 base model with its vocabulary head removed, adapted with LoRA and merged into the released
weights. Each question sees the state and itself only, so one forward pass answers every question in a request, and
question order never changes an answer. A pointer head scores each option against the question and normalises the
scores into probabilities.

```
<state> page text · elements · recent actions
<q> which operation? <opt> CLICK </opt> <opt> TYPE_TEXT </opt> … <opt> DONE </opt> <decide>
<q> which element to CLICK? <opt> [1] Search </opt> <opt> [2] Sign in </opt> … <decide>
```

The [paper](https://zenodo.org/records/22941164) has the full method, ablations and failure analysis.

## Train your own

<details>
<summary><b>Data, training, evaluation and export commands</b></summary>

```bash
pip install -e ".[train,serve,dev]"

# data
python -m wev.mind2web --out data/m2w-v2 --dev_frac 0.15 --test_frac 0.1
python -m wev.nnetnav  --out data/nnetnav-v2
python -m wev.general  --out data/general --kev_repo path/to/kev
python -m wev.external --out data/ext
python -m wev.webchain --out data/webchain-v1 --max_traces 4000   # downloads WebChain's per-step HTML snapshots
# DONE relabelling of NNetNav by an LLM judge (OpenAI-compatible endpoint from WEV_JUDGE_BASE_URL / _API_KEY / _MODEL)
python scripts/judge_done.py --data data/nnetnav-v2/train.jsonl --out judge-train.jsonl
python scripts/apply_judge.py --data data/nnetnav-v2 --judge 'judge-{split}.jsonl' --out data/nnetnav-v3-clean

# wev-4b v0.2: continue from Kev-4B on the v0.1 mix, then add the GLM teacher episodes (resumable: rerun the same
# line after a kill, on any number of GPUs), export, and fit the serving temperature on dev rows
torchrun --nproc_per_node 8 -m wev.train --init_from jaredpalmer/kev-4b --lr 5e-5 --global_accum 8 --epochs 1 \
  --batch_tokens 4000 --resume --out runs/wev-4b-a --data <the v0.1 mix below>
torchrun --nproc_per_node 8 -m wev.train --init_from runs/wev-4b-a --lr 3e-5 --global_accum 8 --epochs 1 \
  --batch_tokens 4000 --resume --out runs/wev-4b-v0.2 \
  --data data/teacher-v3-glm:3,data/teacher-v1,data/teacher-v2,data/m2w-v2,data/nnetnav-v3-clean,data/general/kev-v7,data/general/typed-decisions-train:2,data/ext/td-synth
wev export --run runs/wev-4b-v0.2 --out exports/wev-4b
python scripts/calibrate.py --model exports/wev-4b --data data/m2w-v2 data/nnetnav-v3-clean data/teacher-v3-glm data/general/kev-v7 data/general/typed-decisions-train

# the wev-4b v0.1 recipe; dir:K repeats a training file K times. One GPU works too: drop torchrun.
# wev-8b and wev-1.7b use the same mix without data/teacher-v2.
torchrun --nproc_per_node 8 -m wev.train --base Qwen/Qwen3-4B-Base --head pointer --lr 1e-4 --epochs 1 \
  --batch_tokens 6000 --accum 1 --checkpointing 0 --out runs/wev-4b \
  --data data/m2w-v2,data/nnetnav-v3-clean,data/teacher-v1:3,data/teacher-v2:3,data/general/kev-v7:2,data/general/typed-decisions-train:8,data/ext/tasksource-jev,data/ext/jev-distill,data/ext/td-synth

wev evaluate --model runs/wev-4b --data data/general/kev-transfer-v4 --split dev
wev export --run runs/wev-4b --out exports/wev-4b --check data/general/typed-decisions/dev.jsonl
pytest -q tests

# the Hugging Face dataset (parquet configs + the wev-bench test set) and its PII scan
python scripts/build_hf_dataset.py --data data --tasks tasks-v1.jsonl --out hf/wev-data
python scripts/scan_pii.py --data hf/wev-data --out hf/pii-report.json
```

The converted browser data and the teacher collections are on the Hub as
[`alanhuangya/OpenJev-Data`](https://huggingface.co/datasets/alanhuangya/OpenJev-Data); a downloaded config folder can be
passed to `--data` as is (JSONL or parquet, `validation` is read as the development split). To collect new teacher
episodes, use `scripts/teacher_server.py`, `make_tasks.py`, `collect.py` and `build_teacher_data.py`; for on-policy
rows, `scripts/dagger_server.py` and `build_teacher_data.py --onpolicy`.

</details>

<details>
<summary><b>Training data and licenses</b></summary>

| Source | License | What it adds |
|---|---|---|
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | CC BY 4.0 | human browser steps (click, type, select), split by website |
| [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) | Apache-2.0 | live-web steps; DONE relabelled by an LLM judge |
| teacher episodes | outputs of qwen3-max | jev-ultrafast on live sites with qwen3-max as System One; judge-verified |
| [Kev decision-v7](https://github.com/jaredpalmer/kev) | per source | ten public classification and QA sources plus rule-composition records |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | agent and ops workflows (80% of its train split) |
| [tasksource-jev](https://huggingface.co/datasets/tasksource/tasksource-jev) | mixed, per source task | hundreds of classification tasks recast as decisions |
| [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Apache-2.0 | synthetic operational scenarios with soft labels |
| [typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | MIT | multi-question cases over 149 domains |

**Use terms.** The weights are released under Apache-2.0, but some training data carries its own terms: several
tasksource-jev source tasks are licensed for research only, and the teacher episodes are outputs of qwen3-max,
subject to its provider's terms. Treat the models as research artifacts and check those terms before commercial use.

</details>

## Limitations

- English only. wev makes decisions and does not write text; in jev-ultrafast, typed values come from its separate
  text model.
- Browser targets are scored among the candidates the agent lists (8–40 per step), not every element on the page, so
  the numbers are not comparable to the Mind2Web leaderboard.
- Rare operations are rarely predicted: BLOCKED and scrolling have low recall. On NNetNav, OpenJev-4B says DONE too early
  on 8–10% of steps unless you threshold it.
- On Kev's own suite, Kev is more accurate.
- wev has not been compared with Jev itself, because we have no API access.

## Citation

Jun Huang and Xin Ren contributed equally (University of Electronic Science and Technology of China).

```bibtex
@misc{huang2026wev,
  title     = {wev: Distilling LLM Browser Agents into Open, Local System-One Decision Models},
  author    = {Huang, Jun and Ren, Xin},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22941164},
  url       = {https://doi.org/10.5281/zenodo.22941164}
}
```

## Acknowledgements and license

Apache-2.0. The model code builds on [kev](https://github.com/jaredpalmer/kev) (Apache-2.0) and uses instruction text
from [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT); the models build on Qwen3 base models
(Apache-2.0). wev is an independent project, not affiliated with TypeSafe AI, and does not use Jev. See
[NOTICE](https://github.com/alanhuangyoo/wev/blob/main/NOTICE).
