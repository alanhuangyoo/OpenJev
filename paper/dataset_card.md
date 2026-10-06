---
license: other
license_name: mixed-see-card
language:
- en
pretty_name: wev data
tags:
- browser-agent
- web-navigation
- decision-model
- system-one
- distillation
- benchmark
size_categories:
- 10K<n<100K
configs:
- config_name: mind2web
  data_files:
  - {split: train, path: mind2web/train.parquet}
  - {split: validation, path: mind2web/validation.parquet}
  - {split: test, path: mind2web/test.parquet}
- config_name: nnetnav_audited
  data_files:
  - {split: train, path: nnetnav_audited/train.parquet}
  - {split: validation, path: nnetnav_audited/validation.parquet}
  - {split: test, path: nnetnav_audited/test.parquet}
- config_name: webchain
  data_files:
  - {split: train, path: webchain/train.parquet}
  - {split: validation, path: webchain/validation.parquet}
  - {split: test, path: webchain/test.parquet}
- config_name: teacher_qwen3max
  data_files:
  - {split: train, path: teacher_qwen3max/train.parquet}
  - {split: validation, path: teacher_qwen3max/validation.parquet}
  - {split: test, path: teacher_qwen3max/test.parquet}
- config_name: teacher_glm
  data_files:
  - {split: train, path: teacher_glm/train.parquet}
  - {split: validation, path: teacher_glm/validation.parquet}
  - {split: test, path: teacher_glm/test.parquet}
- config_name: onpolicy_glm
  data_files:
  - {split: train, path: onpolicy_glm/train.parquet}
  - {split: validation, path: onpolicy_glm/validation.parquet}
  - {split: test, path: onpolicy_glm/test.parquet}
- config_name: live_tasks
  data_files:
  - {split: train, path: live_tasks/train.parquet}
  - {split: test, path: live_tasks/test.parquet}
- config_name: bench
  data_files:
  - {split: test, path: bench/test.parquet}
---

# wev data

**Jun Huang\*, Xin Ren\*** · University of Electronic Science and Technology of China · \*Equal contribution

Browser-step data and a benchmark for **decision models**: the small, fast models that pick a web agent's next
operation and target element. Every row is one browser step, written as a `POST /v1/systemone` request (the page
state plus typed questions) with its labelled answers. The requests use exactly the format the open browser agent
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast) sends to its System One, so a model trained or scored
here can be served behind that agent unchanged. The data trains the [wev](https://github.com/alanhuangyoo/OpenJev) models
([wev-1.7b](https://huggingface.co/alanhuangya/OpenJev-1.7B), [wev-4b](https://huggingface.co/alanhuangya/OpenJev-4B),
[wev-8b](https://huggingface.co/alanhuangya/OpenJev-8B)), and the `bench` config is a frozen test set, **wev-bench**, that
scores any System One endpoint.

## Load it

Every config is parquet and loads with one call:

```python
from datasets import load_dataset

m2w      = load_dataset("alanhuangya/OpenJev-Data", "mind2web")          # human steps, split by website
nnetnav  = load_dataset("alanhuangya/OpenJev-Data", "nnetnav_audited")   # live-web steps with audited DONE labels
webchain = load_dataset("alanhuangya/OpenJev-Data", "webchain")          # human steps on 200+ live sites
qwen     = load_dataset("alanhuangya/OpenJev-Data", "teacher_qwen3max")  # qwen3-max acting as System One
glm      = load_dataset("alanhuangya/OpenJev-Data", "teacher_glm")       # GLM-5.3-Flash acting as System One
onpolicy = load_dataset("alanhuangya/OpenJev-Data", "onpolicy_glm")      # student-visited states, GLM labels
tasks    = load_dataset("alanhuangya/OpenJev-Data", "live_tasks")        # goals + start URLs; test = 153 held-out tasks
bench    = load_dataset("alanhuangya/OpenJev-Data", "bench", split="test")  # wev-bench, frozen
```

## Configs

| Config | Rows (train / validation / test) | Source | License |
|---|---|---|---|
| `mind2web` | 5,861 / 586 / 875 | [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web), converted | CC BY 4.0 |
| `nnetnav_audited` | 11,408 / 1,140 / 1,150 | [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live), converted, stopping labels audited | Apache-2.0 |
| `webchain` | 17,876 / 1,069 / 1,589 | [WebChain](https://huggingface.co/datasets/webagentlab/webchain) v2, converted from its HTML snapshots | CC BY 4.0 |
| `teacher_qwen3max` | 4,781 / 385 / 57 | qwen3-max as the System One of jev-ultrafast on live sites (two collections) | see *Terms* |
| `teacher_glm` | 5,969 / 370 / 147 | GLM-5.3-Flash in the same role | see *Terms* |
| `onpolicy_glm` | 5,782 / 275 / 119 | DAgger-style: a wev student or the GLM teacher acts, the GLM teacher labels | see *Terms* |
| `live_tasks` | 1,402 / – / 153 | goals and start URLs for live runs; `test` is the held-out end-to-end suite | see *Terms* |
| `bench` | – / – / 2,346 | wev-bench: frozen test steps from the configs above | per row's source |

In v1 the two qwen3-max collections were separate configs (`teacher_first`, `teacher_second`); they are now one
config with a `collection` column (`teacher-v1`, `teacher-v2`). v1 shipped JSONL; v2 ships parquet.

The general typed-decision corpora used in training (Kev decision-v7, typed-decisions, tasksource-jev,
jev-distill-corpus-v3, typed-decisions-synth) are not redistributed here; the repository's builders download and
convert them from their sources.

## Columns

Step configs (all but `live_tasks`) share one schema:

| Column | Meaning |
|---|---|
| `request` | the full `POST /v1/systemone` request as a JSON string; send it as is to replay the step |
| `labels` | JSON string: the gold option key per labelled question, e.g. `{"operation": "CLICK", "click_target": "7"}` |
| `meta` | JSON string: the source's own metadata (for teacher rows: episode, task, teacher, its one-sentence reason) |
| `id` | stable row id (hash of request and labels) |
| `url`, `title`, `goal` | page URL and title, task goal |
| `n_elements`, `n_recent_actions` | candidate elements offered (8–40 for converted data), actions in the history |
| `operation` | gold operation: CLICK, TYPE_TEXT, SELECT, SCROLL_DOWN, SCROLL_UP, WAIT, DONE or BLOCKED |
| `target_question`, `target`, `target_element` | the labelled target question (`click_target`, ...), its gold key and the element's text |
| `source`, `collection` | origin dataset and the collection it was built in (`m2w-v2`, `teacher-v3-glm`, `dagger-g1`, ...) |
| `website`, `task_id`, `episode_id` | site, task and episode (or trajectory) ids |
| `actor` | `onpolicy_glm` only: who acted at this step, `student` or `teacher` |
| `teacher`, `reason` | teacher rows: teacher model and its stated reason |

`bench` adds `bench_subset`, the config each row comes from. `live_tasks` has `id`, `url`, `goal`, `source`.

A request looks like this (shortened):

```json
{
  "model": "wev-latest",
  "state": {
    "page": {"url": "...", "title": "...", "text": "visible page text"},
    "elements": [{"index": "1", "role": "button", "label": "Search", "operations": ["CLICK"]}],
    "recent_actions": [{"action": "...", "kind": "click", "text": null, "page_changed": true}]
  },
  "questions": {
    "operation": {"type": "choice",
                  "instructions": {"goal": "...", "rules": ["..."]},
                  "criteria": {"CLICK": "...", "TYPE_TEXT": "...", "DONE": "...", "BLOCKED": "..."}},
    "click_target": {"type": "choice",
                     "instructions": {"goal": "...", "operation": "CLICK", "rules": ["..."]},
                     "criteria": {"1": {"element": "[1] Search", "current_value": "", "role": "button"}}}
  }
}
```

A step asks for the operation and, for each operation that needs one, a target (`click_target`, `type_text_target`
or `select_target`). A step is answered correctly when the operation and, if the operation takes one, its target both
match. Targets are chosen among the candidates listed in the state, not among every element on the page. To train
with the wev package, pass JSONL folders to `wev train --data`; `scripts/build_hf_dataset.py` documents the mapping.

## How the configs were built

**mind2web.** Each recorded step becomes one request containing the page's visible text, a sample of 8–40
candidate elements that includes the gold element, the task and the preceding actions. Splits are by website, so every
test website is unseen in training. Mind2Web has no DONE, WAIT, SCROLL or BLOCKED steps.

**nnetnav_audited.** NNetNav supplies the stopping (DONE), giving-up (BLOCKED) and scrolling steps that Mind2Web
lacks. Its trajectories come from unsupervised exploration, so its stopping labels are noisy. An LLM judge
(DeepSeek-V4.1-Flash) read the goal and the state at every step and decided whether the goal was already achieved.
On the training split it rejected 592 of 1,898 DONE labels (31%) and found the goal already achieved at 2,281 of the
10,102 other steps (23%). Those steps were relabelled DONE and the rejected DONE steps were dropped. The validation
and test splits were audited the same way.

**webchain.** WebChain v2 releases screenshots and multimodal SFT conversations, but its action metadata also links,
for most steps, a pre-action HTML snapshot and an accessibility tree together with the CSS selector of the element
acted on. `wev/webchain.py` resolves that selector in the snapshot (unique matches only, climbing from e.g. a `<span>`
to its `<button>`), samples 8–40 negatives from the snapshot's other visible interactive elements, and renders the
request exactly as for Mind2Web. Clicks, hovers and double clicks become CLICK, typing and pasting TYPE_TEXT. 4,000
of WebChain's 31,445 English-goal trajectories were converted (a fixed hash order); pages not in English, steps
without a snapshot, unresolvable selectors and custom (non-`<select>`) dropdowns were skipped. Splits are by host
(175 / 15 / 16 hosts). Like Mind2Web, it has no DONE, WAIT, SCROLL or BLOCKED steps.

**teacher_qwen3max, teacher_glm.** A prompted LLM served as the System One of jev-ultrafast on live websites:
qwen3-max for the two collections in `teacher_qwen3max` (the second ran on the tasks the first had not solved), and
GLM-5.3-Flash, served through Alibaba Bailian, for `teacher_glm`. The prompt forbade signing in, registering, buying,
booking, posting, messaging and submitting personal information, and asked for BLOCKED on CAPTCHAs, unusual-traffic
pages and login walls. Every request and the teacher's choice were logged. An LLM judge then read the final page of
each episode, and the kept rows are:

- every decision of an episode whose DONE the judge confirmed;
- every decision up to a BLOCKED on a page that blocked the agent;
- every decision but the last of an episode cut short by the browser harness.

Episodes that looped or exhausted their budget were dropped, as were repeated states. The judge rejected 155 of
qwen3-max's 431 DONE claims (36%) and 127 of GLM-5.3-Flash's 530 (24%).

**onpolicy_glm.** Trained only on teacher episodes, a student never sees the states its own mistakes lead to. This
config was collected DAgger-style with a mixed policy: at every step both a wev student and the GLM-5.3-Flash teacher
answered the request, one of them chosen at random with equal probability acted (`actor`: 2,894 student and 2,896
teacher steps in train), and the teacher's choice was always recorded as the label. An episode's ending was judged
from the actions actually executed: the judge-confirmed final state is labelled DONE, earlier teacher DONE labels
(unverified) are dropped, and a BLOCKED ending is kept only when the teacher agrees. The judge rejected 135 of 587
DONE endings (23%). Otherwise the filters are those of the teacher configs.

**Splits of the teacher configs.** Episodes are split by a hash of the task id, so a task is in the same split in
every collection. A held-out task becomes `test` only when neither its task id nor its goal occurs in any train split
of any config, and `validation` otherwise. NNetNav-sourced tasks take their goals from NNetNav's train split, so the
teacher test splits hold only the Google Flights and Wikipedia tasks: 323 steps from 8 tasks.

**live_tasks.** Goals paired with start URLs: NNetNav goals on their live sites, Wikipedia look-ups and Google
Flights searches. The 153 `test` tasks are the end-to-end suite used to evaluate the wev models; none of them appears
in a teacher config. Checked by task id and by goal against every train split: one eval goal ("Find hotels in San
Francisco." on booking.com) also occurs at one step of `nnetnav_audited` train, because NNetNav repeats some goals
across its own splits. The 123 NNetNav eval tasks come from NNetNav's test split, so their goals appear in
`nnetnav_audited` test (555 steps); that is by construction, not a leak.

## wev-bench

`bench` is a frozen test set for browser decision models: 2,346 steps, each a request a System One must answer.

| Subset | Steps | What it tests |
|---|---|---|
| `mind2web` test | 873 | clicks, typing and dropdowns on 8 websites unseen in training |
| `nnetnav_audited` test | 1,150 | stopping (DONE 382), giving up (BLOCKED 29) and scrolling on live sites |
| teacher test slices | 323 | qwen3-max (57), GLM (147) and on-policy (119) steps on tasks in no train split |

Two Mind2Web test requests are left out because one question exceeds the API's 255-option limit.

**Metrics.** A step succeeds when the operation and, if the step has one, the target are right. wev-bench reports
step success, operation accuracy, target accuracy, the premature-DONE rate (steps whose gold operation is not DONE
that the model answered DONE: an agent that stops early abandons its task), DONE recall, top-label ECE (10 bins; on
the probability the model gives its own choice) and p50/p95 latency per request, overall and per subset. Conventions
follow [JevBench](https://github.com/fstandhartinger/jevbench): failed or malformed answers count as wrong, and an
answer without a distribution is scored as one-hot and flagged.

**Run it** against any endpoint that speaks `POST /v1/systemone`:

```bash
wev serve --model alanhuangya/OpenJev-4B --port 8009        # or any System One server
python scripts/wev_bench.py --url http://127.0.0.1:8009/v1/systemone --name my-model \
  --hardware "RTX 5090, bf16" --out results/wev-bench/my-model.json
```

The script is in the [wev repository](https://github.com/alanhuangyoo/OpenJev/blob/main/scripts/wev_bench.py) and needs
only the Python standard library and `pyarrow`. The result is a JSON file in JevBench's per-system layout (`key`,
`display`, `open`, `licence`, `has_distribution`, a `browser` lane block with `by_family`, `speed_block`, ...), plus
the dataset revision and the bench file's sha256, so it can be submitted to a leaderboard as is.

| System One | Step success | Operation | Target | Premature DONE | ECE | p50 / p95 latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| wev-4b | 68.0 | 81.3 | 79.0 | 5.0 | 0.051 | 166 / 409 ms |

wev-4b was served with `wev serve` on one RTX 5090 (bf16) and queried serially from the same host. Its training
included `mind2web`, `nnetnav_audited` and `teacher_qwen3max` train splits, so it is in-distribution there.

## Personal and sensitive information

All step rows were scanned for e-mail addresses, phone numbers, card-like and SSN-like numbers, API-key, JWT and
private-key shapes, credential-like URL parameters, values in password fields, and personal names typed into fields.
Most matches are public business contacts in page text and element labels (phone numbers on 2,510 rows, e-mail
addresses on 1,525).
Text an agent typed: made-up contact details dictated by Mind2Web and NNetNav goals (e.g. an e-mail on gmail.com,
example.com or sample.com), names typed into name fields (23 steps, plus 75 where the goal supplies the name), and
values typed into password fields by NNetNav's exploring agent (17 steps) and a 4-digit gift-card PIN in Mind2Web
(1). URLs: short-lived signed tokens (JWT-shaped `edgeauth`/OAuth `state` values, session ids; 163 rows). Page
text: one Hugging Face token in a code sample on a public model page (2 rows) and one store loyalty number (2 rows).
No credentials were entered by the teacher, which was instructed never to sign in.

Applied in this release (`scripts/redact.py`, run by `scripts/build_hf_dataset.py`): typed e-mail addresses, phone
numbers and personal names not dictated by the goal became `user@example.com`, `555-0100` and `Alex Doe`; values in
password fields, API keys and tokens, and credential-like URL parameters became `<redacted>`; other e-mail addresses
in page text and labels became `<email>`; the 3 rows with card- or SSN-like numbers were dropped. Published business
phone numbers on public pages were kept. Separately, 86 teacher and on-policy steps labelled DONE on a page that had
not loaded (no elements, almost no text) were dropped.

## Terms

| Source | License |
|---|---|
| Mind2Web-derived rows | CC BY 4.0 |
| NNetNav-derived rows | Apache-2.0 |
| WebChain-derived rows | CC BY 4.0 |
| teacher, on-policy rows and live tasks | see below |

Attribute the original datasets (see *Citation*). The teacher and on-policy configs and the live tasks contain text
captured from public websites, which remains subject to those sites' terms. The labels in `teacher_qwen3max` are
outputs of qwen3-max, and those in `teacher_glm` and `onpolicy_glm` are outputs of GLM-5.3-Flash (served through
Alibaba Bailian); they are subject to their providers' terms. The DONE verdicts and NNetNav relabelling come from an
LLM judge (DeepSeek-V4.1-Flash). Treat these configs as research data and check the applicable terms before any
commercial use. WebChain's page snapshots were captured by its annotators on live websites and remain subject to
those websites' terms; its card asks users not to attempt to recover personal information.

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

If you use the converted configs, please also cite their sources:

```bibtex
@inproceedings{deng2023mind2web,
  title     = {Mind2Web: Towards a Generalist Agent for the Web},
  author    = {Xiang Deng and Yu Gu and Boyuan Zheng and Shijie Chen and Samuel Stevens and Boshi Wang and Huan Sun and Yu Su},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
@misc{murty2024nnetnav,
  title         = {NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild},
  author        = {Shikhar Murty and Hao Zhu and Dzmitry Bahdanau and Christopher D. Manning},
  year          = {2024},
  eprint        = {2410.02907},
  archivePrefix = {arXiv}
}
@misc{webchain2026,
  title         = {WebChain: A Large-Scale Human-Annotated Dataset of Real-World Web Interaction Traces},
  author        = {Sicheng Fan and Rui Wan and Yifei Leng and Gaoning Liang and Li Ling and Yanyi Shang and Dehan Kong},
  year          = {2026},
  eprint        = {2603.05295},
  archivePrefix = {arXiv}
}
```
