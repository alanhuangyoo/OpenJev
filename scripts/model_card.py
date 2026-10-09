"""Write the Hugging Face model card (README.md) for an exported OpenJev model from result files.

    python scripts/model_card.py --export exports/wev-4b --repo alanhuangya/OpenJev-4B --locked locked/wev-4b \
        --baselines results/baselines --e2e "22/153" --teacher_e2e "27/153"

Every number comes from a result JSON: --locked holds this model's single test read (wev evaluate), --baselines the
same test splits scored by scripts/compare.py for the other models (<model>-<benchmark>.json).
"""
import argparse
import json
from pathlib import Path

GENERAL = [("kev-v7", "kev decision-v7"), ("kev-transfer-v4", "kev transfer-v4"), ("typed-decisions", "typed-decisions")]
BASELINES = [("kev-4b", "Kev-4B"), ("kev-8b", "Kev-8B"), ("laya-typed-decisions", "Laya (typed-decisions)"), ("laya", "Laya")]


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def table(header, grid):
    """Markdown table of (label, [values]) rows, the best value of each column in bold."""
    best = [max((v[i] for _, v in grid if v[i] is not None), default=None) for i in range(len(header) - 1)]
    rows = [f"| {label} | " + " | ".join(f"**{pct(x)}**" if x is not None and x == best[i] else pct(x)
                                        for i, x in enumerate(v)) + " |" for label, v in grid]
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *rows])


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--e2e", default="", help="judge-verified end-to-end successes of this model, e.g. 22/153")
    ap.add_argument("--teacher_e2e", default="")
    ap.add_argument("--github", default="https://github.com/alanhuangyoo/OpenJev")
    ap.add_argument("--init_from", default="", help="Hub repo the adapter was trained from, if not the bare base")
    ap.add_argument("--glm_teacher_e2e", default="", help="end-to-end successes of the GLM-5.3-Flash teacher")
    ap.add_argument("--latency", default="", help="one line on measured latency")
    ap.add_argument("--test_reads", default="For OpenJev-4B and OpenJev-8B, two candidates each were read on test (see the "
                                            "repository README).")
    a = ap.parse_args()

    info = json.loads((Path(a.export) / "wev.json").read_text())
    base, targs = info["base"], info["train_args"]
    name = a.repo.split("/")[-1]   # the Hub name, e.g. OpenJev-4B
    locked, bdir = Path(a.locked), Path(a.baselines)
    ours = {b: read(locked / f"{b}.json") for b, _ in GENERAL + [("m2w-v2", ""), ("nnetnav-v3-clean", "")]}

    def acc(m, key="all_questions"):
        return None if m is None else (m["step_success"] if key == "step" else m[key]["accuracy"])

    grid = [(f"**{name}**", [acc(ours[b]) for b, _ in GENERAL])]
    grid += [(label, [acc(read(bdir / f"{key}-{b}.json")) for b, _ in GENERAL]) for key, label in BASELINES]
    general = table(["model"] + [l for _, l in GENERAL], grid)

    m2w = ours["m2w-v2"]
    # the evaluator drops requests beyond the model's context; the baselines saw all of them: count those as wrong
    n_all = max([m2w["n"]] + [m["n"] for m in (read(bdir / f"{k}-m2w-v2.json") for k, _ in BASELINES) if m])
    keep = m2w["n"] / n_all
    grid = [(f"**{name}**", [m2w["step_success"] * keep, m2w["operation"]["accuracy"] * keep])]
    for key, label in BASELINES:
        m = read(bdir / f"{key}-m2w-v2.json")
        grid.append((label, [None, None] if m is None else [m["step_success"], m.get("operation", {}).get("accuracy")]))
    browser = table(["model", "step success", "operation"], grid)
    nn = ours["nnetnav-v3-clean"]
    done = nn["operation_by_gold"].get("DONE", {}) if nn else {}

    card = f"""---
license: apache-2.0
base_model: {a.init_from or base}
language:
- en
tags:
- decision-model
- browser-agent
- system-one
- lora
datasets:
- osunlp/Mind2Web
- stanfordnlp/nnetnav-live
- LocalLLaMA/typed-decisions
- tasksource/tasksource-jev
- SargeDev/jev-distill-corpus-v3
- n4ze3m/typed-decisions-synth
---

# {name}

**Jun Huang\\*, Xin Ren\\*** · University of Electronic Science and Technology of China · \\*Equal contribution

**A local decision model: typed questions in, calibrated probabilities out, in one forward pass.** {name} answers
the `POST /v1/systemone` request shape (choice, yes/no and score questions over a free-form state), for general
decisions and for browser-agent steps (*which operation? which element?*). It runs on your own GPU: no API key,
no per-call cost, nothing generated.

Code, training and evaluation: [{a.github.split('github.com/')[1]}]({a.github}). Paper:
[doi:10.5281/zenodo.22941164](https://zenodo.org/records/22941164). Independent project; not affiliated
with TypeSafe AI.

## Quickstart

```bash
pip install "wev-ai[serve]"
```

```python
import wev
m = wev.load("{a.repo}")
out = m.predict(
    state="Refund request: order #4411 arrived damaged, customer attached photos, first refund this year.",
    questions={{
        "action": {{"type": "choice", "instructions": "What should support do?",
                   "criteria": {{"refund": "Refund the order.", "replace": "Ship a replacement.",
                                "escalate": "Send to a human agent."}}}},
        "fraud_risk": {{"type": "noul", "instructions": "This request looks fraudulent.",
                       "criteria": {{"true": "Likely fraud.", "false": "No sign of fraud."}}}},
    }},
)
print(out["answers"])
```

```bash
wev serve --model {a.repo} --port 8009   # drop-in POST /v1/systemone, e.g. for jev-ultrafast
```

## Results

Test splits, held out from training; every other model was run on the same requests and scored the same way
(per-question accuracy; `scripts/compare.py`). {a.test_reads}

**General typed decisions**

{general}

{name} trains on 80% of the typed-decisions train split, like the Laya (typed-decisions) specialist; Kev and Laya
do not, so on that column they are generalists. kev decision-v7 is Kev's own training suite ({name} also trains on
its train split); transfer-v4 is out-of-domain for every model here.

**Browser steps** (Mind2Web test split: websites unseen in training, jev-ultrafast request format; step success =
operation and target element both right)

{browser}

{n_all} requests; {n_all - m2w["n"]} exceed the context {name} is evaluated with and count as wrong for it.

NNetNav test split (live-web steps, DONE judged by an LLM): step success {pct(acc(nn, 'step'))}, DONE recall
{pct(done.get('recall'))}, premature DONE {pct(nn.get('premature_done_rate') if nn else None)}.
"""
    if a.e2e:
        card += f"""
**End to end** (153 held-out tasks on live websites, run by [jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
with {name} as its System One; success = the agent says DONE and an LLM judge reading the final page agrees):
{a.e2e} tasks{f', vs {a.teacher_e2e} for the qwen3-max teacher behind the same agent' if a.teacher_e2e else ''}{f' and {a.glm_teacher_e2e} for the GLM-5.3-Flash teacher' if a.glm_teacher_e2e else ''}.
Live sites differ from run to run; treat gaps of a few tasks as noise.
"""
    card += f"""
## Model

- Backbone: `{base}` without its vocabulary head, LoRA r={targs['lora']} on every attention and MLP projection, merged
  into the weights of this export; {info['num_layers']} layers, bf16.
- Readout: a {info['head_type']} head scores each option's `</opt>` state against the question's `<decide>` state.
- Each question sees the state and itself only (positions restart after the state), so the state is encoded once
  per request however many questions it asks, and answers never depend on question order.
- Probabilities are calibrated: serving divides the logits by a temperature of {info.get('temperature', 1.0)} fitted on
  development rows, which leaves every answer unchanged.{chr(10) + '- ' + a.latency if a.latency else ''}
- Context: state up to {info['max_state']} tokens, each question up to {info['max_branch']} tokens (trained with
  {info['train_max_branch']}); longer page states are shrunk before encoding.

## Training

{targs['epochs']} epoch, lr {targs['lr']}, one-cycle schedule, soft-label cross-entropy where the source has soft labels{f'; the adapter starts from `{a.init_from}`' if a.init_from else ''}.
Recipe and data builders: [{a.github.split('github.com/')[1]}]({a.github}).

| source | license | what it adds |
|---|---|---|
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | CC BY 4.0 | human browser steps: click, type, select |
| [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) | Apache-2.0 | live-web steps; DONE relabelled by an LLM judge |
| teacher episodes | outputs of qwen3-max | jev-ultrafast on live sites with qwen3-max as System One, success judge-verified |
| GLM teacher episodes | outputs of GLM-5.3-Flash | the same collection with GLM-5.3-Flash as System One, success judge-verified |
| [kev decision-v7](https://github.com/jaredpalmer/kev) | per source | ten public classification / QA sources plus rule records |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | agent / ops workflows, 5 questions per case (80% of train) |
| [tasksource-jev](https://huggingface.co/datasets/tasksource/tasksource-jev) | mixed (per source task; some research-only) | hundreds of classification tasks as decisions |
| [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Apache-2.0 | synthetic operational scenarios, soft labels |
| [typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | MIT | multi-question cases over 149 domains |

**Use terms.** Some training data carries its own terms: several tasksource-jev source tasks are research-only, and the
teacher episodes are qwen3-max and GLM-5.3-Flash outputs subject to their providers' terms. Treat this model as a research artifact and
check those terms before any commercial use.

## Limitations

- English only. Decisions, not text: TYPE values come from a separate text model, as in jev-ultrafast.
- Browser targets are scored among the candidates the agent lists (8–40 per step), not every element on the page.
- DONE and BLOCKED are the hardest operations; gate DONE on its probability when early stops are costly.
- Not compared with Jev itself (no API access).

## Citation

Jun Huang and Xin Ren contributed equally (University of Electronic Science and Technology of China). The paper
describes OpenJev under its earlier name, wev.

```bibtex
@misc{{huang2026wev,
  title     = {{wev: Distilling LLM Browser Agents into Open, Local System-One Decision Models}},
  author    = {{Huang, Jun and Ren, Xin}},
  year      = {{2026}},
  publisher = {{Zenodo}},
  doi       = {{10.5281/zenodo.22941164}},
  url       = {{https://doi.org/10.5281/zenodo.22941164}}
}}
```

## License

Apache-2.0, like the base model. Architecture code adapted from [kev](https://github.com/jaredpalmer/kev) (Apache-2.0){f'; weights initialised from [{a.init_from}](https://huggingface.co/{a.init_from}) (Apache-2.0)' if a.init_from else ''}.
"""
    (Path(a.export) / "README.md").write_text(card)
    print(card)


if __name__ == "__main__":
    main()
