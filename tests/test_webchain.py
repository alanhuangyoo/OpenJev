import pytest

pytest.importorskip("cssselect")

from wev.api import SystemOneRequest  # noqa: E402
from wev.webchain import clean_goal, convert_trace  # noqa: E402

LINKS = "".join(f'<a href="/p{i}">Product {i}</a>' for i in range(12))
HTML = f"""<html lang="en-US"><head><title>Shop</title></head><body>
<div id="top"><label for="q">Search products</label><input id="q" type="search" value="">
<button class="go"><span>Search</span></button>
<select id="size"><option value="s" selected>Small</option><option value="l">Large</option></select>
<a href="/login">Sign in</a><div style="display:none"><button>Hidden</button></div>{LINKS}</div>
<p>Free shipping on orders over $50</p></body></html>""".encode()
TRACE = {"uid": "t1", "query": 'Task 2: "Find a large red shirt"', "primary_host": "shop.example",
         "web_type": "shopping", "intent_type": "Multi-Constraint"}


def action(i, kind, selector, value="", text="no input text"):
    return {"trace_uid": "t1", "source_step_index": i, "action_type": kind, "included_in_sft": True,
            "html_dom_url": f"https://snap/{i}.html", "selector": selector, "href": "https://shop.example/",
            "value": value, "input_text": text, "attributes": "{}"}


def convert(actions, html=HTML):
    rows, skips = convert_trace(TRACE, actions, lambda url: html, 0, 8, 12, 100, 6000)
    for r in rows:
        SystemOneRequest.model_validate(r["request"])
    return rows, skips


def target(row):
    op = row["labels"]["operation"]
    qid = op.lower() + "_target"
    return row["request"]["questions"][qid]["criteria"][row["labels"][qid]]["element"]


def test_type_click_select_and_history():
    rows, skips = convert([action(0, "launchApp", ""), action(1, "type", "#q", text="red shirt"),
                           action(2, "click", "#top >.go >span:nth-child(1)", "Search"),
                           action(3, "select", "#size", text="Large")])
    assert [r["labels"]["operation"] for r in rows] == ["TYPE_TEXT", "CLICK", "SELECT"]
    assert target(rows[0]).endswith("Search products")
    assert target(rows[1]).endswith("] Search")            # the <span> climbs to its <button>
    assert target(rows[2]).endswith("→ Large")
    assert rows[1]["request"]["state"]["recent_actions"] == [
        {"action": "Search products", "kind": "fill", "text": "red shirt", "page_changed": None}]
    assert rows[0]["request"]["questions"]["operation"]["instructions"]["goal"] == "Find a large red shirt"
    labels = [e["label"] for e in rows[1]["request"]["state"]["elements"]]
    assert "Hidden" not in labels
    assert skips["action:launchApp"] == 1


def test_unresolved_selector_and_non_english_are_skipped():
    rows, skips = convert([action(1, "click", ".missing")])
    assert not rows and skips["selector_unresolved"] == 1
    rows, skips = convert([action(1, "click", "#q")], HTML.replace(b'lang="en-US"', b'lang="zh-CN"'))
    assert not rows and skips["not_english"] == 1


def test_clean_goal():
    assert clean_goal('- Task 4: ""Look for seafood."""') == "Look for seafood."
