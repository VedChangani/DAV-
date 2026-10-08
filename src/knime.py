"""Reads the KNIME workflow straight from its saved files (workflow.knime + node settings.xml)."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = ROOT / "Assignment"
KNWF_PATH = ROOT / "knime" / "Student_Risk_Workflow.knwf"
NS = {"k": "http://www.knime.org/2008/09/XMLConfig"}

STAGES = {
    "CSV Reader": "1 · Ingest",
    "Column Renamer": "2 · Clean",
    "Missing Value": "2 · Clean",
    "Duplicate Row Filter": "2 · Clean",
    "Math Formula": "3 · Feature engineering",
    "Expression": "3 · Feature engineering",
    "Rule Engine": "4 · Risk segmentation",
    "GroupBy": "5 · Aggregate & visualise",
    "Bar Chart": "5 · Aggregate & visualise",
    "Column Filter": "6 · Model prep",
    "One to Many": "6 · Model prep",
    "Table Partitioner": "6 · Model prep",
    "Decision Tree Learner": "7 · Model & evaluate",
    "Decision Tree Predictor": "7 · Model & evaluate",
    "Decision Tree View": "7 · Model & evaluate",
    "Scorer": "7 · Model & evaluate",
}


def _child(e, key):
    return next((c for c in e.findall("k:config", NS) if c.get("key") == key), None)


def _entry(e, key):
    x = next((c for c in e.findall("k:entry", NS) if c.get("key") == key), None)
    return None if x is None else x.get("value")


def _find_entries(root, key):
    return [e.get("value") for e in root.iter(f"{{{NS['k']}}}entry") if e.get("key") == key]


def _describe(node_type: str, settings: ET.Element) -> str:
    model = _child(settings, "model")
    if model is None:
        return ""
    if node_type == "Math Formula":
        return f"{_entry(model, 'replaced_column')} = {_entry(model, 'expression')}"
    if node_type == "Expression":
        return f"{_entry(model, 'createdColumn')} = {_entry(model, 'script')}"
    if node_type == "Rule Engine":
        rules = _child(model, "rules")
        lines = [_entry(rules, str(i)) for i in range(int(_entry(rules, "array-size") or 0))] if rules is not None else []
        return f"{_entry(model, 'new-column-name')}: " + "  |  ".join(lines)
    if node_type == "Table Partitioner":
        return f"{_entry(model, 'percentage')}% train / rest test, {_entry(model, 'mode')} sampling"
    if node_type == "Decision Tree Learner":
        return (f"class = {_entry(model, 'classifyColumn')}, {_entry(model, 'splitQualityMeasure')}, "
                f"min records/node = {_entry(model, 'minNumberRecordsPerNode')}")
    if node_type == "Decision Tree Predictor":
        return "applies the learned tree to the 20% test partition"
    if node_type == "Decision Tree View":
        return "interactive view of the learned tree"
    if node_type == "Scorer":
        return f"{_entry(model, 'first')} vs {_entry(model, 'second')}"
    if node_type in ("Bar Chart",):
        return f"category = {_entry(model, 'categoryColumnV3')}"
    if node_type == "GroupBy":
        return "group by target → mean of performance metrics"
    if node_type == "One to Many":
        return "one-hot encodes categorical columns for the tree"
    if node_type == "Column Filter":
        return "keeps 25 model features + target (drops marital status, application info, macro-economics)"
    if node_type == "CSV Reader":
        return "reads df_output.csv (4,424 students × 37 columns)"
    if node_type == "Column Renamer":
        return "37 columns → snake_case"
    if node_type == "Missing Value":
        return "handles missing values per column type"
    if node_type == "Duplicate Row Filter":
        return "removes duplicate rows"
    return ""


def load_workflow() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ET.parse(WORKFLOW_DIR / "workflow.knime").getroot()
    nodes = []
    for n in _child(root, "nodes").findall("k:config", NS):
        folder = _entry(n, "node_settings_file").split("/")[0]
        m = re.match(r"(.+) \(#(\d+)\)", folder)
        node_type, node_id = m.group(1), int(m.group(2))
        settings = ET.parse(WORKFLOW_DIR / folder / "settings.xml").getroot()
        nodes.append({
            "ID": node_id,
            "Node": node_type,
            "Stage": STAGES.get(node_type, "Other"),
            "Configuration": html.unescape(_describe(node_type, settings)),
        })
    edges = []
    for c in _child(root, "connections").findall("k:config", NS):
        edges.append({"from": int(_entry(c, "sourceID")), "to": int(_entry(c, "destID"))})
    return pd.DataFrame(nodes).sort_values("ID"), pd.DataFrame(edges)


def scorer_metrics() -> dict:
    """Accuracy / kappa as saved by the KNIME Scorer node's output flow variables."""
    root = ET.parse(WORKFLOW_DIR / "Scorer (#26)" / "settings.xml").getroot()
    out = {}
    for cfg in root.iter(f"{{{NS['k']}}}config"):
        name = _entry(cfg, "name")
        val = _entry(cfg, "value")
        if name and val is not None:
            try:
                v = float(val)
            except ValueError:
                continue
            if v:  # the node stores a zeroed default copy first; keep the real values
                out[name] = v
    return out


def workflow_svg() -> str:
    return (WORKFLOW_DIR / "workflow.svg").read_text(encoding="utf-8")
