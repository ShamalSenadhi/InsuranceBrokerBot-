"""Underwriting facts per insurer: (1) seed facts verified from public policy documents (data/underwriting_rules.yaml)
and (2) facts auto-extracted from the insurers' own pages and policy PDFs during ingest (index/facts.json)."""
import json, os, re
import pandas as pd
import yaml

NUM = lambda s: int(s.replace(",", ""))
CAT_HINTS = [("Motor", r"motor|vehicle|\bcar\b|three[- ]wheel"), ("Health", r"health|medical|hospital|surgical"), ("Life (term)", r"life|term|endowment|pension|assurance"),
             ("Travel", r"travel"), ("Home & Fire", r"fire|property|home|householder"), ("Personal Accident", r"accident")]
TOPIC_CAT = {"ncb_max_pct": "Motor", "excess_by_vehicle_age": "Motor", "excess_hire": "Motor", "excess_rental": "Motor", "excess_motorcycle": "Motor",
             "young_driver_excess": "Motor", "max_driver_age": "Motor", "tppd_limit": "Motor"}

def cat_of(hint, topic):
    for c, p in CAT_HINTS:
        if re.search(p, hint or "", re.I): return c
    return TOPIC_CAT.get(topic, "General")

def extract_facts(insurer, text, source, hint=""):
    """Pattern-based extraction. Values are approximate: always shown with source for verification."""
    out, t = [], " ".join(text.split())
    def add(topic, value, m, h=None):
        s = max(0, m.start() - 80); out.append({"insurer": insurer, "category": cat_of((h or hint) + " " + source, topic), "topic": topic, "value": value,
                                               "snippet": t[s: m.end() + 80], "source": source, "origin": "Auto-extracted - verify"})
    for m in re.finditer(r"no[- ]claim bonus[^.%]{0,60}?(?:up ?to|maximum(?: of)?|of up to)\s*(\d{1,2})\s*%", t, re.I): add("ncb_max_pct", int(m.group(1)), m)
    for m in re.finditer(r"compulsory excess", t, re.I):
        w = t[m.start(): m.start() + 420]
        pairs = [[int(a), NUM(b)] for a, b in re.findall(r"(?:over|above|exceeding)?\s*(\d{1,2})\s*years?(?: old)?[^.\d]{0,25}?\(?\s*Rs\.?:?\s*([\d,]+)", w, re.I)]
        if pairs: add("excess_by_vehicle_age", sorted(pairs), m, "motor"); break
    for m in re.finditer(r"hiring vehicles?[^.]{0,80}?excess of Rs\.?\s*([\d,]+)", t, re.I): add("excess_hire", NUM(m.group(1)), m, "motor")
    for m in re.finditer(r"rental vehicles?[^.]{0,80}?excess of Rs\.?\s*([\d,]+)", t, re.I): add("excess_rental", NUM(m.group(1)), m, "motor")
    for m in re.finditer(r"motorcycles?[^.]{0,60}?excess of Rs\.?\s*([\d,]+)", t, re.I): add("excess_motorcycle", NUM(m.group(1)), m, "motor")
    for m in re.finditer(r"driver should be below (\d{2}) years", t, re.I): add("max_driver_age", int(m.group(1)), m, "motor")
    for m in re.finditer(r"below (\d{2}) years of age[^.]{0,60}?excess of Rs\.?\s*([\d,]+)", t, re.I):
        add("young_driver_age", int(m.group(1)), m, "motor"); add("young_driver_excess", NUM(m.group(2)), m, "motor")
    for m in re.finditer(r"maximum (?:limit )?Rs\.?\s*([\d,]{7,})[^.]{0,40}?(?:third party|private)", t, re.I): add("tppd_limit", NUM(m.group(1)), m, "motor")
    for m in re.finditer(r"(?:maximum|max\.?)\s*(?:entry )?age(?: at entry| of entry)?\D{0,25}(\d{2})\b", t, re.I): add("max_entry_age", int(m.group(1)), m)
    for m in re.finditer(r"entry age\D{0,30}(\d{1,2})\s*(?:to|-|–)\s*(\d{2})\b", t, re.I): add("min_entry_age", int(m.group(1)), m); add("max_entry_age", int(m.group(2)), m)
    for m in re.finditer(r"waiting period[^.\d]{0,50}(\d{1,3})\s*(days|months|years)", t, re.I):
        n = int(m.group(1)); add("waiting_days", n * {"days": 1, "months": 30, "years": 365}[m.group(2).lower()], m)
    for m in re.finditer(r"(?:above|over|beyond) (?:the age of )?(\d{2})[^.%]{0,60}?(\d{1,3}(?:\.\d+)?)\s*%\s*loading", t, re.I): add("loading_over_age", [int(m.group(1)), float(m.group(2))], m)
    for m in re.finditer(r"age (?:above|over) (\d{2}) years?\s+(\d{1,3}(?:\.\d+)?)\s*%\s*loading", t, re.I): add("loading_over_age", [int(m.group(1)), float(m.group(2))], m)
    for m in re.finditer(r"co-?pay(?:ment)?[^.\d]{0,30}(\d{1,2})\s*%", t, re.I): add("copay_pct", int(m.group(1)), m)
    for m in re.finditer(r"minimum premium[^.\d]{0,25}Rs\.?\s*([\d,]+)", t, re.I): add("min_premium_rs", NUM(m.group(1)), m)
    for m in re.finditer(r"medical (?:examination|report|test)s?[^.]{0,90}?sum (?:assured|insured)[^.]{0,30}?Rs\.?\s*([\d,]+)", t, re.I): add("medical_above_sa", NUM(m.group(1)), m, "life")
    seen, res = set(), []
    for f in out:
        k = (f["topic"], json.dumps(f["value"]))
        if k not in seen: seen.add(k); res.append(f)
    return res

def same(a, b):
    a, b = a.lower(), b.lower()
    return a in b or b in a

def load_facts(names):
    rows = []
    if os.path.exists("data/underwriting_rules.yaml"):
        for f in (yaml.safe_load(open("data/underwriting_rules.yaml")) or {}).get("facts", []):
            rows.append({**f, "origin": "Verified from source", "snippet": f.get("note", "")})
    if os.path.exists("index/facts.json"):
        rows += json.load(open("index/facts.json"))
    df = pd.DataFrame(rows, columns=["insurer", "category", "topic", "value", "snippet", "source", "origin"]) if rows else pd.DataFrame(columns=["insurer", "category", "topic", "value", "snippet", "source", "origin"])
    df["insurer"] = df["insurer"].map(lambda n: next((x for x in names if same(x, n)), n))
    df["rank"] = (df["origin"] != "Verified from source").astype(int)
    return df.sort_values("rank").drop(columns="rank").reset_index(drop=True)

def getv(df, insurer, cat, topic):
    r = df[(df.insurer == insurer) & (df.topic == topic) & (df.category.isin([cat, "General"]))]
    return None if r.empty else r.iloc[0]["value"]
