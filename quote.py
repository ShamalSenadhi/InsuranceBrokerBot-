"""Underwrite & Quote. Underwriting differs by insurer, so this view uses each insurer's OWN published terms (policy documents, IPIDs,
brochures) gathered into data/underwriting_rules.yaml and index/facts.json. Insurers do not publish exact premiums: premiums come from
REAL quotes you enter, or are clearly-marked placeholder estimates. Customer details stay in this session; no AI model sees them."""
import hashlib, io, json, os, re
from datetime import date
import pandas as pd
import streamlit as st
from explorer import chunk_table, FEATS, GENERIC, FIG, lkr, fmt, same
from facts import load_facts, getv

CATCFG = {
    "Motor": dict(ex="Motor", unit="% of vehicle value / yr", base=3.0, minp=15000),
    "Health": dict(ex="Health & Medical", unit="% of cover limit per person / yr", base=1.5, minp=10000),
    "Life (term)": dict(ex="Term & Protection", unit="% of sum assured / yr", base=0.20, minp=6000),
    "Travel": dict(ex="Travel", unit="Rs per person per day", base=500.0, minp=2000),
    "Home & Fire": dict(ex="Fire & Property", unit="% of insured value / yr", base=0.15, minp=5000),
    "Personal Accident": dict(ex="Personal Accident", unit="% of sum insured / yr", base=0.30, minp=3000),
}
MOTOR_ADDONS = {"Windscreen / glass": (2, "Windscreen / glass"), "Flood / natural disaster": (5, "Flood / natural disaster"),
                "Riot / strike": (3, "Riot / strike"), "Roadside / towing": (2, "Roadside / towing"), "Replacement vehicle": (4, "Replacement vehicle")}
HEALTH_ADDONS = {"Outpatient (OPD)": (20, "Outpatient (OPD)"), "Maternity": (25, "Maternity"), "Dental / optical": (8, "Dental / optical")}
LIFE_ADDONS = {"Critical illness rider": (35, "Critical illness rider"), "Accidental death rider": (8, "Accidental death")}

# ---------------- forms ----------------
def f_motor():
    c1, c2 = st.columns(2); p = {}
    p["vtype"] = c1.selectbox("Vehicle type", ["Car", "SUV / Jeep", "Van", "Motorcycle", "Three-wheeler", "Lorry / goods", "Bus"])
    p["year"] = c2.number_input("Year of manufacture", 1985, date.today().year, 2018)
    p["value"] = c1.number_input("Market value (Rs)", 100000, 500000000, 6000000, step=100000)
    p["usage"] = c2.selectbox("Usage", ["Private", "Hire / rent", "Commercial / goods"])
    p["cover"] = c1.selectbox("Cover", ["Comprehensive", "Third-party only"])
    p["dage"] = c2.number_input("Main driver age", 18, 90, 35)
    p["claims"] = c1.number_input("Claims in last 3 years", 0, 10, 0)
    p["ncb"] = c2.number_input("Current no-claim bonus % (from last policy; 0 if none)", 0, 75, 0, step=5)
    p["addons"] = st.multiselect("Add-ons wanted", list(MOTOR_ADDONS)); return p

def f_health():
    c1, c2 = st.columns(2); p = {}
    p["age"] = c1.number_input("Age of main person", 0, 90, 35)
    p["persons"] = c2.number_input("People to cover (incl. you)", 1, 10, 1)
    p["limit"] = c1.number_input("Cover limit per person (Rs)", 100000, 100000000, 1000000, step=100000)
    p["smoker"] = c2.selectbox("Smoker?", ["No", "Yes"])
    p["preex"] = c1.selectbox("Any existing medical condition?", ["No", "Yes"])
    p["addons"] = st.multiselect("Extras wanted", list(HEALTH_ADDONS)); return p

def f_life():
    c1, c2 = st.columns(2); p = {}
    p["age"] = c1.number_input("Age", 18, 80, 32)
    p["sa"] = c2.number_input("Sum assured (Rs)", 500000, 500000000, 5000000, step=500000)
    p["term"] = c1.number_input("Policy term (years)", 5, 40, 20)
    p["smoker"] = c2.selectbox("Smoker?", ["No", "Yes"])
    p["occ"] = c1.selectbox("Occupation risk", ["Low (office)", "Medium (field / driving)", "High (hazardous)"])
    p["addons"] = st.multiselect("Riders wanted", list(LIFE_ADDONS)); return p

def f_travel():
    c1, c2 = st.columns(2); p = {}
    p["region"] = c1.selectbox("Destination", ["Asia", "Europe / Schengen", "USA / Canada", "Worldwide"])
    p["days"] = c2.number_input("Trip length (days)", 1, 365, 10)
    p["travellers"] = c1.number_input("Travellers", 1, 10, 1)
    p["age"] = c2.number_input("Oldest traveller's age", 0, 90, 35)
    p["medical"] = c1.selectbox("Medical cover (USD)", ["50,000", "100,000", "250,000"])
    p["adv"] = c2.selectbox("Adventure sports?", ["No", "Yes"]); p["addons"] = []; return p

def f_home():
    c1, c2 = st.columns(2); p = {}
    p["building"] = c1.number_input("Building value (Rs)", 0, 1000000000, 15000000, step=500000)
    p["contents"] = c2.number_input("Contents value (Rs)", 0, 500000000, 2000000, step=250000)
    p["age"] = c1.number_input("Building age (years)", 0, 100, 10)
    p["flood"] = c2.selectbox("Flood-prone area?", ["No", "Yes"])
    p["use"] = c1.selectbox("Use", ["Residence", "Business / shop"])
    p["alarm"] = c2.selectbox("Alarm / security guard?", ["No", "Yes"]); p["addons"] = []; return p

def f_pa():
    c1, c2 = st.columns(2); p = {}
    p["age"] = c1.number_input("Age", 1, 90, 35)
    p["si"] = c2.number_input("Sum insured (Rs)", 100000, 200000000, 2000000, step=100000)
    p["occ"] = c1.selectbox("Occupation class", ["1 - Office / clerical", "2 - Field / light manual", "3 - Heavy manual / hazardous"])
    p["addons"] = []; return p

FORMS = {"Motor": f_motor, "Health": f_health, "Life (term)": f_life, "Travel": f_travel, "Home & Fire": f_home, "Personal Accident": f_pa}


EXPECTED = {"Motor": ["ncb_max_pct", "excess_by_vehicle_age", "max_driver_age"], "Health": ["max_entry_age", "waiting_days", "loading_over_age"],
            "Life (term)": ["max_entry_age", "medical_above_sa"], "Travel": ["max_age"], "Home & Fire": ["min_premium_rs"], "Personal Accident": ["max_entry_age"]}
LABEL = {"ncb_max_pct": "Max no-claim bonus", "excess_by_vehicle_age": "Excess by vehicle age", "max_driver_age": "Driver age limit", "max_entry_age": "Maximum entry age",
         "waiting_days": "Waiting period", "loading_over_age": "Age loading", "medical_above_sa": "Medical exam threshold", "max_age": "Maximum age", "min_premium_rs": "Minimum premium",
         "excess_hire": "Hire excess", "excess_rental": "Rental excess", "excess_motorcycle": "Motorcycle excess", "young_driver_age": "Young driver age", "young_driver_excess": "Young driver excess",
         "tppd_limit": "Third-party property limit", "copay_pct": "Co-payment %", "min_entry_age": "Minimum entry age", "ncb_pct_per_year": "NCB per year"}

def basis(cat, p):
    return {"Motor": lambda: p["value"], "Health": lambda: p["limit"] * p["persons"], "Life (term)": lambda: p["sa"],
            "Home & Fire": lambda: p["building"] + p["contents"], "Personal Accident": lambda: p["si"]}.get(cat, lambda: 0)()

def assess(ins, cat, p, F):
    """Apply THIS insurer's published terms to the customer. Nothing is assumed: missing terms are listed as gaps."""
    G = lambda t: getv(F, ins, cat, t)
    r = dict(parts=[], loading=0.0, ncb_cap=G("ncb_max_pct"), notes=[]); ok, bad = [], []
    def chk(val, lim, label, upper=True):
        if lim is None: return
        (ok if (val <= lim if upper else val >= lim) else bad).append(f"{label} {val} vs published limit {lim}")
    if cat == "Motor":
        vage = date.today().year - p["year"]; ex = G("excess_by_vehicle_age")
        if ex:
            amt = max([rs for a, rs in ex if vage >= a], default=0)
            if amt: r["parts"].append((f"Vehicle {vage} years old", amt))
        if p["vtype"] == "Motorcycle" and G("excess_motorcycle"): r["parts"].append(("Motorcycle", G("excess_motorcycle")))
        if p["usage"].startswith("Hire") and G("excess_hire"): r["parts"].append(("Hire use", G("excess_hire")))
        yd, ye = G("young_driver_age"), G("young_driver_excess")
        if yd and ye and p["dage"] < yd: r["parts"].append((f"Driver under {yd}", ye))
        md = G("max_driver_age")
        if md is not None: (ok if p["dage"] < md else bad).append(f"driver age {p['dage']} vs published limit below {md}")
        if G("tppd_limit"): r["notes"].append(f"Third-party property limit Rs {G('tppd_limit'):,.0f}")
    elif cat == "Health":
        chk(p["age"], G("max_entry_age"), "age", True)
        lo = G("loading_over_age")
        if lo and p["age"] >= lo[0]: r["loading"] = float(lo[1]); r["notes"].append(f"Published age loading {lo[1]}% (age {lo[0]}+)")
        if G("waiting_days"): r["notes"].append(f"Waiting period {int(G('waiting_days'))} days")
        if G("copay_pct"): r["notes"].append(f"Co-payment {G('copay_pct')}%")
    elif cat == "Life (term)":
        chk(p["age"], G("max_entry_age"), "age", True)
        ms = G("medical_above_sa")
        if ms and p["sa"] >= ms: r["notes"].append(f"Medical exam required (sum assured Rs {p['sa']:,.0f} is at/above Rs {ms:,.0f})")
    elif cat == "Travel":
        chk(p["age"], G("max_age"), "age", True)
    elif cat == "Personal Accident":
        chk(p["age"], G("max_entry_age"), "age", True)
    r["excess"] = sum(v for _, v in r["parts"])
    r["status"] = "Outside published limits" if bad else "Within published limits" if ok else "Limits not published"
    r["checks"] = ok + bad
    r["gaps"] = [LABEL[t] for t in EXPECTED[cat] if G(t) is None]
    return r

def premium(cat, p, A, row, rate_col, levy):
    quoted = float(row["Quoted premium (Rs/yr)"] or 0)
    if quoted > 0:
        return dict(base=quoted, loading=0, addons=0, ncb=0, min_applied=False, levies=0, total=quoted, real=True)
    rate = float(row[rate_col])
    base = rate * p.get("days", 1) * p.get("travellers", 1) if cat == "Travel" else rate / 100 * basis(cat, p)
    loaded = base * (1 + (float(row["Risk loading % (as quoted)"]) + A["loading"]) / 100)
    addons = base * float(row["Add-ons %"]) / 100
    ncb_pct = min(p.get("ncb", 0), A["ncb_cap"]) if (cat == "Motor" and A["ncb_cap"] is not None) else (p.get("ncb", 0) if cat == "Motor" else 0)
    ncb = ncb_pct / 100 * loaded
    raw = loaded + addons - ncb; sub = max(raw, float(row["Min premium (Rs)"])); lv = sub * levy / 100
    return dict(base=base, loading=loaded - base, addons=addons, ncb=-ncb, min_applied=sub > raw, levies=lv, total=sub + lv, real=False)

# ---------------- PDF ----------------
def make_pdf(cat, p, ins, A, q, name, levy, basis_note, sources):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    st_ = getSampleStyleSheet(); buf = io.BytesIO()
    ref = "QT-" + date.today().strftime("%Y%m%d") + "-" + hashlib.md5(f"{cat}{p}{ins}".encode()).hexdigest()[:5].upper()
    d = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    def tb(rows, w):
        t = Table([[Paragraph(str(c), st_["BodyText"]) for c in r] for r in rows], colWidths=w)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke), ("VALIGN", (0, 0), (-1, -1), "TOP")])); return t
    inputs = [[k.replace("_", " ").title(), ", ".join(v) if isinstance(v, list) else str(v)] for k, v in p.items()]
    pb = [["Item", "Amount (Rs)"], ["Premium (quoted / base)", f"{q['base']:,.0f}"]]
    if not q["real"]:
        pb += [["Risk loading", f"{q['loading']:,.0f}"], ["Add-ons", f"{q['addons']:,.0f}"], ["No-claim discount", f"{q['ncb']:,.0f}"], [f"Taxes / levies ({levy}%)", f"{q['levies']:,.0f}"]]
    pb += [["TOTAL ANNUAL PREMIUM", f"{q['total']:,.0f}"]]
    terms = [[ "Eligibility", A["status"] + ("; " + "; ".join(A["checks"]) if A["checks"] else "")],
             ["Excess", (f"Rs {A['excess']:,.0f} (" + "; ".join(f"{a}: Rs {b:,.0f}" for a, b in A["parts"]) + ")") if A["parts"] else "Not published / none found"],
             ["Max no-claim bonus", f"{A['ncb_cap']}%" if A["ncb_cap"] is not None else "Not published"], ["Other published terms", "; ".join(A["notes"]) or "-"],
             ["Not published (ask insurer)", ", ".join(A["gaps"]) or "-"]]
    story = [Paragraph("Indicative Quotation", st_["Title"]), Paragraph(f"Ref: {ref} &nbsp; Date: {date.today():%d %B %Y}", st_["Normal"]), Spacer(1, 8),
             Paragraph(f"<b>Prepared for:</b> {name or 'Customer'} &nbsp; <b>Insurer:</b> {ins} &nbsp; <b>Class:</b> {cat}", st_["Normal"]), Spacer(1, 8),
             Paragraph("Customer details", st_["Heading3"]), tb(inputs, [150, 360]), Spacer(1, 8),
             Paragraph(f"{ins}: published underwriting terms applied", st_["Heading3"]), tb(terms, [150, 360]), Spacer(1, 8),
             Paragraph("Premium", st_["Heading3"]), tb(pb, [330, 180]), Spacer(1, 6),
             Paragraph(f"Premium basis: {basis_note}", st_["Italic"]), Spacer(1, 6),
             Paragraph("Sources of terms: " + "; ".join(sources[:5]) if sources else "Sources: none published.", st_["Italic"]), Spacer(1, 6),
             Paragraph("This is an indicative estimate from a student prototype dashboard. It is NOT an offer, quotation or contract of insurance. The final premium, terms and acceptance are decided by the insurer after its own underwriting. Confirm everything with a licensed insurance broker.", st_["Italic"])]
    d.build(story); return buf.getvalue()

# ---------------- view ----------------
def rates_for(cat, cfg, p, offering, names, real, F):
    rows = []; tp = cat == "Motor" and p.get("cover", "").startswith("Third")
    for n in names:
        r = next((x for x in real if x["category"] == cat and same(x["insurer"], n)), None)
        mp = getv(F, n, cat, "min_premium_rs")
        rows.append({"Include": bool(r) or n in offering, "Insurer": n,
                     f"Rate ({cfg['unit']})": float(r["rate"]) if r and str(r["rate"]).strip() else (1.0 if tp else cfg["base"]),
                     "Min premium (Rs)": float(r["min_premium"]) if r and str(r.get("min_premium", "")).strip() else float(mp or cfg["minp"]),
                     "Risk loading % (as quoted)": 0.0, "Add-ons %": 0.0,
                     "Quoted premium (Rs/yr)": float(r["quoted_premium"]) if r and str(r.get("quoted_premium", "")).strip() else 0.0,
                     "Basis of figure": (r["source"] or "Real figure (data/rates.csv)") if r else "ASSUMPTION - replace with a real quote"})
    return pd.DataFrame(rows)

def quote_view(vs, ver, names):
    st.title("🧮 Underwrite & Quote")
    st.caption("Underwriting differs by company, so every check below uses that insurer's own published terms (gathered from policy documents, IPIDs and brochures, each with its source). Exact premiums are not published: use real quotes in step 5. Customer details stay in this session; no AI model sees them.")
    df = chunk_table(vs, ver, tuple(names)); F = load_facts(names)
    web = df[df.kind.isin(["web", "manual", "policydoc"]) | (df.platform == "manual")]
    cat = st.selectbox("1. What does the customer want to insure?", list(CATCFG)); cfg = CATCFG[cat]
    with st.form("uw_" + cat):
        st.markdown("**2. Customer details**")
        p = FORMS[cat]()
        c1, c2 = st.columns(2)
        levy = c1.number_input("Taxes / levies on estimated premiums (%) - enter the actual rate", 0.0, 50.0, 0.0)
        name = c2.text_input("Customer name (optional, for the quotation)")
        go = st.form_submit_button("Apply every insurer's underwriting terms", type="primary")
    if go: st.session_state["uw"] = dict(cat=cat, p=p, levy=levy, name=name)
    S = st.session_state.get("uw")
    if not S or S["cat"] != cat:
        st.info("Fill in the details and press the button."); return
    p, levy, name = S["p"], S["levy"], S["name"]
    ex = cfg["ex"]; sub = web[web[ex]]; offering = set(sub.insurer.unique())
    fc = F[F.category.isin([cat])]
    st.markdown("### 3. Underwriting terms gathered from each insurer")
    if fc.empty:
        st.warning("No underwriting terms found yet. Reload the content with 'Include policy PDFs' ticked (sidebar), or add terms to data/underwriting_rules.yaml.")
    else:
        cnt = fc.groupby(["insurer", "origin"]).size().unstack(fill_value=0)
        st.bar_chart(cnt)
        tab = fc.assign(Term=fc.topic.map(lambda t: LABEL.get(t, t)), Value=fc.value.astype(str), Source=fc.source)[["insurer", "Term", "Value", "origin", "Source", "snippet"]].rename(columns={"insurer": "Insurer", "origin": "Status", "snippet": "Evidence"})
        st.dataframe(tab, hide_index=True, column_config={"Source": st.column_config.LinkColumn("Source", display_text="open")})
        st.caption("'Verified from source' = checked by hand against the insurer's document. 'Auto-extracted - verify' = pulled from text by pattern matching: open the source and check before relying on it. Missing terms are not published or not found.")
    try:
        ql = [x for x in json.load(open("index/quote_links.json")) if x["insurer"] in offering]
        if ql:
            with st.expander("Official quote / proposal pages found on insurer sites (get real quotes here)"):
                st.markdown("\n".join(f"- **{x['insurer']}**: [{x['label']}]({x['url']})" for x in ql))
    except Exception:
        pass

    cand = sorted(offering | set(fc.insurer.unique()))
    A = {n: assess(n, cat, p, F) for n in cand}
    st.markdown("### 4. This customer under each insurer's published rules")
    st.dataframe(pd.DataFrame([{"Insurer": n, "Status": a["status"], "Excess (Rs)": a["excess"], "Excess detail": "; ".join(f"{x}: {y:,.0f}" for x, y in a["parts"]) or "-",
                                "Max NCB": f"{a['ncb_cap']}%" if a["ncb_cap"] is not None else "n/p", "Published loading": f"{a['loading']}%" if a["loading"] else "-",
                                "Other terms": "; ".join(a["notes"]) or "-", "Not published (ask insurer)": ", ".join(a["gaps"]) or "-"} for n, a in A.items()]), hide_index=True)
    st.caption("n/p = not published. Excess items shown are those that apply to this customer; confirm with the insurer whether they add up or apply alternatively. Generic underwriting guesses are NOT used.")
    with st.expander("Documents usually needed (generic) and underwriter questions"):
        st.markdown("Usual documents: " + {"Motor": "vehicle registration (CR), driving licence, previous policy / NCB letter, valuation if requested.", "Health": "NIC / passport, health declaration, medical reports if requested.",
                    "Life (term)": "NIC, proposal and health questionnaire, income proof and medical report for larger sums.", "Travel": "passport, itinerary, pre-existing condition declaration.",
                    "Home & Fire": "title deed, valuation, photos.", "Personal Accident": "NIC, occupation details."}[cat])

    real = pd.read_csv("data/rates.csv").fillna("").to_dict("records") if os.path.exists("data/rates.csv") else []
    st.markdown("### 5. Premiums (real quotes)")
    st.caption("For each insurer, enter the REAL quoted premium (best), or a real rate. Rows marked ASSUMPTION use one generic placeholder rate, so they cannot show real price differences. Real data can also go in data/rates.csv (category,insurer,rate,min_premium,quoted_premium,source).")
    key = f"rates2_{cat}"
    if key not in st.session_state:
        st.session_state[key] = rates_for(cat, cfg, p, offering | set(cand), cand, real, F)
    ed = st.data_editor(st.session_state[key], hide_index=True, key=key + "_ed")
    rate_col = f"Rate ({cfg['unit']})"
    feats = FEATS.get(ex, GENERIC); want = []
    rows = []
    for _, r in ed[ed["Include"]].iterrows():
        a = A.get(r["Insurer"]) or assess(r["Insurer"], cat, p, F)
        q = premium(cat, p, a, r, rate_col, levy)
        txt = " ".join(sub[sub.insurer.apply(lambda x: same(x, r["Insurer"]))].text)
        found = sum(bool(re.search(pt, txt, re.I)) for pt in feats.values())
        rows.append(dict(ins=r["Insurer"], q=q, a=a, found=found, basis=r["Basis of figure"], real=q["real"] or not str(r["Basis of figure"]).startswith("ASSUMPTION")))
    if not rows:
        st.info("Tick at least one insurer in the table."); return
    elig = [x for x in rows if x["a"]["status"] != "Outside published limits"] or rows
    mn = min(x["q"]["total"] for x in elig); mf = max(x["found"] for x in rows) or 1
    for x in rows: x["value"] = round(100 * (0.5 * mn / x["q"]["total"] + 0.5 * x["found"] / mf)) if x in elig else 0
    out = pd.DataFrame([{"Insurer": x["ins"], "Status": x["a"]["status"], "Annual premium (Rs)": round(x["q"]["total"]), "Monthly (Rs)": round(x["q"]["total"] / 12), "Excess (Rs)": x["a"]["excess"],
                         "Max NCB": f"{x['a']['ncb_cap']}%" if x["a"]["ncb_cap"] is not None else "n/p", "Features found on site": x["found"], "Value score": x["value"],
                         "Figure basis": "Real quote / rate" if x["real"] else "Assumption"} for x in rows]).sort_values("Value score", ascending=False)
    st.markdown("### 6. Compare quotations")
    if (out["Figure basis"] == "Assumption").any(): st.warning("Some premiums use placeholder rates. Enter real quotes above before relying on this comparison.")
    st.dataframe(out, hide_index=True)
    x1, x2, x3 = st.columns(3)
    x1.markdown("**Annual premium (Rs)**"); x1.bar_chart(out.set_index("Insurer")["Annual premium (Rs)"])
    x2.markdown("**Excess (Rs)**"); x2.bar_chart(out.set_index("Insurer")["Excess (Rs)"])
    x3.markdown("**Value score (100 = best)**"); x3.bar_chart(out.set_index("Insurer")["Value score"])
    st.caption("Value score = 50% price (cheapest eligible = full marks) + 50% features found on the insurer's pages. Insurers outside their published limits score 0. A rough guide, not advice.")
    ok = out[out["Status"] != "Outside published limits"]
    if not ok.empty:
        cheap, best = ok.sort_values("Annual premium (Rs)").iloc[0], ok.iloc[0]
        st.success(f"💰 Cheapest eligible: **{cheap['Insurer']}** (Rs {cheap['Annual premium (Rs)']:,})   |   🏆 Best value: **{best['Insurer']}** (score {best['Value score']})")
    st.download_button("Download comparison (CSV)", out.to_csv(index=False).encode(), "quotation_comparison.csv", "text/csv")

    st.markdown("### 7. Generate quotation")
    pick = st.selectbox("Insurer", list(out["Insurer"])); x = next(r for r in rows if r["ins"] == pick); q = x["q"]
    items = [("Premium (quoted / base)", q["base"])] + ([] if q["real"] else [("Risk loading", q["loading"]), ("Add-ons", q["addons"]), ("No-claim discount", q["ncb"]), (f"Taxes / levies ({levy}%)", q["levies"])]) + [("TOTAL ANNUAL PREMIUM", q["total"])]
    st.dataframe(pd.DataFrame(items, columns=["Item", "Rs"]).assign(Rs=lambda d: d.Rs.map(lambda v: f"{v:,.0f}")), hide_index=True)
    srcs = list(dict.fromkeys(fc[fc.insurer == pick].source.tolist()))
    try:
        pdf = make_pdf(cat, p, pick, x["a"], q, name, levy, x["basis"], srcs)
        st.download_button("⬇️ Download quotation (PDF)", pdf, f"quotation_{pick.split()[0].lower()}.pdf", "application/pdf", type="primary")
    except Exception as e:
        st.warning(f"PDF could not be created: {e}")
