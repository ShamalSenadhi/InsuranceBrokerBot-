"""Underwrite & Quote: collect customer details -> illustrative underwriting -> quotations per insurer -> compare -> PDF.
IMPORTANT: insurers do not publish exact premiums. Rates here are either real figures YOU enter (data/rates.csv or the table)
or clearly-marked placeholder assumptions. Customer details stay in this browser session and are not sent to any LLM."""
import hashlib, io, os, re
from datetime import date
import pandas as pd
import streamlit as st
from explorer import chunk_table, FEATS, GENERIC, FIG, lkr, fmt, same

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
NCB = {0: 0, 1: 20, 2: 30, 3: 40, 4: 50}  # illustrative scale

def band(x, table):
    for lim, v in table:
        if x < lim: return v
    return table[-1][1]

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
    p["ncb"] = c2.number_input("Claim-free years", 0, 10, 3)
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

# ---------------- illustrative underwriting rules ----------------
def underwrite(cat, p):
    L, refer, docs, qs = [], [], [], []
    add = lambda txt, x: L.append((txt, x))
    basis, addons = 0, {}
    if cat == "Motor":
        basis = p["value"]; age = date.today().year - p["year"]
        if p["dage"] < 25: add("Young driver (under 25)", 25)
        if p["dage"] >= 70: add("Driver aged 70+", 15); refer.append("Driver aged 70+")
        if age > 10: add("Vehicle older than 10 years", 10)
        if age > 15: refer.append("Vehicle older than 15 years - survey likely")
        if p["usage"].startswith("Hire"): add("Hire / rent use", 30)
        if p["usage"].startswith("Comm"): add("Commercial / goods use", 20)
        if p["claims"] == 1: add("1 claim in last 3 years", 10)
        if p["claims"] == 2: add("2 claims in last 3 years", 25); refer.append("2 claims in 3 years")
        if p["claims"] >= 3: add("3+ claims in last 3 years", 50); refer.append("3+ claims in 3 years")
        if p["value"] > 20000000: refer.append("High value vehicle (over Rs 20m) - survey")
        if p["cover"].startswith("Third"): p = {**p, "addons": []}
        addons = {a: MOTOR_ADDONS[a] for a in p["addons"]}
        docs = ["Vehicle registration (CR) copy", "Driving licence", "Previous policy / claim-free letter", "Valuation or survey report (if requested)"]
        qs = ["Is the vehicle modified or imported unregistered?", "Any traffic convictions or licence endorsements?", "Where is it parked overnight?"]
    elif cat == "Health":
        basis = p["limit"] * p["persons"]; a = p["age"]
        ld = band(a, [(30, 0), (40, 10), (50, 30), (60, 60), (70, 100), (999, 150)])
        if ld: add(f"Age {a}", ld)
        if a >= 60: refer.append("Age 60+")
        if a >= 70: refer.append("Age 70+ - special terms likely")
        if p["smoker"] == "Yes": add("Smoker", 15)
        if p["preex"] == "Yes": add("Existing condition declared", 25); refer.append("Existing condition - exclusion, waiting period or loading likely")
        addons = {a: HEALTH_ADDONS[a] for a in p["addons"]}
        docs = ["NIC / passport copy", "Completed health declaration", "Recent medical reports (if age 50+ or condition declared)"]
        qs = ["Any hospital stays or surgery in last 5 years?", "Any regular medication?", "Family history to disclose?"]
    elif cat == "Life (term)":
        basis = p["sa"]; a = p["age"]
        ld = band(a, [(30, 0), (40, 15), (50, 50), (60, 120), (999, 200)])
        if ld: add(f"Age {a}", ld)
        if a >= 60: refer.append("Age 60+")
        if p["smoker"] == "Yes": add("Smoker", 50)
        if p["occ"].startswith("Medium"): add("Medium risk occupation", 15)
        if p["occ"].startswith("High"): add("Hazardous occupation", 40); refer.append("Hazardous occupation")
        if p["term"] > 25: add("Long term (over 25 years)", 10)
        if p["sa"] > 10000000: refer.append("Sum assured over Rs 10m - medical exam likely")
        if p["sa"] > 25000000: refer.append("Sum assured over Rs 25m - financial underwriting")
        addons = {a: LIFE_ADDONS[a] for a in p["addons"]}
        docs = ["NIC copy", "Proposal and health questionnaire", "Income proof (for larger sums)", "Medical report (if requested)"]
        qs = ["Existing life cover with other companies?", "Any hazardous hobbies (diving, racing...)?", "Who are the beneficiaries?"]
    elif cat == "Travel":
        basis = 0; a = p["age"]
        add("Destination: " + p["region"], {"Asia": 0, "Europe / Schengen": 30, "USA / Canada": 80, "Worldwide": 50}[p["region"]])
        add("Medical cover USD " + p["medical"], {"50,000": 0, "100,000": 20, "250,000": 50}[p["medical"]])
        if a >= 65: add("Traveller aged 65+", 60)
        if a >= 75: add("Traveller aged 75+", 60); refer.append("Traveller aged 75+")
        if a >= 85: refer.append("Traveller aged 85+ - may not be accepted")
        if p["adv"] == "Yes": add("Adventure sports", 25)
        docs = ["Passport copy", "Itinerary / visa letter", "Pre-existing condition declaration"]
        qs = ["Any planned medical treatment abroad?", "Which adventure activities exactly?", "Any pre-existing conditions?"]
    elif cat == "Home & Fire":
        basis = p["building"] + p["contents"]
        if p["flood"] == "Yes": add("Flood-prone area", 30)
        if p["age"] > 30: add("Building older than 30 years", 10); refer.append("Old building - survey likely")
        if p["use"].startswith("Business"): add("Business use", 25); refer.append("Business use")
        if p["alarm"] == "Yes": add("Alarm / security (discount)", -5)
        if basis > 100000000: refer.append("High value property (over Rs 100m)")
        docs = ["Title deed / ownership proof", "Building plan or valuation", "Photos of the property"]
        qs = ["Roof / wall construction?", "Any past flood or fire loss?", "Is the property occupied full time?"]
    else:  # Personal Accident
        basis = p["si"]; a = p["age"]
        c = int(p["occ"][0]); add("Occupation class " + str(c), {1: 0, 2: 30, 3: 80}[c])
        if c == 3: refer.append("Hazardous occupation")
        if a > 65: add("Age over 65", 50); refer.append("Age over 65")
        docs = ["NIC copy", "Occupation details"]; qs = ["Any existing disability?", "Any hazardous sports?"]
    load = sum(x for _, x in L) / 100
    decision = "Special terms likely" if any("special terms" in r or "may not be accepted" in r for r in refer) else "Refer to underwriter" if refer else "Standard terms"
    return dict(basis=basis, load=load, reasons=L, refer=refer, decision=decision, docs=docs, qs=qs, addons=addons)

def premium(cat, p, uw, rate, minp, levy):
    days = p.get("days", 1); trav = p.get("travellers", 1)
    base = rate * days * trav if cat == "Travel" else rate / 100 * uw["basis"]
    if cat == "Motor" and p["cover"].startswith("Third"): base *= 0.35  # TP priced as fraction of comprehensive (assumption)
    loaded = base * (1 + uw["load"])
    addons = sum(pc for pc, _ in uw["addons"].values()) / 100 * base
    ncb = NCB.get(min(p.get("ncb", 0), 4), 0) / 100 * loaded if cat == "Motor" else 0
    sub = max(loaded + addons - ncb, minp); lv = sub * levy / 100
    return dict(base=base, loading=loaded - base, addons=addons, ncb=-ncb, min_applied=sub == minp and (loaded + addons - ncb) < minp, levies=lv, total=sub + lv)

# ---------------- PDF ----------------
def make_pdf(cat, p, uw, ins, q, name, levy, rate_note):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    st_ = getSampleStyleSheet(); buf = io.BytesIO()
    ref = "QT-" + date.today().strftime("%Y%m%d") + "-" + hashlib.md5(f"{cat}{p}{ins}".encode()).hexdigest()[:5].upper()
    d = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    tb = lambda rows, w=None: (lambda t: (t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke), ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP")])), t)[1])(Table(rows, colWidths=w))
    inputs = [[k.replace("_", " ").title(), ", ".join(v) if isinstance(v, list) else str(v)] for k, v in p.items()]
    pb = [["Item", "Amount (Rs)"], ["Base premium", f"{q['base']:,.0f}"], ["Risk loading", f"{q['loading']:,.0f}"], ["Add-ons / riders", f"{q['addons']:,.0f}"],
          ["No-claim discount", f"{q['ncb']:,.0f}"]] + ([["Minimum premium applied", "yes"]] if q["min_applied"] else []) + [[f"Taxes / levies ({levy}%)", f"{q['levies']:,.0f}"], ["TOTAL ANNUAL PREMIUM", f"{q['total']:,.0f}"]]
    story = [Paragraph("Indicative Quotation", st_["Title"]), Paragraph(f"Ref: {ref} &nbsp; Date: {date.today():%d %B %Y}", st_["Normal"]), Spacer(1, 8),
             Paragraph(f"<b>Prepared for:</b> {name or 'Customer'} &nbsp; <b>Insurer:</b> {ins} &nbsp; <b>Class:</b> {cat}", st_["Normal"]), Spacer(1, 8),
             Paragraph("Customer details", st_["Heading3"]), tb(inputs, [150, 360]), Spacer(1, 8),
             Paragraph(f"Underwriting view: {uw['decision']} (risk loading {uw['load']*100:.0f}%)", st_["Heading3"]),
             Paragraph("; ".join(r for r in uw["refer"]) or "No referral triggers found.", st_["Normal"]), Spacer(1, 8),
             Paragraph("Premium breakdown", st_["Heading3"]), tb(pb, [330, 180]), Spacer(1, 8),
             Paragraph("Documents usually required: " + ", ".join(uw["docs"]), st_["Normal"]), Spacer(1, 8),
             Paragraph(f"Rate basis: {rate_note}", st_["Italic"]), Spacer(1, 6),
             Paragraph("This is an indicative estimate produced by a student prototype dashboard. It is NOT an offer, quotation or contract of insurance. Final premium, terms and acceptance are decided by the insurer after underwriting. Please confirm all details with a licensed insurance broker.", st_["Italic"])]
    d.build(story); return buf.getvalue()

# ---------------- view ----------------
def rates_for(cat, cfg, offering, names, real):
    rows = []
    for n in names:
        r = next((x for x in real if x["category"] == cat and same(x["insurer"], n)), None)
        rows.append({"Include": bool(r) or n in offering, "Insurer": n, f"Rate ({cfg['unit']})": float(r["rate"]) if r else cfg["base"],
                     "Min premium (Rs)": float(r["min_premium"]) if r and str(r.get("min_premium", "")).strip() else cfg["minp"],
                     "Basis of figure": r["source"] if r else "ASSUMPTION - replace with a real quote"})
    return pd.DataFrame(rows)

def quote_view(vs, ver, names):
    st.title("🧮 Underwrite & Quote")
    st.caption("Insurers do not publish exact premiums, so quotes here combine your customer's details, simple illustrative underwriting rules, and rates you enter from real quotes. Customer details stay in this session and are not sent to any AI model.")
    df = chunk_table(vs, ver, tuple(names))
    web = df[df.kind.isin(["web", "manual"]) | (df.platform == "manual")]
    cat = st.selectbox("1. What does the customer want to insure?", list(CATCFG)); cfg = CATCFG[cat]
    with st.form("uw_" + cat):
        st.markdown("**2. Customer details for underwriting**")
        p = FORMS[cat]()
        c1, c2 = st.columns(2)
        levy = c1.number_input("Taxes / levies on premium (%) - enter the actual rate", 0.0, 50.0, 0.0)
        name = c2.text_input("Customer name (optional, for the quotation)")
        go = st.form_submit_button("Assess risk and build quotes", type="primary")
    if go: st.session_state["uw"] = dict(cat=cat, p=p, levy=levy, name=name)
    S = st.session_state.get("uw")
    if not S or S["cat"] != cat:
        st.info("Fill in the details and press the button."); return
    p, levy, name = S["p"], S["levy"], S["name"]
    uw = underwrite(cat, p)
    st.markdown("### 3. Underwriting assessment (illustrative rules)")
    (st.success if uw["decision"] == "Standard terms" else st.warning)(f"**{uw['decision']}**  |  total risk loading **{uw['load']*100:.0f}%**")
    a, b = st.columns(2)
    with a:
        st.markdown("**Why**")
        st.dataframe(pd.DataFrame(uw["reasons"] or [("No loadings", 0)], columns=["Factor", "Loading %"]), hide_index=True)
        for r in uw["refer"]: st.markdown(f"- ⚠️ {r}")
    with b:
        st.markdown("**Documents usually needed**"); st.markdown("\n".join(f"- {d}" for d in uw["docs"]))
        st.markdown("**Questions the underwriter may ask**"); st.markdown("\n".join(f"- {d}" for d in uw["qs"]))
    st.caption("These rules are simplified examples to show the process. Real insurers use their own underwriting guidelines.")

    ex = cfg["ex"]; sub = web[web[ex]]
    offering = set(sub.insurer.unique())
    real = []
    if os.path.exists("data/rates.csv"):
        real = pd.read_csv("data/rates.csv").fillna("").to_dict("records")
    st.markdown("### 4. Rates per insurer")
    st.caption("Edit the table with REAL rates from quotes, brochures or insurer calculators. Rows marked ASSUMPTION use one generic placeholder, so they do not show real price differences. You can also put real rates in data/rates.csv (columns: category,insurer,rate,min_premium,source).")
    key = f"rates_{cat}"
    if key not in st.session_state:
        st.session_state[key] = rates_for(cat, cfg, offering, names, real)
    ed = st.data_editor(st.session_state[key], hide_index=True, key=key + "_ed")
    rate_col = f"Rate ({cfg['unit']})"
    feats = FEATS.get(ex, GENERIC); want = [v[1] for v in uw["addons"].values()]
    rows = []
    for _, r in ed[ed["Include"]].iterrows():
        q = premium(cat, p, uw, float(r[rate_col]), float(r["Min premium (Rs)"]), levy)
        txt = " ".join(sub[sub.insurer.apply(lambda x: same(x, r["Insurer"]))].text)
        hit = {f: bool(re.search(pt, txt, re.I)) for f, pt in feats.items()}
        got = [w for w in want if hit.get(w)]
        rows.append(dict(Insurer=r["Insurer"], total=q["total"], q=q, found=sum(hit.values()), want=len(want), got=len(got),
                         missing=[w for w in want if not hit.get(w)], basis=r["Basis of figure"], offers=r["Insurer"] in offering or bool(txt)))
    if not rows:
        st.info("Tick at least one insurer in the table."); return
    mn = min(x["total"] for x in rows); mf = max(x["found"] for x in rows) or 1
    for x in rows:
        fs = (x["got"] / x["want"]) if x["want"] else x["found"] / mf
        x["value"] = round(100 * (0.5 * mn / x["total"] + 0.5 * fs))
    out = pd.DataFrame([{"Insurer": x["Insurer"], "Annual premium (Rs)": round(x["total"]), "Monthly (Rs)": round(x["total"] / 12),
                         "Features found on site": x["found"], "Requested features found": f"{x['got']}/{x['want']}" if x["want"] else "n/a",
                         "Not found (confirm)": ", ".join(x["missing"]) or "-", "Value score": x["value"],
                         "Figure basis": "Assumption" if x["basis"].startswith("ASSUMPTION") else "Real quote"} for x in rows]).sort_values("Value score", ascending=False)
    st.markdown("### 5. Compare quotations")
    if (out["Figure basis"] == "Assumption").any():
        st.warning("Some premiums use placeholder rates. Enter real quotes in the table above before relying on the comparison.")
    st.dataframe(out, hide_index=True)
    x1, x2, x3 = st.columns(3)
    x1.markdown("**Annual premium (Rs)**"); x1.bar_chart(out.set_index("Insurer")["Annual premium (Rs)"])
    x2.markdown("**Features found on site**"); x2.bar_chart(out.set_index("Insurer")["Features found on site"])
    x3.markdown("**Value score (100 = best)**"); x3.bar_chart(out.set_index("Insurer")["Value score"])
    st.caption("Value score = 50% price (cheapest = full marks) + 50% features (share of requested add-ons found on the insurer's pages, or features found vs. the best insurer if no add-ons were asked for). It is a rough guide, not advice.")
    cheap, best = out.sort_values("Annual premium (Rs)").iloc[0], out.iloc[0]
    st.success(f"💰 Cheapest: **{cheap['Insurer']}** (Rs {cheap['Annual premium (Rs)']:,})   |   🏆 Best value: **{best['Insurer']}** (score {best['Value score']})")
    st.download_button("Download comparison (CSV)", out.to_csv(index=False).encode(), "quotation_comparison.csv", "text/csv")

    st.markdown("### 6. Generate quotation")
    pick = st.selectbox("Insurer", list(out["Insurer"]))
    x = next(r for r in rows if r["Insurer"] == pick); q = x["q"]
    st.dataframe(pd.DataFrame([("Base premium", q["base"]), ("Risk loading", q["loading"]), ("Add-ons / riders", q["addons"]), ("No-claim discount", q["ncb"]), (f"Taxes / levies ({levy}%)", q["levies"]), ("TOTAL ANNUAL PREMIUM", q["total"])], columns=["Item", "Rs"]).assign(Rs=lambda d: d.Rs.map(lambda v: f"{v:,.0f}")), hide_index=True)
    try:
        pdf = make_pdf(cat, p, uw, pick, q, name, levy, x["basis"])
        st.download_button("⬇️ Download quotation (PDF)", pdf, f"quotation_{pick.split()[0].lower()}.pdf", "application/pdf", type="primary")
    except Exception as e:
        st.warning(f"PDF could not be created: {e}")

    with st.expander("Premium-related figures found in the loaded content for this category"):
        fg = []
        for _, r in sub.iterrows():
            for m in FIG.finditer(r.text):
                ctx = r.text[max(0, m.start() - 90): m.end() + 90]
                if re.search(r"premium|per month|monthly|per year|annual|starting", ctx, re.I) and lkr(m.group(1), m.group(2)) >= 500:
                    fg.append({"Insurer": r.insurer, "Figure": fmt(lkr(m.group(1), m.group(2))), "Context": "..." + ctx + "...", "Source": r.source})
        st.dataframe(pd.DataFrame(fg).drop_duplicates(["Insurer", "Figure"]) if fg else pd.DataFrame([{"Note": "None found. Use insurer calculators or ask for quotes, then enter them in the table."}]), hide_index=True)
