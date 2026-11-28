"""Market Explorer: every general and life insurance category -> which companies offer it, policies/pages,
posts, figures, and side-by-side comparisons. All numbers are computed from the loaded content."""
import re, warnings
import pandas as pd
warnings.filterwarnings("ignore", message="This pattern is interpreted")
import streamlit as st

CATS = {
    "General": {
        "Motor": r"motor|vehicle|car insurance|three[- ]wheel|motorcycle|third[- ]party|comprehensive cover",
        "Health & Medical": r"health|medical|hospitali[sz]ation|surgical|\bOPD\b|out-?patient",
        "Fire & Property": r"\bfire\b|property|home insurance|householder|burglary|flood|natural (disaster|perils)|all risks",
        "Marine & Cargo": r"marine|cargo|\bhull\b|goods in transit",
        "Engineering": r"engineering|contractors?.? all risks?|machinery breakdown|erection",
        "Liability": r"liability|professional indemnity|directors",
        "Personal Accident": r"personal accident|accidental death|permanent disability",
        "Travel": r"travel",
        "Agriculture & Crop": r"\bcrop|agricultur|livestock|farmer|paddy|cattle",
        "Miscellaneous": r"cash in transit|fidelity|mobile phone|gadget|workmen|workers.? compensation|jewel",
        "Takaful (Islamic)": r"takaful|shariah",
    },
    "Life": {
        "Term & Protection": r"term (life|insurance|assurance|plan)|protection plan|life cover|death benefit",
        "Endowment & Whole Life": r"endowment|whole life|money back|savings plan|maturity",
        "Investment-linked": r"investment[- ]linked|unit[- ]linked|universal (life )?plan|investment account",
        "Pension & Annuity": r"pension|annuity|retirement",
        "Child & Education": r"child|education|junior",
        "Critical Illness & Hospital": r"critical illness|serious illness|cancer|hospital cash|hospitali[sz]ation",
        "Group Life & Group Health": r"group life|group (health|medical|insurance)|corporate|employee",
        "Micro & Postal": r"micro|postal|low[- ]income|affordable",
    },
}
GENERIC = {"Hospitalisation": r"hospitali[sz]ation|in-?patient", "Accident": r"accident", "Critical illness": r"critical illness|serious illness",
           "Life cover": r"life (insurance|cover|assurance)|death benefit", "Online purchase": r"buy online|online (purchase|payment|quote|portal)|instant",
           "Digital / fast claims": r"claims? (app|online|digital)|instaclaim|on the spot|cashless", "Takaful": r"takaful|shariah"}
FEATS = {
    "Motor": {"Comprehensive": r"comprehensive", "Third-party": r"third[- ]party", "Motorcycle / 3-wheeler": r"motorcycle|three[- ]wheel",
              "Windscreen / glass": r"windscreen|glass", "Flood / natural disaster": r"flood|natural (disaster|perils)", "Riot / strike": r"riot|strike|civil commotion",
              "Replacement vehicle": r"replacement vehicle|similar vehicle", "Roadside / towing": r"roadside|towing|breakdown", "Agreed value": r"agreed value",
              "Bundled health / accident benefit": r"hospitali[sz]ation|accidental health|medical cover", "Fast claims": r"on the spot|claims? (app|online|digital)|doorstep"},
    "Health & Medical": {"Hospitalisation": r"hospitali[sz]ation|in-?patient", "Outpatient (OPD)": r"\bOPD\b|out-?patient", "Critical illness": r"critical illness|serious illness",
              "Maternity": r"maternity", "Dental / optical": r"dental|optical", "Ambulance": r"ambulance", "Overseas treatment": r"overseas|abroad",
              "No-claim bonus": r"no[- ]claim", "Cashless": r"cashless", "Buy online": r"buy online|online (purchase|portal)|one-click"},
    "Fire & Property": {"Fire": r"\bfire\b", "Flood": r"flood", "Burglary": r"burglary|theft", "Natural disasters": r"natural (disaster|perils)|earthquake|cyclone",
              "Contents": r"contents", "Business interruption": r"business interruption|consequential loss", "Riot / strike / terrorism": r"riot|strike|terroris",
              "Home / householder": r"home|householder", "SME package": r"\bSME\b|small business"},
    "Travel": {"Medical expenses": r"medical", "Trip delay / cancellation": r"delay|cancell", "Baggage loss": r"baggage|luggage", "Medical evacuation": r"evacuation|repatriation",
              "Personal liability": r"liability", "Schengen": r"schengen", "Adventure sports": r"adventure", "Annual multi-trip": r"annual|multi[- ]trip"},
    "Personal Accident": {"Accidental death": r"accidental death", "Permanent disability": r"permanent (total )?disab", "Medical expenses": r"medical",
              "Hospital cash": r"hospital cash|per day", "Students": r"student", "Tourists": r"tourist", "Weekly benefit": r"weekly"},
    "Term & Protection": {"Death benefit": r"death benefit", "Accidental death": r"accidental death", "Critical illness rider": r"critical illness",
              "Waiver of premium": r"waiver", "Hospital cash": r"hospital cash|hospitali[sz]ation", "Digital onboarding": r"online|app|digital"},
    "Investment-linked": {"Investment account": r"investment account", "Fund options": r"\bfunds?\b", "Partial withdrawal": r"withdraw", "Top-up": r"top[- ]up",
              "Death benefit": r"death benefit", "Fees / charges": r"\bfee|charge"},
    "Pension & Annuity": {"Retirement income": r"retirement", "Annuity": r"annuit", "Lump sum": r"lump sum", "Tax benefit": r"\btax", "Death benefit": r"death benefit"},
}
FIG = re.compile(r"(?:Rs\.?|LKR)\s?(\d[\d,]*(?:\.\d+)?)\s?(million|mn|billion|bn|thousand|m\b|k\b)?", re.I)
MULT = {"million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "thousand": 1e3, "k": 1e3}
ALL = {c: p for g in CATS.values() for c, p in g.items()}

def same(a, b):
    a, b = a.lower(), b.lower()
    return a in b or b in a

@st.cache_resource(show_spinner="Analysing loaded content...")
def chunk_table(_vs, ver, names):
    rows = []
    for d in _vs.docstore._dict.values():
        m = d.metadata
        ins = next((n for n in names if same(n, m["insurer"])), m["insurer"])
        rows.append({"insurer": ins, "kind": m.get("kind", "web"), "platform": m.get("platform", ""), "date": m.get("date", ""),
                     "source": m.get("source", ""), "title": m.get("title", ""), "text": d.page_content})
    df = pd.DataFrame(rows)
    for c, p in ALL.items():
        df[c] = df["text"].str.contains(p, case=False, regex=True)
    return df

def lkr(v, u):
    return float(v.replace(",", "")) * MULT.get((u or "").lower(), 1)

def fmt(x):
    return f"Rs. {x/1e9:.1f} bn" if x >= 1e9 else f"Rs. {x/1e6:.1f} mn" if x >= 1e6 else f"Rs. {x:,.0f}"

def explorer(vs, ver, names, llm):
    df = chunk_table(vs, ver, tuple(names))
    web = df[df.kind.isin(["web", "manual"]) | (df.platform == "manual")]
    st.title("📈 Market Explorer")
    st.caption("Built only from the content you loaded (websites, news, posts). Higher counts mean more coverage in that content, not a better product.")
    grp = st.radio("Insurance class", list(CATS), horizontal=True)
    cat = st.selectbox("Category", list(CATS[grp]))
    sub = web[web[cat]]
    posts = df[df.kind.isin(["social", "news"]) & df[cat]]
    rank = sub.groupby("insurer").agg(Mentions=("text", "count"), Pages=("source", "nunique")).sort_values("Mentions", ascending=False)
    total = len(set(names))
    c = st.columns(4)
    c[0].metric("Companies offering", f"{len(rank)} of {total}")
    c[1].metric("Pages found", int(sub.source.nunique()))
    c[2].metric("Posts / news", int(posts.source.nunique()))
    figs = []
    for _, r in pd.concat([sub, posts]).iterrows():
        for m in FIG.finditer(r.text):
            v = lkr(m.group(1), m.group(2))
            if v >= 1000:
                figs.append({"Insurer": r.insurer, "Figure": fmt(v), "LKR": v, "Context": "..." + r.text[max(0, m.start() - 70): m.end() + 70] + "...", "Source": r.source})
    fdf = pd.DataFrame(figs).drop_duplicates(["Insurer", "Figure", "Context"]) if figs else pd.DataFrame(columns=["Insurer", "Figure", "LKR", "Context", "Source"])
    c[3].metric("Figures found", len(fdf))
    t1, t2, t3, t4, t5 = st.tabs(["🏢 Companies & policies", "⚖️ Compare", "📣 Posts", "🔢 Figures", "🗺️ Market map"])
    with t1:
        if rank.empty:
            st.info("No loaded content mentions this category yet. Load more insurer pages or add manual files.")
        else:
            st.bar_chart(rank["Mentions"])
            st.dataframe(rank.reset_index().rename(columns={"insurer": "Insurer"}), hide_index=True, use_container_width=True)
            for ins in rank.index[:12]:
                with st.expander(f"{ins} - pages about {cat}"):
                    for (_, g) in sub[sub.insurer == ins].drop_duplicates("source").head(6).iterrows():
                        st.markdown(f"- [{g.title or g.source}]({g.source}): {g.text[:200]}...")
            if llm and st.button(f"List {cat} policies with AI"):
                ctx = "\n\n".join(f"[{r.insurer}] {r.text[:500]}" for _, r in sub.drop_duplicates("source").groupby("insurer").head(2).head(14).iterrows())
                try:
                    from langchain_core.messages import HumanMessage
                    st.markdown(llm.invoke([HumanMessage(f"Using ONLY these sources, list the {cat} insurance products or policies named, as a markdown table (Insurer | Product | Notable benefit). If none are named, say so. Do not invent.\n\n{ctx}")]).content)
                except Exception as e:
                    st.warning(f"AI error: {e}")
    with t2:
        feats = FEATS.get(cat, GENERIC)
        pick = st.multiselect("Companies to compare", list(rank.index), list(rank.index[:4]))
        if pick:
            rows, cnt = [], []
            for ins in pick:
                txt = " ".join(sub[sub.insurer == ins].text)
                hit = {f: bool(re.search(p, txt, re.I)) for f, p in feats.items()}
                rows.append({"Insurer": ins, **{f: ("✅" if v else "–") for f, v in hit.items()}})
                cnt.append({"Insurer": ins, "Features found": sum(hit.values()), "Mentions": int(rank.loc[ins, "Mentions"]), "Pages": int(rank.loc[ins, "Pages"])})
            cdf = pd.DataFrame(cnt).set_index("Insurer")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            a, b = st.columns(2)
            a.markdown("**Features found**"); a.bar_chart(cdf["Features found"])
            b.markdown("**Content coverage (mentions)**"); b.bar_chart(cdf["Mentions"])
            st.caption("✅ = mentioned on loaded pages; – = not found there. Confirm with the insurer.")
        else:
            st.info("Pick at least one company.")
    with t3:
        if posts.empty:
            st.info("No posts or news about this category loaded. Add posts in data/social/*.csv or set the Facebook / YouTube keys, then reload.")
        else:
            a, b = st.columns(2)
            a.markdown("**Posts per company**"); a.bar_chart(posts.drop_duplicates("source").groupby("insurer").size())
            dt = pd.to_datetime(posts.drop_duplicates("source").date, errors="coerce").dropna()
            if len(dt):
                b.markdown("**Posts per month**"); b.line_chart(dt.dt.to_period("M").astype(str).value_counts().sort_index())
            for _, p in posts.drop_duplicates("source").sort_values("date", ascending=False).head(15).iterrows():
                st.markdown(f"- **{p.insurer}** · {p.platform} · {p.date}: {p.text[:220]}... [open]({p.source})")
    with t4:
        if fdf.empty:
            st.info("No Rs. figures found for this category in the loaded content.")
        else:
            a, b = st.columns(2)
            a.markdown("**Largest figure quoted per company (Rs.)**"); a.bar_chart(fdf.groupby("Insurer")["LKR"].max())
            b.markdown("**Number of figures quoted**"); b.bar_chart(fdf.groupby("Insurer").size())
            st.dataframe(fdf.sort_values("LKR", ascending=False).drop(columns=["LKR"]), hide_index=True, use_container_width=True)
            st.caption("Figures are picked up automatically from text (benefit limits, claims paid, prices...). Read the context before comparing them.")
    with t5:
        mat = {ct: web[web[ct]].groupby("insurer").size() for ct in CATS[grp]}
        m = pd.DataFrame(mat).fillna(0).astype(int)
        st.markdown(f"**Companies per {grp.lower()} category**")
        st.bar_chart((m > 0).sum().sort_values(ascending=False))
        lvl = lambda n: "–" if n == 0 else "●" if n < 3 else "●●" if n < 6 else "●●●"
        st.dataframe((m.map if hasattr(m, "map") else m.applymap)(lvl).reset_index().rename(columns={"insurer": "Insurer"}), hide_index=True, use_container_width=True)
        st.caption("● = a little content, ●●● = a lot of content about that category on the company's loaded pages.")
