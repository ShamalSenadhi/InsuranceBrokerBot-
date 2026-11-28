import json, os, re, pandas as pd, streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ingest
from explorer import explorer
from quote import quote_view

st.set_page_config(page_title="Insurance Broker Chatbot", page_icon="💬", layout="wide")
for k in ("GROQ_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "FB_ACCESS_TOKEN", "YOUTUBE_API_KEY"):
    try:
        os.environ.setdefault(k, st.secrets[k])
    except Exception:
        pass

SYSTEM = """You are a friendly assistant for a licensed Sri Lankan insurance broker. Use ONLY the SOURCES below.
Rules:
1. If the message is not about insurance or the sources do not cover it, say so in one sentence and ask what they want to insure.
2. If basics are missing (what to cover, who/what is covered, age or dependants, budget), ask at most 2 short questions. No recommendation yet.
3. Otherwise answer in EXACTLY this format, in simple words, under 150 words:
🏆 **Best pick: <product> - <insurer>** [n]
*<one short, catchy, friendly line>*
**Why it wins**
- <three short bullets tied to what the customer said>
**How the others compare**
| Option | Good for | Trade-off |
|---|---|---|
<2 or 3 rows: other products from the sources, with [n]>
**Check with your broker:** <one short line>
BEST_INSURER: <exact insurer name of the best pick>
4. Never invent products, prices or benefits. If something is not in the sources write 'not stated - broker to confirm'.
5. Reply in the customer's language (English, Sinhala or Tamil), but keep the BEST_INSURER line in English."""

def ollama_up():
    try:
        import requests
        return requests.get(os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/tags", timeout=1).ok
    except Exception:
        return False

class PoolLLM:
    """Keyless free LLMs via freellmpool (Pollinations, OVHcloud, Kilo...; auto-failover, also uses any keys you set)."""
    def __init__(self):
        from freellmpool import Pool
        self.pool = Pool.from_default_config()
    def invoke(self, messages):
        from types import SimpleNamespace
        role = {"SystemMessage": "Instructions", "HumanMessage": "Customer", "AIMessage": "Assistant"}
        prompt = "\n\n".join(f"{role.get(type(m).__name__, 'Customer')}: {m.content}" for m in messages) + "\n\nAssistant:"
        return SimpleNamespace(content=self.pool.ask(prompt).text)

def get_llm():
    """Order: local Ollama -> keyless free pool (no API key) -> Groq -> Gemini -> Anthropic. LLM_PROVIDER=ollama|pool|groq forces one."""
    want = os.getenv("LLM_PROVIDER", "").lower()
    if want == "ollama" or (not want and ollama_up()):
        from langchain_ollama import ChatOllama
        get_llm.name = "Ollama (local)"
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), temperature=0, num_predict=900,
                          base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    if want in ("", "pool"):
        try:
            llm = PoolLLM()
            get_llm.name = "Free pool (no key)"
            return llm
        except Exception:
            pass
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        get_llm.name = "Groq (free tier)"
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0, max_tokens=900)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        get_llm.name = "Gemini (free tier)"
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=0, max_output_tokens=900)
    return None

@st.cache_resource
def load_index():
    return FAISS.load_local("index", ingest.embeddings(), allow_dangerous_deserialization=True)

def status():
    try:
        return json.load(open("index/status.json"))
    except Exception:
        return []

with st.sidebar:
    page = st.radio("View", ["💬 Chatbot", "📈 Market Explorer", "🧮 Underwrite & Quote"])
    st.header("Insurer content")
    pdfs = st.checkbox("Include policy PDFs (underwriting terms; slower)", value=True)
    if st.button("Load / refresh content from insurer URLs", type="primary"):
        bar = st.progress(0.0)
        try:
            ingest.build_index(lambda p, m: bar.progress(p, text=m), include_pdfs=pdfs)
            load_index.clear()
            st.success("Content loaded.")
        except Exception as e:
            st.error(str(e))
    st.dataframe(status(), hide_index=True)
    try:
        with st.expander("Official social pages found on insurer sites"):
            st.dataframe(json.load(open("index/social_links.json")), hide_index=True)
    except Exception:
        pass
    names = [s["insurer"] for s in status()]
    only = st.multiselect("Limit suggestions to insurers (optional)", names)
    if st.button("Clear chat"):
        st.session_state.pop("msgs", None)
    llm = get_llm()
    st.caption("AI model: " + (get_llm.name if llm else "none - pip install freellmpool or install Ollama"))

st.title("💬 Insurance Broker Chatbot")
st.caption("Content is loaded from Sri Lankan insurers' websites and posts. Suggestions are indicative; confirm every detail with a licensed broker.")

try:
    vs = load_index()
except Exception:
    st.info("No content loaded yet. Click 'Load / refresh content from insurer URLs' in the sidebar (takes a few minutes).")
    st.stop()

FEATURES = {
    "Hospitalisation": r"hospitali[sz]ation|in-?patient",
    "Outpatient (OPD)": r"\bOPD\b|out-?patient",
    "Critical illness": r"critical illness|serious illness|cancer",
    "Accident cover": r"accident",
    "Maternity": r"maternity",
    "Dental / optical": r"dental|optical",
    "Life cover": r"life (insurance|cover|assurance)|death benefit",
    "Motor": r"motor|vehicle|car insurance",
    "Travel": r"travel",
    "Property / fire": r"\bfire\b|property|home insurance",
    "Buy online": r"buy online|online (purchase|payment|quote|portal)|instant",
    "Digital / fast claims": r"claims? (app|online|digital)|instaclaim|on the spot|cashless",
    "Takaful (Shariah)": r"takaful|shariah",
}

def same(a, b):
    a, b = a.lower(), b.lower()
    return a in b or b in a

def build_viz(scored, best):
    sims = {}
    for h, d in scored:
        if h.metadata.get("kind", "web") not in ("web", "policydoc"):
            continue
        n = h.metadata["insurer"]
        key = next((k for k in sims if same(k, n)), n)
        sims.setdefault(key, []).append(1 / (1 + d))
    if not sims:
        return None
    score = {k: sum(sorted(v, reverse=True)[:3]) / min(3, len(v)) for k, v in sims.items()}
    top = sorted(score, key=score.get, reverse=True)[:5]
    b = next((k for k in score if best and same(k, best)), None)
    if b and b not in top:
        top = top[:4] + [b]
    mx = max(score[k] for k in top)
    docs = [d for d in vs.docstore._dict.values() if d.metadata.get("kind", "web") in ("web", "policydoc")]
    match, feat, links = [], [], []
    for k in top:
        label = ("🏆 " if k == b else "") + k
        txt = " ".join(d.page_content for d in docs if same(k, d.metadata["insurer"]))
        hit = {f: bool(re.search(p, txt, re.I)) for f, p in FEATURES.items()}
        match.append({"Insurer": label, "Match %": round(100 * score[k] / mx)})
        feat.append({"Insurer": label, **{f: ("✅" if v else "–") for f, v in hit.items()}, "Features found": sum(hit.values())})
        pg = next((h for h, _ in scored if same(k, h.metadata["insurer"]) and h.metadata.get("kind", "web") in ("web", "policydoc")), None)
        if pg:
            links.append(f"- **{label}**: [{pg.metadata.get('title') or pg.metadata['source']}]({pg.metadata['source']})")
    return {"match": pd.DataFrame(match), "feat": pd.DataFrame(feat), "links": links}

def render_viz(v):
    st.markdown("### 📊 Compare at a glance")
    t1, t2, t3, t4 = st.tabs(["Match score", "Feature table", "Feature count", "Pages"])
    with t1:
        st.bar_chart(v["match"].set_index("Insurer")["Match %"])
        st.caption("How closely each insurer's loaded content matches your request (best = 100). It is not a quality rating.")
    with t2:
        st.dataframe(v["feat"].drop(columns=["Features found"]), hide_index=True)
        st.caption("✅ = mentioned on the loaded pages. – = not found there (it may still exist). Confirm with the broker.")
    with t3:
        st.bar_chart(v["feat"].set_index("Insurer")["Features found"])
    with t4:
        st.markdown("\n".join(v["links"]))

if page.startswith("🧮"):
    quote_view(vs, os.path.getmtime("index/index.faiss"), names)
    st.stop()

if page.startswith("📈"):
    explorer(vs, os.path.getmtime("index/index.faiss"), names, llm)
    st.stop()

if "msgs" not in st.session_state:
    st.session_state.msgs = [{"role": "assistant", "content": "Hello! Tell me what you would like to insure (vehicle, health, life, travel, property or business) and a little about your situation, and I will suggest suitable products."}]
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                st.markdown("\n".join(m["sources"]))
        if m.get("viz"):
            render_viz(m["viz"])

if q := st.chat_input("Describe your insurance needs..."):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    user_turns = [m["content"] for m in st.session_state.msgs if m["role"] == "user"][-4:]
    query = " ".join(user_turns)
    if llm and not get_llm.name.startswith("Free pool"):  # (skipped on the keyless pool to save its tiny rate limits)
        # rewrite the conversation into one clean English search query
        try:
            query = llm.invoke([HumanMessage("Rewrite this customer conversation as one short English search query for finding insurance products. Output only the query.\n\n" + "\n".join(user_turns))]).content.strip() or query
        except Exception:
            pass
    scored = vs.similarity_search_with_score(query, k=60)
    if only:
        scored = [(h, d) for h, d in scored if h.metadata["insurer"] in only]
    pool = [h for h, _ in scored]
    products = [h for h in pool if h.metadata.get("kind", "web") in ("web", "policydoc")][:6]
    posts = [h for h in pool if h.metadata.get("kind") in ("social", "news")][:4]
    ctx_docs = products + posts[:2]
    ctx = "\n\n".join(f"[{i+1}] {h.metadata['insurer']} ({h.metadata.get('platform','')} {h.metadata.get('date','')}) - {h.page_content[:700]}" for i, h in enumerate(ctx_docs))
    srcs = [f"[{i+1}] {h.metadata['insurer']} - {h.metadata['source']}" for i, h in enumerate(ctx_docs)]
    with st.chat_message("assistant"):
        reply, best = None, None
        if llm:
            hist = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in st.session_state.msgs[-7:-1]]
            try:
                with st.spinner("Finding your best match..."):
                    reply = llm.invoke([SystemMessage(SYSTEM + "\n\nSOURCES:\n" + ctx)] + hist + [HumanMessage(q)]).content
            except Exception as e:
                st.warning(f"AI model error (free tier limit?): {e}")
        if reply is None:
            reply = "Closest passages from insurer sites (add an AI key for a best-pick answer):\n\n" + "\n\n".join(f"**{h.metadata['insurer']}**: {h.page_content[:300]}..." for h in products[:4])
        mt = re.search(r"BEST_INSURER:\s*(.+)", reply)
        if mt:
            cand = mt.group(1).strip().lower()
            reply = reply[:mt.start()].rstrip()
            best = next((n for n in names if n.lower() in cand or cand in n.lower()), mt.group(1).strip())
        st.markdown(reply)
        with st.expander("Sources"):
            st.markdown("\n".join(srcs))
        if best:
            bl = best.lower()
            pp = [h for h in vs.similarity_search(f"{best} {query}", k=80)
                  if h.metadata.get("kind") in ("social", "news") and (h.metadata["insurer"].lower() in bl or bl in h.metadata["insurer"].lower())][:3]
            st.markdown(f"**📣 Latest from {best}**")
        else:
            pp = posts[:3]
            if pp: st.markdown("**📣 Related posts and updates**")
        for h in pp:
            m = h.metadata
            st.markdown(f"- {m.get('platform','')} · {m.get('date','')}: {h.page_content[:220]}... [open]({m['source']})")
        if best and not pp:
            try:
                lk = next((x for x in json.load(open("index/social_links.json")) if x["insurer"] == best), {})
                lk = {k: v for k, v in lk.items() if k != "insurer"}
                st.caption("No posts loaded for this insurer yet. Official pages: " + (" · ".join(f"[{k}]({v})" for k, v in lk.items()) or "none found"))
            except Exception:
                pass
        viz = build_viz(scored, best)
        if viz:
            render_viz(viz)
    st.session_state.msgs.append({"role": "assistant", "content": reply, "sources": srcs, "viz": viz})
