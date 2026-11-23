import json, os, streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ingest

st.set_page_config(page_title="Insurance Broker Chatbot", page_icon="💬", layout="wide")
for k in ("ANTHROPIC_API_KEY", "FB_ACCESS_TOKEN", "YOUTUBE_API_KEY"):
    try:
        os.environ.setdefault(k, st.secrets[k])
    except Exception:
        pass

SYSTEM = """You are an assistant for a licensed Sri Lankan insurance broker. You help a customer find suitable insurance products from the companies registered with IRCSL, using ONLY the SOURCES provided.
Behaviour:
1. If you do not yet know the basics (what they want to cover, who/what is covered, age or dependants, rough budget, main worries), ask at most two short questions. Do not recommend yet.
2. Once you have enough, suggest 3 to 5 products: insurer, product name, why it fits, and source numbers like [2].
3. If the sources do not state a price, limits, waiting periods or exclusions, say 'not stated - broker to confirm'. Never invent products, prices or benefits.
4. End recommendations with the key things the customer should check with the broker (exclusions, sum insured adequacy, claims record).
Be concise and friendly. Reply in the customer's language (English, Sinhala or Tamil)."""

@st.cache_resource
def load_index():
    return FAISS.load_local("index", HuggingFaceEmbeddings(model_name=ingest.EMB), allow_dangerous_deserialization=True)

def status():
    try:
        return json.load(open("index/status.json"))
    except Exception:
        return []

with st.sidebar:
    st.header("Insurer content")
    if st.button("Load / refresh content from insurer URLs", type="primary"):
        bar = st.progress(0.0)
        try:
            ingest.build_index(lambda p, m: bar.progress(p, text=m))
            load_index.clear()
            st.success("Content loaded.")
        except Exception as e:
            st.error(str(e))
    st.dataframe(status(), hide_index=True, use_container_width=True)
    try:
        with st.expander("Official social pages found on insurer sites"):
            st.dataframe(json.load(open("index/social_links.json")), hide_index=True)
    except Exception:
        pass
    names = [s["insurer"] for s in status()]
    only = st.multiselect("Limit suggestions to insurers (optional)", names)
    if st.button("Clear chat"):
        st.session_state.pop("msgs", None)

st.title("💬 Insurance Broker Chatbot")
st.caption("Content is loaded from Sri Lankan insurers' websites. Suggestions are indicative; confirm every detail with a licensed broker.")

try:
    vs = load_index()
except Exception:
    st.info("No content loaded yet. Click 'Load / refresh content from insurer URLs' in the sidebar (takes a few minutes).")
    st.stop()

if "msgs" not in st.session_state:
    st.session_state.msgs = [{"role": "assistant", "content": "Hello! Tell me what you would like to insure (vehicle, health, life, travel, property or business) and a little about your situation, and I will suggest suitable products."}]
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                st.markdown("\n".join(m["sources"]))

if q := st.chat_input("Describe your insurance needs..."):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    user_turns = [m["content"] for m in st.session_state.msgs if m["role"] == "user"][-4:]
    hits = vs.similarity_search(" ".join(user_turns), k=30)
    if only:
        hits = [h for h in hits if h.metadata["insurer"] in only]
    posts = [h for h in hits if h.metadata.get("kind") in ("social", "news")][:5]
    hits = hits[:10]
    ctx = "\n\n".join(f"[{i+1}] {h.metadata['insurer']} ({h.metadata.get('platform','')}, {h.metadata.get('date','')}) - {h.page_content}" for i, h in enumerate(hits))
    srcs = [f"[{i+1}] {h.metadata['insurer']} - {h.metadata['source']}" for i, h in enumerate(hits)]
    with st.chat_message("assistant"):
        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), temperature=0, max_tokens=1200)
            hist = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in st.session_state.msgs[-9:-1]]
            msgs = [SystemMessage(SYSTEM + "\n\nSOURCES:\n" + ctx)] + hist + [HumanMessage(q)]
            with st.spinner("Thinking..."):
                reply = llm.invoke(msgs).content
        else:
            reply = "No API key set, so here are the closest passages from insurer sites:\n\n" + "\n\n".join(f"**{h.metadata['insurer']}**: {h.page_content[:300]}..." for h in hits[:5])
        st.markdown(reply)
        with st.expander("Sources"):
            st.markdown("\n".join(srcs))
        if posts:
            st.markdown("**Related posts and updates**")
            for h in posts:
                m = h.metadata
                st.markdown(f"- **{m['insurer']}** · {m.get('platform','')} · {m.get('date','')}: {h.page_content[:220]}... [open]({m['source']})")
    st.session_state.msgs.append({"role": "assistant", "content": reply, "sources": srcs})
