import os, streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
st.set_page_config(page_title="Broker Advisor Dashboard", layout="wide")
st.title("Broker Advisor Dashboard (prototype)")
st.caption("Aggregates public insurer content and suggests products. Suggestions are a starting point for the broker, not final advice.")

@st.cache_resource
def load():
    return FAISS.load_local("index", HuggingFaceEmbeddings(model_name=EMB),
                            allow_dangerous_deserialization=True)  # index built locally by ingest.py
try:
    vs = load()
except Exception:
    st.error("No index found. Run: python ingest.py"); st.stop()

with st.sidebar:
    st.header("Customer requirements")
    need = st.multiselect("Cover needed", ["Motor", "Health / medical", "Life / protection", "Property / fire", "Travel", "Business / SME"], ["Motor"])
    age = st.number_input("Age", 18, 90, 35)
    dependants = st.number_input("Dependants", 0, 10, 2)
    budget = st.text_input("Budget (LKR / year)", "")
    notes = st.text_area("Other details (assets, conditions, concerns)")
    k = st.slider("Chunks to retrieve", 4, 20, 10)
    go = st.button("Suggest products", type="primary")

if go:
    query = f"Insurance products for: {', '.join(need)}. Age {age}, {dependants} dependants, budget {budget}. {notes}"
    hits = vs.similarity_search(query, k=k)
    ctx = "\n\n".join(f"[{i+1}] ({d.metadata['insurer']}, {d.metadata['category']}) {d.page_content}" for i, d in enumerate(hits))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Suggested products")
        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), temperature=0, max_tokens=1200)
            prompt = ("You support a Sri Lankan insurance broker. Using ONLY the sources below, shortlist up to 5 suitable "
                      "products for the customer. For each: insurer, product, why it fits, and source numbers like [2]. "
                      "If sources lack price, exclusions or eligibility, say 'not stated - broker to confirm'. "
                      "Do not invent products, prices or benefits.\n\n"
                      f"CUSTOMER: {query}\n\nSOURCES:\n{ctx}")
            st.markdown(llm.invoke(prompt).content)
        else:
            st.info("Set ANTHROPIC_API_KEY for AI-written suggestions. Showing ranked source passages instead.")
            for i, d in enumerate(hits):
                st.markdown(f"**{d.metadata['insurer']}** ({d.metadata['category']}): {d.page_content[:300]}...")
        with st.expander("Sources used"):
            for i, d in enumerate(hits):
                st.markdown(f"[{i+1}] {d.metadata['insurer']} - {d.metadata['source']}")
    with right:
        st.subheader("Broker value-add checklist")
        st.markdown("""
- Is the sum insured adequate (replacement value, income replacement)?
- Exclusions, waiting periods, co-pays: confirmed in the actual policy wording?
- Claims record and service quality of each insurer?
- Existing cover or gaps (AI cannot see these)?
- Premium vs. cover trade-off explained to the client?
- Regulatory points: disclosure, duty to the client, IRCSL rules.
""")
