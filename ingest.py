"""Collect insurer content -> split -> embed -> FAISS index.
Sources: (1) public web pages listed in insurers.yaml (robots.txt respected)
         (2) manual files in data/manual/<Insurer>__<category>.txt
             (paste Facebook/LinkedIn posts, brochure text here by hand)."""
import pathlib, urllib.robotparser as rp
from urllib.parse import urlparse
import requests, yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

UA = "BrokerAdvisorStudentProject/0.1"
EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # handles Sinhala/Tamil better than English-only models

def allowed(url):
    p = urlparse(url)
    r = rp.RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        r.read()
        return r.can_fetch(UA, url)
    except Exception:
        return False

def fetch(url):
    h = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    h.raise_for_status()
    s = BeautifulSoup(h.text, "html.parser")
    for t in s(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()
    return " ".join(s.get_text(" ").split())

def main():
    docs = []
    cfg = yaml.safe_load(open("insurers.yaml"))
    for ins in cfg["insurers"]:
        for pg in ins["pages"]:
            url = pg["url"]
            if not allowed(url):
                print("SKIP (robots/unreachable):", url); continue
            try:
                docs.append(Document(page_content=fetch(url), metadata={
                    "insurer": ins["name"], "type": ins["type"],
                    "category": pg["category"], "source": url}))
                print("OK", url)
            except Exception as e:
                print("FAIL", url, e)
    for f in pathlib.Path("data/manual").glob("*__*.txt"):
        name, cat = f.stem.split("__", 1)
        docs.append(Document(page_content=f.read_text(encoding="utf-8"), metadata={
            "insurer": name, "type": "manual", "category": cat, "source": f"manual:{f.name}"}))
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name=EMB)).save_local("index")
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents")

if __name__ == "__main__":
    main()
