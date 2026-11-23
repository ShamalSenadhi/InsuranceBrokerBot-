"""Load insurer content by URL -> split -> embed -> FAISS. Also indexes data/manual/*.txt."""
import json, os, pathlib, re, time, urllib.robotparser as rp
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urljoin, urldefrag
import requests, yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

UA = "BrokerAdvisorStudentProject/0.2"
EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KEY = re.compile(r"product|insur|plan|cover|motor|vehicle|health|medical|life|travel|fire|accident|personal|individual|corporate|sme|takaful|protect|saving|pension", re.I)
SKIP = re.compile(r"\.(pdf|jpe?g|png|gif|zip|docx?|xlsx?)(\?|$)|login|career|news|tender|csr|annual-report|investor|complaint|branch", re.I)
MAX_PAGES = 8
_robots = {}

def allowed(url):
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in _robots:
        r = rp.RobotFileParser(base + "/robots.txt")
        try:
            requests.get(base + "/robots.txt", headers={"User-Agent": UA}, timeout=10)  # fail fast if site is down
            r.read()
        except Exception:
            r = None
        _robots[base] = r
    r = _robots[base]
    return True if r is None else r.can_fetch(UA, url)

def get(url):
    try:
        if not allowed(url):
            return None
        h = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        h.raise_for_status()
        return h.text
    except Exception:
        return None

def text_of(soup):
    for t in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())

def crawl(ins):
    home = ins["url"]
    host = urlparse(home).netloc.replace("www.", "")
    html = get(home)
    if not html:
        return ins, [], "unreachable or blocked"
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        u = urldefrag(urljoin(home, a["href"]))[0]
        if urlparse(u).netloc.replace("www.", "") != host or SKIP.search(u) or u.rstrip("/") == home.rstrip("/"):
            continue
        if KEY.search(u) or KEY.search(a.get_text(" ")):
            if u not in links:
                links.append(u)
    docs = []
    for u in [home] + links[: MAX_PAGES - 1]:
        page = html if u == home else get(u)
        if not page:
            continue
        s = BeautifulSoup(page, "html.parser")
        title = s.title.get_text(strip=True) if s.title else ""
        txt = text_of(s)
        if len(txt) > 200:
            docs.append(Document(page_content=txt, metadata={
                "insurer": ins["name"], "type": ins.get("type", ""), "title": title, "source": u}))
        time.sleep(0.5)
    return ins, docs, f"{len(docs)} pages" if docs else "no readable text (JS site?) - add manual file"

def build_index(progress=None):
    cfg = yaml.safe_load(open("insurers.yaml"))["insurers"]
    docs, status, done = [], [], 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for ins, d, msg in ex.map(crawl, cfg):
            docs += d
            status.append({"insurer": ins["name"], "url": ins["url"], "result": msg})
            done += 1
            if progress:
                progress(done / len(cfg), f"{ins['name']}: {msg}")
    for f in pathlib.Path("data/manual").glob("*__*.txt"):
        name, cat = f.stem.split("__", 1)
        docs.append(Document(page_content=f.read_text(encoding="utf-8"), metadata={
            "insurer": name, "type": "manual", "title": cat, "source": f"manual:{f.name}"}))
    if not docs:
        raise RuntimeError("No content could be loaded.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name=EMB)).save_local("index")
    json.dump(status, open("index/status.json", "w"))
    return status

if __name__ == "__main__":
    for s in build_index(lambda p, m: print(f"{p:.0%} {m}")):
        pass
