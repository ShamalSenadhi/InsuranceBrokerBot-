"""Load insurer content -> split -> embed -> FAISS.
Sources: (1) website pages incl. news/promotion pages, (2) social links found on each site,
(3) social posts via OFFICIAL routes only: CSV import, Facebook Graph API, YouTube Data API,
(4) manual text files."""
import csv, json, os, pathlib, re, time, urllib.robotparser as rp
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urljoin, urldefrag
import requests, yaml
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

UA = "BrokerAdvisorStudentProject/0.3"
EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KEY = re.compile(r"product|insur|plan|cover|motor|vehicle|health|medical|life|travel|fire|accident|personal|individual|corporate|sme|takaful|protect|saving|pension|news|promo|offer|blog|media|update|campaign", re.I)
NEWS = re.compile(r"news|promo|offer|blog|media|update|campaign", re.I)
SKIP = re.compile(r"\.(pdf|jpe?g|png|gif|zip|docx?|xlsx?)(\?|$)|login|career|tender|csr|annual-report|investor|complaint|branch", re.I)
SOCIAL = {"facebook": "facebook.com", "linkedin": "linkedin.com", "youtube": "youtube.com", "instagram": "instagram.com", "tiktok": "tiktok.com", "x": "twitter.com"}
MAX_PAGES = 10

@lru_cache(maxsize=1)
def embeddings():
    return HuggingFaceEmbeddings(model_name=EMB)  # one shared copy keeps memory low on free hosting

_robots = {}

def allowed(url):
    p = urlparse(url); base = f"{p.scheme}://{p.netloc}"
    if base not in _robots:
        r = rp.RobotFileParser(base + "/robots.txt")
        try:
            requests.get(base + "/robots.txt", headers={"User-Agent": UA}, timeout=10); r.read()
        except Exception:
            r = None
        _robots[base] = r
    return True if _robots[base] is None else _robots[base].can_fetch(UA, url)

def get(url):
    try:
        if not allowed(url): return None
        h = requests.get(url, headers={"User-Agent": UA}, timeout=20); h.raise_for_status()
        return h.text
    except Exception:
        return None

def text_of(soup):
    for t in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]): t.decompose()
    return " ".join(soup.get_text(" ").split())

def D(ins, text, source, kind="web", platform="website", date="", title=""):
    return Document(page_content=text, metadata={"insurer": ins, "kind": kind, "platform": platform,
                                                 "date": date, "title": title, "source": source})

def crawl(ins):
    home = ins["url"]; host = urlparse(home).netloc.replace("www.", "")
    html = get(home)
    if not html: return ins, [], {}, "unreachable or blocked"
    soup = BeautifulSoup(html, "html.parser")
    links, soc = [], {}
    for a in soup.find_all("a", href=True):
        u = urldefrag(urljoin(home, a["href"]))[0]; d = urlparse(u).netloc.replace("www.", "")
        for k, dom in SOCIAL.items():
            if dom in d or (k == "x" and d == "x.com"): soc.setdefault(k, u)
        if d != host or SKIP.search(u) or u.rstrip("/") == home.rstrip("/"): continue
        if (KEY.search(u) or KEY.search(a.get_text(" "))) and u not in links: links.append(u)
    docs = []
    for u in [home] + links[: MAX_PAGES - 1]:
        page = html if u == home else get(u)
        if not page: continue
        s = BeautifulSoup(page, "html.parser"); title = s.title.get_text(strip=True) if s.title else ""
        txt = text_of(s)
        if len(txt) > 200:
            docs.append(D(ins["name"], txt, u, "news" if NEWS.search(u) else "web", "website", "", title))
        time.sleep(0.5)
    return ins, docs, soc, f"{len(docs)} pages" if docs else "no readable text (JS site?) - add manual/social file"

# ---- official social routes -------------------------------------------------
def fb_posts(name, page, token):
    try:
        r = requests.get(f"https://graph.facebook.com/v21.0/{page}/posts", timeout=20, params={
            "fields": "message,created_time,permalink_url", "limit": 25, "access_token": token}).json()
        return [D(name, p["message"], p.get("permalink_url", ""), "social", "facebook", p.get("created_time", "")[:10])
                for p in r.get("data", []) if p.get("message")]
    except Exception:
        return []

def yt_videos(name, channel, key):
    try:
        r = requests.get("https://www.googleapis.com/youtube/v3/search", timeout=20, params={
            "part": "snippet", "channelId": channel, "order": "date", "maxResults": 25, "type": "video", "key": key}).json()
        return [D(name, f"{i['snippet']['title']}. {i['snippet']['description']}",
                  f"https://www.youtube.com/watch?v={i['id']['videoId']}", "social", "youtube", i["snippet"]["publishedAt"][:10])
                for i in r.get("items", [])]
    except Exception:
        return []

def csv_posts():
    """data/social/*.csv with columns: insurer,platform,date,url,text (posts you copied from public pages)."""
    out = []
    for f in pathlib.Path("data/social").glob("*.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            if r.get("text"):
                out.append(D(r.get("insurer", "?"), r["text"], r.get("url", ""), "social", r.get("platform", "social"), r.get("date", "")))
    return out

def build_index(progress=None):
    cfg = yaml.safe_load(open("insurers.yaml"))["insurers"]
    docs, status, links, done = [], [], [], 0
    fb, yt = os.getenv("FB_ACCESS_TOKEN"), os.getenv("YOUTUBE_API_KEY")
    with ThreadPoolExecutor(max_workers=6) as ex:
        for ins, d, soc, msg in ex.map(crawl, cfg):
            docs += d
            if fb and ins.get("facebook_page"): docs += fb_posts(ins["name"], ins["facebook_page"], fb)
            if yt and ins.get("youtube_channel"): docs += yt_videos(ins["name"], ins["youtube_channel"], yt)
            status.append({"insurer": ins["name"], "result": msg})
            links.append({"insurer": ins["name"], **soc})
            done += 1
            if progress: progress(done / len(cfg), f"{ins['name']}: {msg}")
    docs += csv_posts()
    for f in pathlib.Path("data/manual").glob("*__*.txt"):
        name, cat = f.stem.split("__", 1)
        docs.append(D(name, f.read_text(encoding="utf-8"), f"manual:{f.name}", "web", "manual", "", cat))
    if not docs: raise RuntimeError("No content could be loaded.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    FAISS.from_documents(chunks, embeddings()).save_local("index")
    json.dump(status, open("index/status.json", "w")); json.dump(links, open("index/social_links.json", "w"))
    return status

if __name__ == "__main__":
    build_index(lambda p, m: print(f"{p:.0%} {m}"))
