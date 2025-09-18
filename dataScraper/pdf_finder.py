#!/usr/bin/env python3
"""
pdf_finder.py

Crawls a site up to TARGET_PAGES pages, finds PDFs, downloads them one-by-one,
and posts each PDF to an ingest endpoint.

Dependencies:
pip install requests beautifulsoup4 tqdm
"""

import os
import time
import csv
import subprocess
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# =========================
# GLOBAL CONFIG
# =========================
BASE_URL = "https://www.ucc.co.ug/"        # <-- change to your target
API_ENDPOINT = "http://localhost:8000/ingest"  # <-- change to your ingest endpoint
DOWNLOAD_DIR = "pdfs"                   # local folder to save PDFs
CRAWL_DEPTH = 2                         # how deep to crawl (0 = only start page)
MAX_PAGES = 1000                        # safety upper bound to avoid runaway (keeps original)
TARGET_PAGES = 5                      # <-- NEW: stop after visiting this many pages
SLEEP_BETWEEN_REQUESTS = 0.2            # polite delay between requests (seconds)
REQUEST_TIMEOUT = 15                    # seconds for GET/HEAD
USER_AGENT = "pdf-finder-bot/1.0"       # change if you want
USE_CURL = False                        # If True, uses curl subprocess to POST file; else uses requests.post
CSV_SUMMARY = "pdf_results.csv"         # CSV summary output
# =========================

# internal state
visited_pages = set()
found_pdfs = {}   # pdf_url -> {"found_on": set(), "download_file": str or "", "download_status": str, "ingest_status": str}
pages_queued = 0

# helpers
def same_domain(url1, url2):
    try:
        return urlparse(url1).netloc == urlparse(url2).netloc
    except:
        return False

def normalize(link, base):
    # join to base, remove fragment, strip trailing space
    try:
        joined = urljoin(base, link.strip())
        clean, _ = urldefrag(joined)
        return clean
    except:
        return None

def is_probably_pdf(url):
    return url.lower().split('?')[0].endswith(".pdf")

def head_is_pdf(url):
    try:
        resp = requests.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        ct = resp.headers.get("Content-Type", "")
        return (resp.status_code == 200) and ("application/pdf" in ct.lower())
    except Exception:
        return False

def unique_filename(folder, filename):
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(folder, candidate)):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate

def download_pdf(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or "unnamed.pdf"
    filename = filename.split("?")[0]
    filename = unique_filename(dest_folder, filename)
    local_path = os.path.join(dest_folder, filename)
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        size = os.path.getsize(local_path)
        return {"path": local_path, "status": "downloaded", "size": size, "http_status": r.status_code}
    except Exception as e:
        return {"path": "", "status": "download_error", "error": str(e)}

def ingest_via_curl(local_path, ingest_url):
    # Using curl to match your example. Returns dict with status and response.
    try:
        cmd = [
            "curl", "-sS", "-X", "POST", ingest_url,
            "-F", f"file=@{local_path};type=application/pdf"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (proc.returncode == 0)
        status_code = None
        return {"status": "ok" if ok else "curl_failed", "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def ingest_via_requests(local_path, ingest_url):
    try:
        with open(local_path, "rb") as fh:
            files = {"file": (os.path.basename(local_path), fh, "application/pdf")}
            resp = requests.post(ingest_url, files=files, timeout=120)
        return {"status": "ok" if resp.ok else "http_error", "http_status": resp.status_code, "response_text": resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def ingest_file(local_path, ingest_url):
    if USE_CURL:
        return ingest_via_curl(local_path, ingest_url)
    else:
        return ingest_via_requests(local_path, ingest_url)

# crawler (BFS)
def crawl_and_process():
    global pages_queued
    parsed_base = urlparse(BASE_URL)
    base_root = f"{parsed_base.scheme}://{parsed_base.netloc}"

    q = deque()
    q.append((BASE_URL, 0))
    pages_visited = 0

    while q and pages_visited < min(MAX_PAGES, TARGET_PAGES):
        url, depth = q.popleft()
        # If we've reached our target pages, break
        if pages_visited >= TARGET_PAGES:
            break

        # Skip if already visited
        if url in visited_pages:
            continue
        if depth > CRAWL_DEPTH:
            continue

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
            status = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
        except Exception as e:
            print(f"[!] Failed to GET {url}: {e}")
            visited_pages.add(url)
            continue

        # Mark as visited (count this page)
        visited_pages.add(url)
        pages_visited += 1
        print(f"[i] Visited ({pages_visited}) {url}  [{status}]")

        # If page itself is a PDF (some sites point directly), register it
        if "application/pdf" in (content_type or "").lower() or is_probably_pdf(url):
            if url not in found_pdfs:
                found_pdfs[url] = {"found_on": set([url])}
            else:
                found_pdfs[url]["found_on"].add(url)
            # don't parse as HTML
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        # Skip non-HTML
        if "html" not in (content_type or "").lower() and "text" not in (content_type or "").lower():
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        # parse links
        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.find_all("a", href=True)
        for a in anchors:
            raw = a["href"].strip()
            if not raw:
                continue
            tgt = normalize(raw, url)
            if not tgt:
                continue
            # ensure same domain
            if not same_domain(base_root, tgt):
                continue

            # If link is clearly a PDF by extension -> record
            if is_probably_pdf(tgt):
                found_pdfs.setdefault(tgt, {"found_on": set()})["found_on"].add(url)
                continue

            # Optional: do HEAD to check content-type for links that don't end with .pdf
            try:
                head = requests.head(tgt, allow_redirects=True, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
                head_ct = head.headers.get("Content-Type", "")
                if head.status_code == 200 and "application/pdf" in (head_ct or "").lower():
                    found_pdfs.setdefault(tgt, {"found_on": set()})["found_on"].add(url)
                    continue
            except Exception:
                # ignore head errors
                pass

            # otherwise enqueue page if not visited and within depth
            if tgt not in visited_pages and depth + 1 <= CRAWL_DEPTH and len(visited_pages) + len(q) < TARGET_PAGES * 2:
                # small heuristic to avoid endlessly expanding queue:
                q.append((tgt, depth + 1))
                pages_queued += 1

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"[i] Crawl finished. Pages visited: {pages_visited}. PDFs found: {len(found_pdfs)}")

    # Download + ingest PDFs one by one
    if not found_pdfs:
        return

    # Process PDFs sequentially (resume-safe)
    for pdf_url in tqdm(list(found_pdfs.keys()), desc="Processing PDFs"):
        meta = found_pdfs[pdf_url]
        # skip if already processed (resume-safe)
        if meta.get("download_status") == "downloaded" and meta.get("ingest_status") == "ok":
            continue

        # Prefer HEAD check before download if not already known
        if not is_probably_pdf(pdf_url) and not head_is_pdf(pdf_url):
            # still attempt download (some sites mislabel) but flag
            pass

        dl = download_pdf(pdf_url, DOWNLOAD_DIR)
        meta["download_file"] = dl.get("path", "")
        meta["download_status"] = dl.get("status", "")
        meta["download_size"] = dl.get("size", 0)
        if dl.get("status") != "downloaded":
            meta["ingest_status"] = "skipped_download_failed"
            continue

        # ingest/upload
        ing = ingest_file(meta["download_file"], API_ENDPOINT)
        meta["ingest_status"] = ing.get("status", "")
        meta["ingest_detail"] = ing

    # write CSV summary
    with open(CSV_SUMMARY, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["pdf_url", "found_on", "download_status", "download_file", "download_size", "ingest_status", "ingest_detail"])
        for pdf, meta in found_pdfs.items():
            writer.writerow([
                pdf,
                ";".join(sorted(meta.get("found_on", []))),
                meta.get("download_status", ""),
                meta.get("download_file", ""),
                meta.get("download_size", ""),
                meta.get("ingest_status", ""),
                str(meta.get("ingest_detail", ""))
            ])

    print(f"[i] Summary saved to {CSV_SUMMARY}")

if __name__ == "__main__":
    # sanity check for curl if USE_CURL
    if USE_CURL:
        from shutil import which
        if which("curl") is None:
            print("[!] WARNING: curl not found in PATH but USE_CURL is True. Either install curl or set USE_CURL = False.")
    print(f"[*] Starting crawl on: {BASE_URL}")
    crawl_and_process()
