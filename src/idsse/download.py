"""Download or repair the IDSSE files from figshare (CC BY 4.0).

    from src.idsse import download
    download.ensure()                  # fetch every file that is missing, truncated or corrupt

Source: figshare article 28196177 (doi:10.6084/m9.figshare.28196177), Bassek et al., Scientific
Data (2025). The figshare API lists each file with its size and MD5. A local file is kept only when
its size matches (and its MD5, with verify_md5=True); anything else is downloaded again to a `.part`
file, checked against the MD5 and only then moved into place. The download links redirect to S3 URLs
that expire after 10 s, so each file is requested fresh and streamed straight away.
"""

import hashlib
import json
import os
import time
import urllib.request

from . import parse

ARTICLE_ID = 28196177
API = "https://api.figshare.com/v2/articles/{}/files?page_size=100".format(ARTICLE_ID)
CHUNK = 1 << 20


def target_dir():
    """$IDSSE_DIR, else the existing external-data/idsse, else external-data/idsse in the repo (gitignored)."""
    try:
        return parse.find_data_dir()
    except FileNotFoundError:
        return os.path.join(parse.REPO_ROOT, "external-data", "idsse")


def remote_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        return json.load(r)


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url, dest, size, md5, retries=3):
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            h, done, t0 = hashlib.md5(), 0, time.time()
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as fh:
                for block in iter(lambda: r.read(CHUNK), b""):
                    fh.write(block)
                    h.update(block)
                    done += len(block)
            if done != size or h.hexdigest() != md5:
                raise IOError("size {} / md5 {} do not match figshare".format(done, h.hexdigest()))
            os.replace(part, dest)
            return done / max(time.time() - t0, 1e-6) / 1e6
        except Exception as e:  # noqa: BLE001 -- retry any network or checksum failure
            if os.path.exists(part):
                os.remove(part)
            if attempt == retries:
                raise
            print("    retry {} after: {}".format(attempt, e))
            time.sleep(2 * attempt)


def ensure(data_dir=None, matches=None, verify_md5=False, dry_run=False):
    """Make the local IDSSE folder match figshare. Returns one row per file with what was done.

    matches: optional list of match ids (e.g. ["J03WPY"]) to limit the check.
    verify_md5: also hash files whose size already matches (~1 s per 100 MB).
    """
    data_dir = data_dir or target_dir()
    os.makedirs(data_dir, exist_ok=True)
    rows = []
    for f in remote_files():
        name = f["name"]
        if matches and not any(m in name for m in matches):
            continue
        dest = os.path.join(data_dir, name)
        local = os.path.getsize(dest) if os.path.exists(dest) else None
        ok = local == f["size"] and (not verify_md5 or md5_file(dest) == f["computed_md5"])
        row = {"file": name, "remote_mb": round(f["size"] / 1e6, 1),
               "local_mb": None if local is None else round(local / 1e6, 1)}
        if ok:
            row["action"] = "ok"
        elif dry_run:
            row["action"] = "would download"
        else:
            print("downloading {} ({:.0f} MB) ...".format(name, f["size"] / 1e6), flush=True)
            speed = _fetch(f["download_url"], dest, f["size"], f["computed_md5"])
            row["action"] = "downloaded ({:.1f} MB/s)".format(speed)
        rows.append(row)
    return rows
