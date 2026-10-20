#!/usr/bin/env python3
"""
download_models.py: fetch the trained models from the GitHub release and
check every file against its published SHA-256. Standard library only.

    python scripts/download_models.py                 # version 2.0.0 -> models/
    python scripts/download_models.py --sha256 <hex>  # also check the zip itself
"""
import argparse, hashlib, os, sys, tempfile, urllib.error, urllib.request, zipfile

REPO = "https://github.com/bioinfbrad/burn-ml"
# SHA-256 of each published model bundle, checked automatically on download.
KNOWN_SHA256 = {
    "2.0.0": "86c458507d9ae1069440f02d92d2ae41fd3003be28777e6538bc14eac5900cf9",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="2.0.0")
    ap.add_argument("--url", default=None, help="override the download URL (for example a Zenodo file link)")
    ap.add_argument("--repo", default=REPO, help=argparse.SUPPRESS)
    ap.add_argument("--sha256", default=None, help="expected SHA-256 of the zip (from the release notes)")
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    a = ap.parse_args()
    name = f"burn-ml-models-v{a.version}.zip"
    urls = [a.url] if a.url else [f"{a.repo}/releases/download/{tag}/{name}" for tag in (a.version, f"v{a.version}")]
    tmp = os.path.join(tempfile.mkdtemp(), "models.zip")
    for url in urls:
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, tmp)
            break
        except urllib.error.HTTPError as e:
            print(f"  not found ({e.code})")
    else:
        sys.exit(f"Could not download {name}. See {REPO}/releases for available versions.")
    expected = (a.sha256 or KNOWN_SHA256.get(a.version, "")).lower()
    digest = sha256(tmp)
    if expected and digest != expected:
        sys.exit(f"SHA-256 mismatch: expected {expected}, got {digest}. Not extracted.")
    print(f"SHA-256 {digest}" + (" (verified)" if expected else " (no published checksum to compare)"))
    with zipfile.ZipFile(tmp) as z:
        z.extractall(a.dest)
    dest = os.path.abspath(a.dest); bad = 0
    for line in open(os.path.join(dest, "models", "SHA256SUMS")):
        h, rel = line.strip().split("  ", 1)
        if sha256(os.path.join(dest, rel)) != h:
            print(f"  CORRUPT: {rel}"); bad += 1
    if bad:
        sys.exit(f"{bad} file(s) failed verification")
    print(f"Models extracted to {os.path.join(dest, 'models')} and verified.")


if __name__ == "__main__":
    main()
