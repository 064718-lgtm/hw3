"""Minimal data ingestion helper to download the CSV dataset locally."""
from pathlib import Path
import urllib.request


def download(url: str, out: str = "data/sms_spam_no_header.csv") -> str:
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return str(dest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=False, help="Dataset URL")
    parser.add_argument("--out", default="data/sms_spam_no_header.csv")
    args = parser.parse_args()
    url = args.url or "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    print(download(url, args.out))
