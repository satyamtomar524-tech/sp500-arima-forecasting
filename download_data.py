"""Optional fresh Yahoo historical download; no credentials or login automation.

The endpoint is publicly reachable but has no verified stable API contract.
Do not redistribute downloaded data without resolving provider rights.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=946684800&period2=1609459200&interval=1d"


def main():
    folder = ROOT / "local_data"
    folder.mkdir(exist_ok=True)
    target = folder / "yahoo_gspc_2000_2020_close.csv"
    if target.exists():
        raise SystemExit("Local data already exist. Preserve them; move to a dated backup before fetching a new vintage.")
    request = Request(URL, headers={"User-Agent": "Academic historical time-series reconstruction"})
    with urlopen(request, timeout=40) as response:
        payload = json.load(response)
    if payload["chart"].get("error"):
        raise RuntimeError(payload["chart"]["error"])
    chart = payload["chart"]["result"][0]
    rows = []
    for timestamp, close in zip(chart["timestamp"], chart["indicators"]["quote"][0]["close"]):
        if close is None:
            raise ValueError("Missing close: investigate source; no interpolation is permitted")
        date = datetime.fromtimestamp(timestamp, ZoneInfo("America/New_York")).date().isoformat()
        if "2000-01-01" <= date < "2021-01-01":
            rows.append((date, close))
    assert rows and len({date for date, _ in rows}) == len(rows)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Date", "Close"])
        writer.writerows(rows)
    metadata = {
        "status": "NEW_RECONSTRUCTION_EXTRACT_NOT_ORIGINAL_COLLEGE_DATA",
        "provider": "Yahoo Finance", "symbol": "^GSPC", "download_url": URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "row_count": len(rows),
        "target": "Unadjusted Close; S&P 500 price index points, excluding dividend reinvestment",
        "licensing_status": "Open redistribution rights not verified",
        "csv_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (folder / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} observed daily closes to {target}")


if __name__ == "__main__":
    main()
