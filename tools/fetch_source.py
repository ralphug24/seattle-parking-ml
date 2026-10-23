"""Download a bounded portion of the original public Kaggle source, version 1."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASET = "arundhativt/seattle-parking-dataset"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=55)
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    name = f"Parking_Holiday_Weather_Seattle_2015/PaidParkingOccupancyWeatherData_2015_{args.part}.csv"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/{urllib.parse.quote(name, safe='')}?datasetVersionNumber=1"
    target = raw / Path(name).name
    download = target.with_suffix(".download")
    if not target.exists():
        print("Downloading source part", args.part, flush=True)
        with urllib.request.urlopen(url, timeout=90) as response, download.open("wb") as out:
            print("Content type:", response.headers.get("Content-Type"), flush=True)
            while block := response.read(1024 * 1024):
                out.write(block)
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as archive:
                csvs = [n for n in archive.namelist() if n.endswith(".csv")]
                if len(csvs) != 1:
                    raise ValueError(f"Expected one source CSV, got {csvs}")
                with archive.open(csvs[0]) as source, target.open("wb") as out:
                    while block := source.read(1024 * 1024):
                        out.write(block)
            download.unlink()
        else:
            download.rename(target)
    manifest = {"dataset": DATASET, "version": 1, "file": name, "download_url": url,
                "bytes": target.stat().st_size, "sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest(),
                "scope": "One source shard; not the complete original dataset or original experiment sample."}
    dest = ROOT / "reports" / f"source-part-{args.part}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
