#!/usr/bin/env python3
"""Resume this task's verified-range Point4D checkpoint download and check LFS SHA."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "checkpoints/point4d/point4d_final.pt"
LOGS = ROOT / "experiments/mosca_baseline_20260922/point4d_setup"
URL = "https://huggingface.co/minsikj/point4d/resolve/1ad5ce9a54be2d69697ce5d0f08b05b5f4fdb9b9/point4d_final.pt"
TOTAL = 5797376562
EXPECTED = "dacdabdd8678da9f247f6ac51aefcd1771a94af4b677166a5683b6971c4950cb"


def fetch(path):
    start, end = map(int, path.stem.split("-"))
    attempts = 0
    while path.stat().st_size < end - start + 1:
        offset = start + path.stat().st_size
        stop = min(end, offset + 64 * 1024 * 1024 - 1)
        request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-{stop}"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                expected_range = f"bytes {offset}-{stop}/{TOTAL}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError("Unexpected Content-Range; refuse corrupting partial file")
                with path.open("ab") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            if start + path.stat().st_size == offset:
                raise RuntimeError("No bytes received")
            print(path.name, path.stat().st_size, flush=True)
            attempts = 0
        except Exception as error:
            attempts += 1
            print("Retry", path.name, type(error).__name__, flush=True)
            if attempts > 8:
                raise
            time.sleep(min(3 * attempts, 15))
    if path.stat().st_size != end - start + 1:
        raise RuntimeError("Range size mismatch")
    return path


if __name__ == "__main__":
    start_time = time.time()
    parts = sorted((TARGET.parent / "download_parts").glob("*.part"), key=lambda path: int(path.stem.split("-")[0]))
    if not parts:
        raise RuntimeError("This recovery script expects the existing bounded download parts")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, parts))
    prefix_size = TARGET.stat().st_size
    first = int(parts[0].stem.split("-")[0])
    if prefix_size != first:
        raise RuntimeError("Existing sequential prefix differs from expected; refuse duplicate append")
    with TARGET.open("ab") as output:
        for path in parts:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    output.write(block)
    digest = hashlib.sha256()
    with TARGET.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    result = {"url": URL, "path": str(TARGET), "expected_size": TOTAL, "size": TARGET.stat().st_size,
              "expected_sha256": EXPECTED, "sha256": digest.hexdigest(),
              "verified": TARGET.stat().st_size == TOTAL and digest.hexdigest() == EXPECTED,
              "range_resume_prefix_bytes": prefix_size, "resume_seconds": time.time() - start_time}
    manifest = LOGS / "weight_manifest.json"
    if manifest.exists():
        manifest.rename(LOGS / "weight_download_sequential_partial.json")
    manifest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not result["verified"]:
        raise RuntimeError("Checkpoint SHA mismatch")
    for path in parts:
        path.unlink()
