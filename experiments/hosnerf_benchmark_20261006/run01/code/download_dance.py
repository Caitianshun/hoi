#!/usr/bin/env python3
"""Fetch the public Dance archive to a partial file and verify before rename."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile


FILE_ID = "1vC-CCaieOjSX-OrAqrt40006Ti5TtMvO"
EXPECTED_BYTES = 664214396  # Public Google Drive folder metadata, not a measured result.
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, state):
    state["updated_at_utc"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(state, stream, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path):
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"Archive length {path.stat().st_size} does not match published {EXPECTED_BYTES}")
    with zipfile.ZipFile(path) as z:
        failed = z.testzip()
        if failed is not None:
            raise ValueError(f"CRC failure in ZIP member {failed}")
        numeric_images = [n for n in z.namelist() if n.startswith("Dance/images/")
                          and Path(n).stem.isdigit() and n.endswith(".png")]
        if len(numeric_images) < 16:
            raise ValueError("Downloaded ZIP is not the expected Dance image dataset")
    return {"sha256": sha(path), "zip_crc_verified": True,
            "numeric_image_count": len(numeric_images), "actual_bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    run = Path(__file__).resolve().parents[1]
    p.add_argument("--archive", type=Path, default=run.parents[2] / "other data" / "Dance.zip")
    p.add_argument("--state", type=Path, default=run / "protocol" / "dance_download.json")
    a = p.parse_args()
    archive = a.archive.resolve()
    part = archive.with_name(archive.name + ".part")
    state = {"status": "starting", "source_url": URL, "public_file_id": FILE_ID,
             "published_expected_bytes": EXPECTED_BYTES, "archive": str(archive),
             "partial_archive": str(part), "started_at_utc": now()}
    started = time.monotonic()
    try:
        if archive.exists():
            state.update(verify(archive))
            state["status"] = "existing_verified"
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            offset = part.stat().st_size if part.exists() else 0
            if offset > EXPECTED_BYTES:
                raise ValueError("Existing partial archive is larger than the declared public object")
            state.update(status="downloading", downloaded_bytes=offset)
            save(a.state, state)
            if offset < EXPECTED_BYTES:
                # Cache-busting avoids reuse of the earlier 1 KiB Range probe.
                url = URL + "&request_nonce=" + uuid.uuid4().hex
                headers = {"User-Agent": "Mozilla/5.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        raise RuntimeError("Google Drive returned HTML instead of the public archive")
                    if offset and (response.status != 206 or not
                                   response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError("Server did not honor a safe partial-file resume")
                    state["http_status"] = response.status
                    state["content_type"] = content_type
                    checkpoint = offset
                    with part.open("ab" if offset else "xb") as target:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if offset + len(block) > EXPECTED_BYTES:
                                raise ValueError("Response exceeds declared object length")
                            target.write(block)
                            offset += len(block)
                            if offset - checkpoint >= 32 * 1024 * 1024:
                                target.flush()
                                state["downloaded_bytes"] = offset
                                save(a.state, state)
                                print(json.dumps({"downloaded_bytes": offset, "expected_bytes": EXPECTED_BYTES}), flush=True)
                                checkpoint = offset
                        target.flush()
                        os.fsync(target.fileno())
            state.update(status="verifying", downloaded_bytes=part.stat().st_size)
            save(a.state, state)
            state.update(verify(part))
            if archive.exists():
                raise FileExistsError("Archive destination appeared during download; refusing to replace it")
            # Hardlink publication is atomic and refuses an existing destination.
            os.link(part, archive)
            part.unlink()
            state["status"] = "completed"
        state["wall_seconds"] = time.monotonic() - started
        save(a.state, state)
        print(json.dumps(state), flush=True)
        return 0
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                     downloaded_bytes=part.stat().st_size if part.exists() else 0,
                     wall_seconds=time.monotonic() - started)
        save(a.state, state)
        print(json.dumps(state), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
