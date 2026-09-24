#!/usr/bin/env python3
"""Resumable bounded HTTP ranges, safe against silently truncated large bodies.

Also usable for other authorized public weights. Requires expected byte size
and SHA-256 from the official LFS metadata. Never treats mere HTTP success as
complete. Existing sequential prefix is preserved; disjoint parts are appended
only after every range is complete, then the full file is hashed.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-mib", type=int, default=64)
    args = parser.parse_args()
    args.target = args.target.resolve()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    lock = args.target.with_name(args.target.name + ".download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.target.exists():
        args.target.touch()
    prefix = args.target.stat().st_size
    begin = time.time()
    if prefix > args.size:
        raise RuntimeError("Existing target exceeds expected size; refusing mutation")
    parts_dir = args.target.with_name(args.target.name + ".range_parts")
    parts_dir.mkdir(exist_ok=True)
    manifest = {"url": args.url, "path": str(args.target), "expected_size": args.size,
                "expected_sha256": args.sha256, "resumed_prefix_bytes": prefix}

    def fetch(path):
        start, end = map(int, path.stem.split("-"))
        failures = 0
        while path.stat().st_size < end - start + 1:
            offset = start + path.stat().st_size
            stop = min(end, offset + args.request_mib * 1024 * 1024 - 1)
            try:
                req = urllib.request.Request(args.url, headers={"Range": f"bytes={offset}-{stop}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{stop}/{args.size}":
                        raise RuntimeError("Unexpected Content-Range; no bytes appended")
                    with path.open("ab") as output:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                if start + path.stat().st_size == offset:
                    raise RuntimeError("No progress from range request")
                failures = 0
                print(f"{path.name}: {path.stat().st_size}/{end-start+1}", flush=True)
            except Exception as error:
                failures += 1
                print("Retry", path.name, type(error).__name__, flush=True)
                if failures > 8:
                    raise
                time.sleep(min(3 * failures, 15))
        if path.stat().st_size != end - start + 1:
            raise RuntimeError("Part exceeds expected range; refusing concatenation")

    try:
        if prefix < args.size:
            parts = sorted(parts_dir.glob("*.part"), key=lambda path: int(path.stem.split("-")[0]))
            if not parts:
                step = (args.size - prefix + args.workers - 1) // args.workers
                for index in range(args.workers):
                    start = prefix + index * step
                    end = min(args.size, start + step) - 1
                    if start <= end:
                        path = parts_dir / f"{start}-{end}.part"
                        path.touch()
                        parts.append(path)
            expected_start = prefix
            for path in parts:
                start, end = map(int, path.stem.split("-"))
                if start != expected_start:
                    raise RuntimeError("Parts are not contiguous with existing prefix")
                expected_start = end + 1
            if expected_start != args.size:
                raise RuntimeError("Part ranges do not reach expected EOF")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(fetch, parts))
            with args.target.open("ab") as output:
                for path in parts:
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                            output.write(block)
        manifest.update(size=args.target.stat().st_size, sha256=digest(args.target), elapsed_seconds=time.time()-begin)
        manifest["verified"] = manifest["size"] == args.size and manifest["sha256"] == args.sha256
        args.manifest.write_text(json.dumps(manifest, indent=2))
        if not manifest["verified"]:
            raise RuntimeError("Full file size/SHA mismatch; preserved for diagnosis")
        for path in parts_dir.glob("*.part"):
            path.unlink()
        print(json.dumps(manifest, indent=2), flush=True)
    except Exception as error:
        manifest.update(verified=False, error=str(error), elapsed_seconds=time.time()-begin)
        args.manifest.write_text(json.dumps(manifest, indent=2))
        raise


if __name__ == "__main__":
    main()
