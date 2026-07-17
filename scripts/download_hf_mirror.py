"""Download a gated Hugging Face snapshot through a mirror with size checks.

Some mirrors reject the metadata HEAD request used by ``hf download`` for gated
repositories while accepting authenticated GET requests with ``?download=true``.
This downloader uses the Hub API only for the file manifest, then downloads each
file with resumable authenticated GET requests and verifies its declared size.
"""

import argparse
import concurrent.futures
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, get_token


def download_file(endpoint, repo_id, revision, token, target_dir, sibling, retries):
    target = target_dir / sibling.rfilename
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = sibling.size
    if target.is_file() and (expected is None or target.stat().st_size == expected):
        return f"cached {sibling.rfilename}"

    partial = target.with_name(target.name + ".part")
    url = (
        f"{endpoint.rstrip('/')}/{repo_id}/resolve/{quote(revision, safe='')}/"
        f"{quote(sibling.rfilename, safe='/')}?download=true"
    )
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=(30, 300),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
            actual = partial.stat().st_size
            if expected is not None and actual != expected:
                raise OSError(
                    f"size mismatch for {sibling.rfilename}: {actual} != {expected}"
                )
            partial.replace(target)
            return f"downloaded {sibling.rfilename} ({actual} bytes)"
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"failed {sibling.rfilename} after {retries} attempts"
                ) from exc
            time.sleep(min(5 * attempt, 30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise SystemExit("Hugging Face login required")
    api = HfApi(endpoint=args.endpoint, token=token)
    info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    revision = info.sha or args.revision
    siblings = info.siblings or []
    args.target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(siblings)} files at revision {revision}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                download_file,
                args.endpoint,
                args.repo_id,
                revision,
                token,
                args.target_dir,
                sibling,
                args.retries,
            )
            for sibling in siblings
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)

    print(f"Snapshot complete: {args.target_dir}", flush=True)


if __name__ == "__main__":
    main()
