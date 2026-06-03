#!/usr/bin/env python3
"""
charon_watch.py — the ferryman keeps watch.

Polls a repo for NEW open PRs since the last high-water mark and folds them into
a review queue. Incremental by construction: each poll processes only the delta
and converges to a fixed point — when no soul is newer than the waterline, there
is nothing to do. Same Banach idea as the rest of the fold family, applied to an
ever-growing PR stream, so a busy repo costs one API page per poll once caught up.

    python charon_watch.py --repo owner/name --once     # one incremental poll
    python charon_watch.py --repo owner/name --watch     # poll forever every --interval s

State lives beside the data: high-water PR number + an append-only queue.jsonl.
First run sets the waterline to the newest PR and skips the backlog (the batch
triage in charon.py already covers what is already on the shore). 100% local.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from charon import API, _get, _to_soul, github_token

DATA = Path(__file__).resolve().parent.parent / "data"
HIGH = DATA / "high_water.txt"
QUEUE = DATA / "queue.jsonl"


def _read_high() -> int:
    try:
        return int(HIGH.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _newest_number(repo: str, token: str | None) -> int:
    url = f"{API}/repos/{repo}/pulls?state=open&sort=created&direction=desc&per_page=1"
    batch = json.loads(_get(url, token))
    return batch[0]["number"] if batch else 0


def _fetch_newer(repo: str, token: str | None, since: int,
                 page: int = 1, acc: list | None = None) -> list:
    """Recursively gather open PRs with number > `since`, newest-first.

    Converges to a fixed point: stops the moment a page contains a soul at or
    below the waterline (we have caught up) or the river runs dry. Only the
    delta is ever materialised — the recursion IS the speed-up.
    """
    acc = acc if acc is not None else []
    url = (f"{API}/repos/{repo}/pulls?state=open&sort=created&direction=desc"
           f"&per_page=100&page={page}")
    batch = json.loads(_get(url, token))
    if not batch:
        return acc
    fresh = [p for p in batch if p["number"] > since]
    acc.extend(fresh)
    if len(fresh) < len(batch):          # crossed the waterline -> fixed point
        return acc
    return _fetch_newer(repo, token, since, page + 1, acc)


def poll(repo: str, token: str | None) -> int:
    """One incremental poll. Returns the count of newly-queued souls."""
    DATA.mkdir(exist_ok=True)
    since = _read_high()
    if since == 0:                       # first run: set the waterline, skip the backlog
        HIGH.write_text(str(_newest_number(repo, token)))
        return 0
    fresh = _fetch_newer(repo, token, since)
    if not fresh:
        return 0
    with QUEUE.open("a") as q:
        q.writelines(json.dumps(asdict(_to_soul(p))) + "\n" for p in fresh)
    HIGH.write_text(str(max(p["number"] for p in fresh)))
    return len(fresh)


def _watch(repo: str, token: str | None, interval: int) -> None:
    while True:
        try:
            n = poll(repo, token)
            if n:
                print(f"[charon-watch] +{n} new souls (waterline {_read_high()})", flush=True)
        except Exception as e:                       # never let the watch die on a blip
            print(f"[charon-watch] poll error: {e}", flush=True)
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    token = github_token()  # env GH_TOKEN/GITHUB_TOKEN, else the gh CLI's token
    if args.watch:
        _watch(args.repo, token, args.interval)
        return 0
    n = poll(args.repo, token)
    print(f"queued {n} new souls (waterline {_read_high()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
