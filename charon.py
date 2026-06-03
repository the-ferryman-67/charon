#!/usr/bin/env python3
"""
charon.py — the ferryman of the Styx.

A local-first triage gatekeeper for an open-source repository drowning in
pull requests. Charon fetches the open PRs, finds the souls that are really
the *same* soul (semantic clustering), weighs each offering, and renders a
verdict: ABOARD, OBOL-NEEDED, or SHORE.

He runs entirely on your own machine. No cloud. No tracking. No data leaves
your box. Your repository stays yours.

    python charon.py --repo owner/name                  # triage open PRs
    python charon.py --repo owner/name --json out.json  # machine-readable
    python charon.py --dry-run

Charon does not judge the dead. He just checks if they brought the coin.

Auth is optional: public repos work unauthenticated (60 req/hr is plenty for a
few hundred PRs). To lift the rate limit, set one of GH_AUTH_ENV_NAMES below.
The value is used only in a read-request header. It is never logged or stored.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass

API = "https://api.github.com"

# Environment variables Charon will read a GitHub token from, in order.
# The names are listed here in the open; the value is only ever placed in an
# Authorization header on a read request, never printed or written to disk.
GH_AUTH_ENV_NAMES = ("GH_TOKEN", "GITHUB_TOKEN")


def first_env_value(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among the named environment variables."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def github_token() -> str | None:
    """A GitHub token for read requests: env vars first, else the local gh CLI's
    stored token. The value stays inside this process — never logged, never set as
    an environment assignment — turning a 60/hr unauthenticated cap into 5000/hr."""
    tok = first_env_value(GH_AUTH_ENV_NAMES)
    if tok:
        return tok
    try:
        import subprocess
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


# ── the offering: one pull request, weighed at the bank of the Styx ──────────


@dataclass
class Soul:
    number: int
    title: str
    body: str
    user: str
    draft: bool
    url: str
    created_at: str
    cluster: int = -1
    obols: float = 5.0          # the coin, 0-10
    verdict: str = "OBOL-NEEDED"
    is_keeper: bool = False     # best soul in its cluster
    reason: str = ""

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


# ── fetch the souls waiting on the shore ─────────────────────────────────────


def _get(url: str, token: str | None) -> bytes:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "charon-ferryman"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < 3:  # rate limit, breathe
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("the river would not answer")


def _to_soul(p: dict) -> Soul:
    return Soul(
        number=p["number"],
        title=p.get("title") or "",
        body=(p.get("body") or "")[:2000],
        user=(p.get("user") or {}).get("login", "?"),
        draft=bool(p.get("draft")),
        url=p.get("html_url", ""),
        created_at=p.get("created_at", ""),
    )


def fetch_open_prs(repo: str, token: str | None, limit: int | None = None) -> list[Soul]:
    souls: list[Soul] = []
    page = 1
    while page <= 30:  # safety bound
        url = f"{API}/repos/{repo}/pulls?state=open&per_page=100&page={page}"
        batch = json.loads(_get(url, token))
        if not batch:
            break
        souls.extend(_to_soul(p) for p in batch)
        page += 1
        if limit and len(souls) >= limit:
            return souls[:limit]
    return souls


# ── recognise the souls who are really one soul ──────────────────────────────


def _tfidf(texts: list[str]):
    """Term-Frequency x Inverse-DOCUMENT-Frequency. Rows L2-normalised."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                        max_features=4096, sublinear_tf=True)
    return v.fit_transform(texts), v


def _cfiof(texts: list[str], authors: list[str], vectorizer):
    """Concept-Frequency x Inverse-OUTLET-Frequency. Rows L2-normalised.

    Concepts = the TF-IDF vocabulary. 'Outlet' = the contributor (PR author).
    A concept used by many different authors is generic boilerplate (low weight);
    one concentrated in few authors is distinctive (high weight). Complements
    TF-IDF, which weights by document rarity, not author rarity — so it separates
    a tide of look-alike PRs from genuinely distinct work. 100% local.
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    cv = CountVectorizer(vocabulary=vectorizer.vocabulary_, ngram_range=(1, 2))
    counts = cv.transform(texts).astype(float)               # CF: concept counts / doc
    uniq = sorted(set(authors))
    aid = {a: i for i, a in enumerate(uniq)}
    rows = np.array([aid[a] for a in authors])
    sel = sp.csr_matrix((np.ones(len(authors)), (rows, np.arange(len(authors)))),
                        shape=(len(uniq), len(authors)))     # author <- doc selector
    author_has = (sel @ (counts > 0).astype(int)) > 0        # authors x concepts
    author_df = np.asarray(author_has.sum(axis=0)).ravel()   # distinct authors / concept
    iof = np.log((1.0 + len(uniq)) / (1.0 + author_df)).reshape(1, -1)
    counts.data = 1.0 + np.log(counts.data)                  # sublinear CF
    return normalize(counts.multiply(iof).tocsr(), norm="l2", axis=1)


def _vectorise(souls: list[Soul], method: str = "blend"):
    """Return (dense L2-normalised vectors, default_threshold, method_label).

    100% local — no network, no data leaves the box. Methods:
      tfidf  - lexical term rarity (catches near-identical phrasings)
      cfiof  - concept x inverse-OUTLET(author) rarity (catches boilerplate tides)
      blend  - both, 50/50 (default)
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize
    texts = [s.text for s in souls]
    authors = [s.user for s in souls]
    tfidf, vec = _tfidf(texts)
    if method == "tfidf":
        return np.asarray(tfidf.todense(), dtype="float32"), 0.42, "tf-idf (lexical)"
    cfiof = _cfiof(texts, authors, vec)
    if method == "cfiof":
        return np.asarray(cfiof.todense(), dtype="float32"), 0.40, "cf-iof (concept x outlet)"
    combined = normalize(sp.hstack([tfidf, cfiof]).tocsr(), norm="l2", axis=1)
    return np.asarray(combined.todense(), dtype="float32"), 0.34, "blend (tf-idf x cf-iof)"


def cluster_souls(souls: list[Soul], threshold: float | None = None, method: str = "blend"):
    """Union-find on cosine similarity. Returns (n_clusters, method_label, threshold)."""
    import numpy as np
    if not souls:
        return 0, "none", 0.0
    vecs, default_thr, method_label = _vectorise(souls, method)
    thr = default_thr if threshold is None else threshold
    sim = vecs @ vecs.T  # cosine (vectors are normalised)
    n = len(souls)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # above-threshold pairs extracted vectorially (no nested Python scan),
    # then a single union pass over just those edges.
    edges = np.argwhere(np.triu(sim >= thr, k=1))
    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, len(roots))
        souls[i].cluster = roots[r]
    return len(roots), method_label, thr


# ── weigh each offering (mechanical floor — deterministic, zero LLM) ─────────

_TEST = re.compile(r"\b(test|spec|coverage|error[- ]?path|regression)\b", re.I)
_TYPED = re.compile(r"^(fix|feat|test|docs|perf|refactor|chore|build|ci)(\(|:|!)", re.I)
_REF = re.compile(r"#\d+")
# the "defensive validation swarm" shape — reject/skip/ignore/normalise invalid X
_SWARM = re.compile(r"^\s*(reject|skip|ignore|normali[sz]e|validate|handle|save only|require)\b", re.I)


def _score_soul(s: Soul, cluster_size: int) -> None:
    """Assign the obol toll (0-10) from cheap, honest signals."""
    obols = 5.0
    why: list[str] = []
    blob = s.text
    if _TEST.search(blob):
        obols += 2; why.append("brings tests")
    if len(s.body) > 200:
        obols += 1; why.append("a real account, not a wish")
    if _TYPED.search(s.title):
        obols += 1; why.append("named its kind")
    if _REF.search(blob):
        obols += 1; why.append("knows the others (cites #)")
    if s.draft:
        obols -= 2; why.append("still a draft")
    if cluster_size >= 4 and _SWARM.search(s.title):
        obols -= 1; why.append(f"one of a {cluster_size}-strong tide of the same shape")
    s.obols = max(0.0, min(10.0, obols))
    s.reason = "; ".join(why) or "a plain offering"
    if round(s.obols) in (6, 7):                  # the uncertain zone 🤷
        s.reason += " — eh, a 6... 6-7"


def _verdict_for(s: Soul, keeper: Soul, cluster_size: int) -> None:
    if cluster_size >= 2 and not s.is_keeper:
        s.verdict = "SHORE"
        s.reason = f"the same soul as #{keeper.number}, already at the oar"
    elif s.obols >= 7:
        s.verdict = "ABOARD"
    elif s.obols >= 4:
        s.verdict = "OBOL-NEEDED"
    else:
        s.verdict = "SHORE"


def _crown_and_judge(members: list[Soul]) -> None:
    """Crown the cluster's keeper, then render every member's verdict."""
    keeper = sorted(members, key=lambda s: (-s.obols, s.number))[0]
    keeper.is_keeper = True
    for s in members:
        _verdict_for(s, keeper, len(members))


def weigh(souls: list[Soul]) -> None:
    by_cluster: dict[int, list[Soul]] = {}
    for s in souls:
        by_cluster.setdefault(s.cluster, []).append(s)
    for s in souls:
        _score_soul(s, len(by_cluster[s.cluster]))
    for members in by_cluster.values():
        _crown_and_judge(members)


# ── the ferry manifest ───────────────────────────────────────────────────────


def _label(titles: list[str]) -> str:
    stop = set("fix feat test docs the a an to of in on for and or non string add "
               "crashes crash when with into not invalid update remove handle".split())
    toks = itertools.chain.from_iterable(re.findall(r"[a-zA-Z]{4,}", t) for t in titles)
    words = Counter(w.lower() for w in toks if w.lower() not in stop)
    top = [w for w, _ in words.most_common(3)]
    return " / ".join(top) if top else "kindred souls"


def _crowd(members: list[Soul]) -> dict:
    keeper = next(s for s in members if s.is_keeper)
    return {
        "label": _label([s.title for s in members]),
        "size": len(members),
        "keeper": keeper.number,
        "members": sorted(s.number for s in members),
    }


def manifest(souls: list[Soul], n_clusters: int) -> dict:
    by_cluster: dict[int, list[Soul]] = {}
    for s in souls:
        by_cluster.setdefault(s.cluster, []).append(s)
    crowds = [_crowd(ms) for ms in by_cluster.values() if len(ms) >= 2]
    crowds.sort(key=lambda c: -c["size"])
    counts = Counter(s.verdict for s in souls)
    return {
        "souls": len(souls),
        "clusters": n_clusters,
        "duplicate_crowds": len(crowds),
        "souls_in_crowds": sum(c["size"] for c in crowds),
        "verdicts": dict(counts),
        "crowds": crowds,
        "ledger": [
            {"pr": s.number, "verdict": s.verdict, "obols": round(s.obols, 1),
             "keeper": s.is_keeper, "title": s.title, "url": s.url, "reason": s.reason}
            for s in sorted(souls, key=lambda s: s.number, reverse=True)
        ],
    }


STYX = "═══════════════  S T Y X  ═══════════════"


def main() -> int:
    ap = argparse.ArgumentParser(description="Charon — the ferryman of the Styx")
    ap.add_argument("--repo", help="owner/name")
    ap.add_argument("--limit", type=int, default=None, help="cap souls (for a quick look)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="cosine cutoff; omitted = method default")
    ap.add_argument("--method", choices=["blend", "tfidf", "cfiof"], default="blend",
                    help="kinship signal (default: blend = tf-idf x cf-iof)")
    ap.add_argument("--json", help="write the manifest here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run or not args.repo:
        print(f"\n{STYX}\n      the ferryman waits. name a repo to begin.\n{STYX}\n")
        return 0

    token = first_env_value(GH_AUTH_ENV_NAMES)
    print(f"{STYX}", file=sys.stderr)
    print(f"Charon rows out to {args.repo} …", file=sys.stderr)
    souls = fetch_open_prs(args.repo, token, args.limit)
    print(f"{len(souls)} souls on the shore. weighing offerings …", file=sys.stderr)
    if not souls:
        print("the shore is empty. nothing to ferry.", file=sys.stderr)
        return 0
    n, method_label, thr = cluster_souls(souls, args.threshold, args.method)
    weigh(souls)
    m = manifest(souls, n)
    print(f"weighed by {method_label} kinship (cosine >= {thr:.2f})", file=sys.stderr)

    v = m["verdicts"]
    print(f"\n{STYX}")
    print(f"  {m['souls']} souls · {m['clusters']} truly-distinct · "
          f"{m['duplicate_crowds']} crowds carrying {m['souls_in_crowds']} kindred souls")
    print(f"  ⛴  ABOARD {v.get('ABOARD', 0)}   🪙 OBOL-NEEDED {v.get('OBOL-NEEDED', 0)}   "
          f"⏳ SHORE {v.get('SHORE', 0)}")
    print(STYX)
    print("\n  The largest crowds at the shore (the same soul, many times over):\n")
    for c in m["crowds"][:8]:
        print(f"   ◦ {c['size']:>3}× — {c['label']}   (keeper: #{c['keeper']})")
    print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(m, f, indent=2)
        print(f"manifest written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
