#!/usr/bin/env python3
"""
charon_learn.py — Charon learns the maintainer's own taste.

He studies the repo's OWN merge history: every closed PR is a soul whose fate
is known — merged (it crossed, 1) or closed unmerged (turned back, 0). Two
classifiers learn from it — logistic regression and complement naive Bayes —
then predict a merge-likelihood for each open PR. The ensemble is a learned
confidence laid on top of Charon's mechanical obol weighing.

Honest by construction: it reports its own cross-validated AUC, because a model
that never says how sure it is, isn't one worth trusting.

    python charon_learn.py --repo owner/name --manifest manifest.json

100% local. The history it learns from never leaves your box.
"""
from __future__ import annotations

import argparse
import json

from charon import API, _get, fetch_open_prs, first_env_value, GH_AUTH_ENV_NAMES


def _label_rows(batch: list[dict]) -> list[tuple[str, int]]:
    out = []
    for p in batch:
        text = f"{p.get('title', '')}\n\n{(p.get('body') or '')[:2000]}".strip()
        out.append((text, 1 if p.get("merged_at") else 0))  # merged = crossed
    return out


def fetch_history(repo: str, token: str | None, max_pages: int = 8) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    page = 1
    while page <= max_pages:
        url = f"{API}/repos/{repo}/pulls?state=closed&per_page=100&page={page}"
        batch = json.loads(_get(url, token))
        if not batch:
            break
        rows.extend(_label_rows(batch))
        page += 1
    return rows


def train(texts: list[str], labels: list[int]):
    """Fit logreg + complement-NB; return (vectoriser, lr, nb, honest-stats)."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.model_selection import cross_val_score
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          max_features=8192, sublinear_tf=True)
    X = vec.fit_transform(texts)
    y = np.asarray(labels)
    minority = int(min(y.sum(), len(y) - y.sum()))
    cv = max(2, min(5, minority))
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    nb = ComplementNB()
    auc = {"lr": None, "nb": None}
    if minority >= 2:
        auc["lr"] = round(float(cross_val_score(lr, X, y, cv=cv, scoring="roc_auc").mean()), 3)
        auc["nb"] = round(float(cross_val_score(nb, X, y, cv=cv, scoring="roc_auc").mean()), 3)
    lr.fit(X, y)
    nb.fit(X, y)
    stats = {"n": int(len(y)), "merged": int(y.sum()), "cv_folds": cv,
             "lr_auc": auc["lr"], "nb_auc": auc["nb"]}
    return vec, lr, nb, stats


def ensemble_proba(vec, lr, nb, texts: list[str]):
    """Mean of the two classifiers' P(merge)."""
    X = vec.transform(texts)
    return (lr.predict_proba(X)[:, 1] + nb.predict_proba(X)[:, 1]) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-pages", type=int, default=8)
    args = ap.parse_args()
    token = first_env_value(GH_AUTH_ENV_NAMES)

    print("Charon studies the merge-history of the dead …")
    hist = fetch_history(args.repo, token, args.max_pages)
    texts = [t for t, _ in hist]
    labels = [y for _, y in hist]
    if len(set(labels)) < 2:
        print("not enough closed history to learn from yet.")
        return 0

    vec, lr, nb, stats = train(texts, labels)
    turned = stats["n"] - stats["merged"]
    print(f"learned from {stats['n']} closed souls "
          f"({stats['merged']} crossed, {turned} turned back)")
    auc_line = f"logistic {stats['lr_auc']} · naive-bayes {stats['nb_auc']}"
    print(f"cross-validated AUC ({stats['cv_folds']}-fold) — {auc_line}")

    souls = fetch_open_prs(args.repo, token)
    proba = ensemble_proba(vec, lr, nb, [s.text for s in souls])
    likelihood = {s.number: round(float(pi), 3) for s, pi in zip(souls, proba)}

    with open(args.manifest) as f:
        m = json.load(f)
    for row in m["ledger"]:
        row["p_merge"] = likelihood.get(row["pr"])
    m["model"] = {**stats, "method": "logreg + complement-NB ensemble (local)"}
    with open(args.manifest, "w") as f:
        json.dump(m, f, indent=2)
    print(f"manifest augmented with merge-likelihood for {len(likelihood)} open souls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
