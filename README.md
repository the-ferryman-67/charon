# Charon — the ferryman of the Styx

> A local-first triage gatekeeper for an open-source repo drowning in pull requests.
> He finds the souls that are really the same soul, weighs each offering, learns your
> taste from your own history, and tells you who's worth ferrying first.
> Runs on your machine. Costs nothing. Nothing leaves the box.

*Charon does not judge the dead. He just checks if they brought the coin.*

---

## What he does

When PRs arrive faster than one human can read them, the hard question isn't "is this PR good?" — it's **"is this PR signal, or one more wave of the same thing?"** Charon answers that, in four moves:

**1. He finds the kindred souls.** Many PRs are really the same offering — the same fix, the same deprecation, the same idea. Charon clusters them by weighing each PR two ways:
- **TF-IDF** — by its *words* (catches near-identical phrasings).
- **CF-IOF** *(Concept-Frequency × Inverse-Outlet-Frequency)* — by its *concepts across the contributors who raised them*. A concept every author submits is boilerplate (low weight); one concentrated in a few is distinctive. This is what separates a coordinated tide from genuine work — it caught a 28-strong `datetime.utcnow()` wave that pure word-matching scattered.

**2. He weighs the offering.** Each PR pays a toll in *obols* (0–10) from honest, cheap signals: does it bring tests, a real description, a named kind of change, does it cite the issues it touches. Then a verdict:
- **⛴ ABOARD** — genuine cargo, worth your eyes.
- **🪙 OBOL-NEEDED** — good intent, missing the coin (needs a test, a description, a little more).
- **⏳ SHORE** — a kindred soul to one already at the oar; it can wait.

**3. He learns your taste.** Charon reads your *closed* PRs — every one you merged, every one you turned back — and trains two classifiers (logistic regression + complement naive Bayes) to predict what *you* accept. Then he scores every open PR with that learned likelihood and sorts your queue by it. He reports his own cross-validated AUC, because a model that won't tell you how sure it is, isn't one to trust. (~0.70 on a real repo: a meaningful, modest signal — never a promise.)

**4. He keeps an honest ledger.** Every verdict can be signed with a local key (ED25519) — replayable, tamper-evident, and it never touches a server. Provenance without surveillance.

## Run it yourself

```bash
# triage the open PRs of any repo, locally
python charon.py --repo owner/name

# learn from the repo's own merge history, add a merge-likelihood to each PR
python charon_learn.py --repo owner/name --manifest manifest.json

# render a shareable proof page from the result
python render_proof.py manifest.json --repo owner/name --out proof.html
```

No Docker. No config file. No account. The only dependency is `scikit-learn`
(embeddings, clustering, classifiers — all local). If you'd rather weigh souls
with a local LLM council, point it at your Ollama — Charon refuses to phone home.

## Why local-first

This isn't a feature, it's the whole point. Triaging your contributors' work means
reading their words. That should happen on *your* machine, under *your* key, and stop
there. Charon is built to be the kind of maintainer's helper you'd actually self-host —
the same bargain the rest of your workspace already makes.

## Honest about what it is

It's a **draft triage**, not a verdict. The clustering is good but not perfect; the
learned score is a real signal, not an oracle; the proof page ran on a public snapshot,
not your live repo. Add the script to your repo and it runs live, on your box. Nothing
here is binding — Charon just makes the river smaller so one person can cross it.

---

*Odysseus sailed to the underworld in Book 11 to find his way home. Every hero's journey
needs a ferryman. 🛶*
