---
title: "Tagging 81 Papers So the Survey Can't Lie: a Tidy-Data Corpus for Structural Entropy"
date: 2026-06-07
description: "How we turned a decade of structural-entropy literature into 1,297 machine-checkable facet tags — and why every number in the survey now traces to a JSON. Milestone: the journal manuscript is submit-ready."
layout: "post"
showTableOfContents: true
tags: ["research", "structural-entropy", "survey", "meta-science", "reproducibility"]
---

## The problem with survey claims

Survey papers make quantitative-sounding claims constantly — "most SE work
targets community detection", "signed graphs are unexplored" — that are almost
never auditable. After our [reproduction
campaign](/projects/2026-06-04-structural-entropy-benchmark/) caught published
claims that dissolved on contact with the original code, we did not want our
own survey to be the next offender.

The fix: treat the literature itself as a dataset.

## The corpus

- **81 core papers** (2015–2026), 100% full-text grounded — every paper's PDF
  or final text was ingested, not just abstracts.
- **10 facets** per paper: graph type, task, SE representation, optimization
  style, the *role* SE plays, supervision, output, domain, node distribution
  π, year.
- **1,297 verified tags** in tidy long format (`key, facet, value, confidence,
  evidence, source, verified`) — one row per claim, each with an evidence
  string quoting the paper.
- Machine proposes, human verifies; corrections land in an audit trail
  (`tag_overrides.json`).

Everything downstream — the graph-type × task coverage table, the phase
diagram, the role taxonomy, the "periodic table" on the project site — is
**generated from this one JSONL** by a script. If a number in the paper
disagrees with the corpus, the build is wrong, not the prose.

## What the discipline caught

A claims-audit pass over the manuscript found five numeric claims that did not
match their source JSONs (seed values, "wins on every graph" that was actually
4-of-6, a default-run number quoted as a tuned one). One citation was nearly
attributed to a paper that doesn't contain the claim. And one paper's abstract
had been silently poisoned by a wrong Semantic Scholar match — a supply-chain
paper masquerading as a graph-theory book — which a single bad tag exposed.

None of these were malicious. All of them are the default state of survey
writing. Tidy data + verification discipline is the cheapest known cure.

## Milestone

With the corpus closed and the audit green, the journal manuscript (TGINA
extension of our IJCAI'25 survey) is **submit-ready**: 33 pages, clean compile,
all referee blockers cleared, plus a full A–F future-research roadmap. The
remaining blocker is external — the journal's official template isn't public
yet.

*Artifacts: the tag corpus ships with per-tag evidence strings in the
[survey artifact repo](https://github.com/SuuTTT/structural-entropy-survey-clean);
the benchmark lives in
[structural-entropy-benchmark](https://github.com/SuuTTT/structural-entropy-benchmark).*
