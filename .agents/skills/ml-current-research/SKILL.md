---
name: ml-current-research
description: Finds and evaluates current ML, vision, NLP, and LLM research for course plans and technical chapters. Use when a method, library, benchmark, model family, or best-practice claim may have changed.
---

# Current ML research

## Research workflow

1. State the exact question, task, data regime, comparison, and date of the snapshot. Do not search for an unqualified "best method".
2. Search primary sources first: peer-reviewed papers/proceedings, official model or library documentation, source repositories, and benchmark protocols. Prefer the paper and implementation that support the particular claim.
3. Check the source date, version, task, benchmark split, baselines, compute, and evaluation conditions. Separate author-reported results from independent replication and production evidence.
4. Compare a new method with the course's existing methods by intended use, changed mechanism, expected benefit, cost, failure modes, and evidence. Include it only when that comparison teaches a decision the learner needs.
5. Label evidence plainly: established mechanism; peer-reviewed empirical result within stated conditions; preprint or workshop result; implementation-specific behavior; or open question. Avoid turning one benchmark result into a universal ranking.
6. Record title, authors/maintainer, publication or release date, primary URL, version, checked-on date, and a one-sentence relevance note in the plan. Keep this research ledger in the plan, not as a dated aside in learner-facing prose.
7. Before writing the chapter, re-open the sources and re-check version, availability, benchmark protocol, and claims. Update the ledger if the state changed.

## Report format

For each topic, record:

```text
Question / task:
Candidate method and what it changes:
Evidence and setting:
Compared with:
Limitations / unknowns:
Decision: core | branch foundation | enrichment | omit
Sources: primary links, dates, versions
Checked on:
Recheck before chapter writing:
```

Use sources to support bounded claims. When no reliable source supports a performance or production claim, omit the claim or say evidence is insufficient. Distinguish inference from what the source directly reports.
