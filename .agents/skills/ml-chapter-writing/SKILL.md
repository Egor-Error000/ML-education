---
name: ml-chapter-writing
description: Writes and reviews rigorous ML, computer vision, NLP, and LLM teaching chapters with self-contained definitions, transparent derivations, worked examples, diagrams, practice, and limits. Use when drafting or revising a course chapter.
---

# ML chapter writing

## Drafting order

1. State the task and the concrete question the chapter answers. Name prerequisites and the next topics that use it.
2. Define each new term at first use, including what object it refers to and its units or shape when relevant. Do not make the reader infer a definition from later examples.
3. Build the explanation in causal order: motivating failure or need → assumptions → mechanism → equation or algorithm → worked example → interpretation.
4. Derive formulas from stated assumptions. Define symbols, dimensions, normalization, objective, and which quantities are fixed or learned. Do not skip the step that makes the conclusion follow.
5. Work through at least one small example with visible intermediate values. State which numbers are illustrative and which are measured results.
6. Draw an original, labelled diagram for each architecture or multi-step pipeline. Show inputs, outputs, blocks, direction of data/control, masks or losses where useful, and explain the diagram in prose.
7. Add practice that changes one interpretable factor, asks the reader to inspect errors, and connects the result to the method's assumptions.
8. State failure modes, scope, tradeoffs, and evidence limits. Separate theoretical guarantees from empirical findings and implementation behavior.
9. End with observable criteria: the reader can explain the mechanism, reproduce the worked calculation, choose a fitting baseline, identify a limitation, and interpret an experiment.

## Reuse and references

- At the first occurrence, write the full explanation. Later chapters use a concise reminder and link to that explanation.
- Follow `ml-current-research` for changing methods, libraries, model families, and benchmark claims. Cite primary sources near the claims they support; give versions and dates for implementation details.
- Preserve meaningful source examples and questions. Correct faulty statements while explaining why the correction matters.
- Do not claim the learner has mastered a topic because the chapter was read. Keep practice, assessment, and personal progress separate.

## Diagram review

For a model or pipeline, verify that the diagram identifies input representation, each transformation, learned and fixed components, branching/skip or residual links, output and its semantics. Add tensor shapes or data contracts when they disambiguate the flow. A decorative block diagram without a narrated data path is incomplete.
