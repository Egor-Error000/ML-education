---
name: ml-course-planning
description: Plans ML, computer vision, NLP, and LLM courses with source inventories, semantic coverage maps, chapter scenarios, dependency paths, and auditable completeness checks. Use when designing or revising technical learning plans.
---

# ML course planning

## Workflow

1. Read the project instructions and identify the course scope, audience, deliverables, and date-sensitive requirements.
2. Inventory existing plans, chapters, notes, discussions, and archived source files before proposing chapters. Preserve source paths and provenance.
3. Build an editorial map before restructuring. Map every meaningful question, definition, claim, formula, derivation, example, limitation, error, and diagram to a specific destination. Mark duplicates only after checking their distinct details.
4. Separate recovered material from additions. If a source claim is wrong, preserve the question and route it to a corrected explanation; never silently repeat it as fact.
5. Organize independent modules and explicit prerequisite paths. Tag every chapter as `общее ядро`, `основа ветки`, or `углубление`. Give each chapter exact prerequisites and downstream users. A skippable chapter must not be an unstated prerequisite for the core route.
6. Write one chapter scenario per chapter. Specify its learning question, ordered subsections, concepts, formulas and assumptions, worked example, diagram, practice, limitations/errors, and observable understanding criteria.
7. Research current methods where requested or where methods are rapidly changing. Link primary evidence and state what is established, preliminary, or implementation-specific.
8. Audit against the source map, module structure, routes, and chapter-scenario template. A count of files or headings is not proof of semantic coverage.

## Chapter scenario template

```markdown
### [Chapter title]
**Level:** ...  
**Prerequisites:** ...  
**Used by:** ...

**Learning question and outcome:** ...
**Order:** 1) ... 2) ... 3) ...
**Definitions and math:** ...; state assumptions and dimensions.
**Worked example:** ...
**Diagram:** ...; show inputs, outputs, blocks, links, and explain their meaning.
**Practice:** ...
**Limits and common errors:** ...
**Understanding check:** ...
```

## Quality rules

- Put the full definition where a term first appears; later mentions link back and use a short reminder.
- Explain a method's motivating failure, changed object, mechanism, empirical evidence, and theoretical guarantee separately.
- Do not use a benchmark, projection, visualization, or cited paper as proof beyond what it actually measures.
- Use local definitions when a chapter belongs to a standalone module.
- Keep universal writing instructions in this skill; keep subject choices and source mappings in course plans.
