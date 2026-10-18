# Жизненный цикл LLM и обучение агентов: учебные материалы

[Версия для чтения с набранными формулами](<../../../Читать конспекты.html#course-10-0>) · работает без интернета.

[План](<../План — жизненный цикл LLM и обучение агентов.md>) · [Общее оглавление](<../../../Учебные конспекты.md>)

Десять глав соответствуют десяти пунктам плана. Систематическое agentic training в исходных чатах не подтверждено: это подробное дополнение по плану, опирающееся на обсуждённые SFT, RLHF, tools, orchestration и serving. Глава А дополнительно углублена: diagnostic hypotheses, layer-specific interventions, условный ceiling, constrained selection/amortization и version bundle. Главы Б–В дополнены POMDP/belief, тремя toy trajectories/returns, termination/truncation, teacher forcing, causal masks/gradient и проверкой preprocessing. Главы Г–Д дополнены induced-distribution shift, expert labels/interventions, mixture/aggregation, ORM/PRM, noisy selection, STaR/Reflexion и cost/coverage calculations. Главы Е–Ж дополнены пошаговыми gradient/GAE/clipping/GRPO/KL расчётами и 12-event memory case с summary/state/versions/accounting. Главы З–И дополнены trajectory rubric/Pareto, preference likelihood и совместимостью среды, а также shared policies, non-stationary partners, CTDE и cross-play. Глава К дополнена полным десятисессионным worksheet, разными units/denominators, cost accounting, timeouts и binomial uncertainty. Все десять глав получили углубления; итоговая содержательная сверка остаётся необходимой. Наличие этих текстов не меняет статус вашего освоения и не означает, что training experiments уже выполнены.

1. [А — Lifecycle и выбор вмешательства](<А — Lifecycle и выбор следующего вмешательства.md>): диагностика задачи, prompting/RAG/SFT/RL, критерии выбора и versioned artifacts. Пункт 1.
2. [Б — Среда и reward](<Б — Среда, observations, actions, reward и horizon.md>): state/observation, история, переходы, частичная наблюдаемость, return, horizon и stop conditions; единая toy-среда. Пункт 2.
3. [В — Behavior cloning](<В — Behavior cloning и SFT на tool trajectories.md>): token-level objective, роли и masks, корректные observations, dependency и recovery demonstrations. Пункт 3.
4. [Г — Distribution shift и DAgger](<Г — Distribution shift, recovery и DAgger.md>): экспертные и посещённые истории, накопление ошибок, агрегирование данных и контролируемое сравнение. Пункт 4.
5. [Д — Reasoning и вычислительный бюджет](<Д — Reasoning, верификация и test-time compute.md>): CoT-SFT/STaR, ORM/PRM, rejection, best-of-N и self-critique. Пункт 5.
6. [Е — RL, PPO и GRPO](<Е — RL, credit assignment, PPO и GRPO.md>): delayed reward, policy gradient, advantage, clipping, group normalization, reference policy и reward hacking. Пункт 6.
7. [Ж — Память и длинный горизонт](<Ж — Память, summary и длинный горизонт.md>): structured state, summaries, evidence, checkpoints, hierarchical plans и ablation. Пункт 7.
8. [З — Предпочтения траекторий](<З — Предпочтения между полными траекториями.md>): rubric, constraints, Pareto, policy likelihood, judge biases и границы переноса DPO. Пункт 8.
9. [И — Multi-agent](<И — Multi-agent, роли и обучение координации.md>): роли против обучения, корреляция ошибок, adapters, debate/self-play и одинаковый общий budget. Пункт 9.
10. [К — Оценка траекторий](<К — Оценка траекторий, стоимость и воспроизводимость.md>): outcome oracle, recovery/loops, стоимость всех попыток, first-attempt и budgeted success, splits/seeds и отчёт. Пункт 10.

## Как проходить

Главы А–В задают основу. Оценку из К нужно спроектировать до обучения; затем выбирать одно вмешательство из Г–И по диагностике. Для теоретического режима достаточно записанных игрушечных траекторий и ручных вычислений, но результаты моделирования следует отличать от настоящего model training.

В каждой главе есть объяснение механизма, формулы LaTeX, разобранные примеры, ограничения, контрольные вопросы с ответами и практика. Примеры с числами — учебные расчёты, а не показатели ваших моделей.

## Связь с исходными чатами

S3:228–423 — SFT/RLHF; S3:509–673 — preferences; S3:820–918 — evaluation. S5:417–1034 — long context/cache; S5:1821–2822 — state/orchestration. S6:4–178 — tool execution; S6:1377–2197 — frameworks/роли; S6:3013–3904 — serving/scaling. Имена и сохранность источников: [реестр миграции](../../migration/MIGRATION_PLAN.md). Уточнённые границы каждой главы указаны в её начале.

Смежные подробные курсы: [адаптация](../../02-adaptation/Учебные%20конспекты/README.md), [инференс](../../03-inference/Учебные%20конспекты/README.md), [приложения](../../04-applications/Учебные%20конспекты/README.md). Runtime validation, retries и tool execution следует читать вместе с материалами приложений; обучение policy их не заменяет.
