# В. Orchestration: состояние, переходы и воспроизводимый loop

[Оглавление](README.md). Пункт3 плана.

**Основа:** S5:1038–2822, S6:1377–2197. Сохранены вопросы о локальных Ollama/LM Studio models и о том, реализован ли каждый AutoGen participant через LangGraph. **Исправления:** одно framework name не определяет internals; loop может быть graph, messages или обычным code; orchestration не равно agent training. **Углубление:** reducers/checkpoints и recovery side effects.

## 1. Chain, branch и loop

Chain — фиксированная последовательность stages. Branch выбирает дальнейший путь по state. Loop повторяет часть работы до condition. В Python это можно выразить functions/if/while; frameworks добавляют integrations, checkpointing, streaming и удобное описание transitions.

Пример documentation assistant:

```text
prepare → retrieve → inspect evidence
                    ├ enough → answer → validate → end
                    ├ retry allowed → rewrite query → retrieve
                    └ no evidence/budget → abstain → end.
```

Каждый arrow должен иметь condition. Если «model решает, искать ещё или хватит», в state должны быть observations, remaining budget и criterion enough evidence. Prompt «будь умным» не задаёт bounded algorithm.

## 2. State и ответственность узлов

State хранит messages, original request, authorization scope, document IDs/versions, observations, pending calls, attempts и result. Не все поля должны целиком копироваться в каждый model prompt: это снижает cost и риск смешать trusted constraints с untrusted text.

Node contract задаёт, что читается и что обновляется. Если два parallel nodes добавляют evidence, reducer должен объединить по IDs/version, а не оставить последний список. Для scalar budget надо учитывать суммарный расход, а не minimum/last update случайно.

Original user constraints полезно хранить отдельно от model summary. Если summary потерял «только документы2025года», retrieval не должен расширить scope молча. Structured trusted state задаёт invariant, model-generated text — содержимое для обработки.

## 3. Checkpoint и resume

Checkpoint сохраняет состояние workflow после определённых transitions, чтобы восстановить работу. Но saved messages не автоматически доказывают, какие external tools действительно были executed. В state нужны call IDs/statuses и durable result provenance.

Crash scenario: mutation выполнилась, process упал до сохранения observation. Resume из previous checkpoint может повторить operation. Поэтому checkpointing сам по себе не гарантирует exactly-once side effects; idempotency/status reconciliation остаются задачами execution layer.

Для read-only учебного поиска repeat безопаснее, но results могут измениться при новой версии corpus. Воспроизводимый replay использует saved observations/snapshot или явно отмечает, что retrieval выполнен заново по другим данным. Равенство graph topology не означает равенство inputs.

## 4. Frameworks и local models

LangChain-style components помогают связать templates/models/tools/parsers; graph runtimes — задавать state/transitions/checkpoints; data frameworks — ingestion/retrieval. AutoGen-style participant coordination может использовать messages/handlers и внутренние tool loops. Оно не обязано быть LangGraph instance и не обязано исключать графовое описание на другом уровне.

Ollama/LM Studio могут дать local inference endpoint; orchestrator отправляет context/tools, получает model output, валидирует и исполняет actions. Compatibility части API не доказывает одинаковые chat templates, tool formats, structured generation и cancellation. Выбранные model/runtime/version должны быть tested на одном простом call до сложной многоагентной схемы.

В исходнике каждый agent описывался формулой «LLM+prompt+tools+loop». Это полезный common pattern, но не обязательное определение всех agents: бывают deterministic participants, actors без tools и другие abstractions. Содержательно важно, где decision, state, execution и termination.

## 5. Multi-agent coordination и бюджет

Researcher находит sources, reviewer проверяет claims, writer оформляет результат — возможное разделение функций. Польза появляется, если роли дают разные проверяемые operations/evidence. Три одинаковых prompts одной model могут просто втрое повторить один bias.

Coordinator должен определить recipients, message schema, доступный scope, handoff state и общий budget. Nested loops легко создают unbounded repeated work: каждый participant «ещё чуть-чуть поищет», но никто не проверяет global stop. По этой причине single well-defined workflow — обязательный baseline сравнения, а не предположение, что multi-agent хуже/лучше.

Сравнивать total model tokens/tool cost/latency и task success при equal budget. If agent A criticises agent B, это не independent truth; полезность review проверяется обнаруженными исправлениями и независимым oracle.

## 6. Самопроверка и практика

1. Чем graph отличается от обучения policy? **Ответ:** graph организует выполнение; policy training меняет параметры по data/objective.
2. Что потеряет last-write-wins evidence reducer? **Ответ:** результаты parallel branch, обновлённые раньше другого.
3. Гарантирует ли checkpoint отсутствие duplicate actions? **Ответ:** нет, external side effect и сохранение checkpoint могут быть неатомарны.

**Практика:** спроектировать graph с retrieve/rewrite/answer/validate и global cap3searches. Добавить checkpoint после search; описать crash перед сохранением observation. Показать recovery отдельно для read-only search и report-creation action. Код framework пока не нужен: сначала проверяется semantics.
