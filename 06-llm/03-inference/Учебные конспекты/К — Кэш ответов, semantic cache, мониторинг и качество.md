# К. Application cache и мониторинг качества

[Оглавление](README.md). Объединены E.3–E.4, практические протоколы5–6.

**Происхождение:** prompt caching подробно обсуждался в S5:417–1034, deployment/scaling — S6:2201–3904. Отдельные E.3/E.4 являются следующими темами после конца исходной цепочки. **Эта глава — дополнение/продолжение по плану**, а не выдуманное прошлое занятие. Различие cache layers опирается на вопросы чата; keys/invalidation и monitoring углублены редакционно.

## 1. Какой именно результат сохраняется

KV/prefix cache хранит intermediate states, ускоряя повторный compute. Answer cache хранит final response/structured result и может вообще пропустить generation. Tool-result cache хранит результат внешней операции, например snapshot документа. Semantic cache ищет похожий query, чтобы переиспользовать answer/result по принятой policy.

Эти уровни имеют разные validity conditions. Одно и то же prefix P с разными вопросами допускает KV reuse, но ответы различаются. Одинаковый вопрос с новыми underlying data может требовать нового ответа даже при identical text. Одинаковый tool query в другом authorized scope не должен возвращать чужие данные.

## 2. Exact cache key и computation identity

Пример key components: normalized input, user/tenant или permission scope, model/adapter revision, system/template version, decoding policy, corpus/index snapshot, tool result versions и relevant locale/time assumptions. Это не универсальная обязательная структура: включается всё, что меняет допустимый результат выбранного cache.

Пусть запрос «какая версия API?» вчера отвечался по docs v1, сегодня docs v2. Key только по строке вернёт stale answer. Если answer зависит от current date или account plan, эти dependencies должны быть versioned/invalidated либо query не следует cache таким способом.

Canonicalization помогает exact hits: упорядочить незначимые whitespace/JSON keys, если это действительно сохраняет semantics. Но нельзя удалять отрицания, numbers, entity IDs или порядок instructions ради hit rate. «Платёж разрешён» и «платёж не разрешён» отличаются одним словом и противоположным смыслом.

TTL ограничивает возраст записи, но не доказывает correctness до expiration: данные могут измениться через секунду. Event/version invalidation точнее привязывает cache к изменениям. Нужны policy deletion/revocation и handling entries с ошибками; temporary tool timeout не должен стать долгим cached «ничего не найдено» без осознанного negative-cache TTL.

## 3. Semantic similarity не равна взаимозаменяемости ответа

Dense embedding может сблизить «удали черновик42» и «покажи черновик42», «лимит100» и «лимит1000». Для retrieval это похожие темы, для answer reuse — разные задачи. Threshold cosine0.95 не является доказательством одинакового смысла или authorization.

Учебный safe pattern для read-only FAQ: filter by scope/version/task type → exact entity/number/negation checks → semantic candidates → separate acceptance rule → reuse verified answer с provenance, либо обычный generation. Для действий с side effects final answer cache не означает, что действие уже выполнено в новом контексте.

Оценивать semantic-cache precision: доля accepted hits, где reused answer допустим; recall/hit rate — сколько работы удалось сэкономить; latency/cost и stale/error rate. Если высокая цена false hit, лучше меньший hit rate. Human audit должен включать near-threshold и counterexample queries, а не только очевидные synonyms.

## 4. Когда caching вообще выгоден

Пусть cache lookup5ms, обычный answer1000ms, accepted hit rate h, hit validation ещё5ms. Упрощённое expected time $5+h\cdot 5+\left(1-h\right)\cdot 1000$. При h0.5 получим507.5ms; при h0 —1005ms, cache только добавил overhead. При false hits скорость не считается успешным решением: quality constraint первична.

Аналогично cost включает lookup/vector storage/validation/maintenance. Semantic judge на каждый query может съесть значительную часть экономии. Сначала измерить actual repetition и sensitivity to stale results; затем сравнивать exact cache, semantic cache и no cache на одном workload.

Stampede: много simultaneous misses одного expensive key создают повторную генерацию. Single-flight/request coalescing может объединить работу, если inputs/scope equivalent. Cancellation одного клиента не должна неправильно отменить результат, нужный другим. Distributed invalidation/concurrency требуют своей semantics.

## 5. Мониторинг технического состояния и качества

Operational metrics: requests, queue age, TTFT/inter-token/end-to-end p50/p95, timeout/error/OOM, input/output lengths, cache hit/reject/stale rates, version distribution и cost. Их следует привязывать к workload slices: language, input length, tenant/task type, retrieved source quality.

Quality metrics: task correctness, schema validity, wrong-tool/args, retrieval relevance, evidence/citation support, false abstention/refusal и successful completion. Ни latency, ни user thumbs-up в одиночку не доказывают correctness. Feedback имеет selection bias: довольные/недовольные пользователи оставляют оценки с разной вероятностью.

Delayed labels требуют оценивать quality после появления результата, сохраняя связь с version/time cohort. Drift input embeddings/length/language может быть ранним indicator, но не равен доказанному падению task quality. Concept drift может ухудшить correctness без большого сдвига внешней формы запросов.

## 6. Regression suite и judge

Фиксированный curated набор ловит известные failures; свежая случайная sample production показывает новые. Оба нужны: static benchmark может загрязняться tuning, а только fresh sample плохо воспроизводится. Учитывать source/group/time dependence при uncertainty.

LLM judge получает rubric, evidence, blind/order-swapped outputs и проверяется на human subset. Это дополнение к deterministic oracle там, где он доступен. Изменение judge version может изменить score при той же policy — сохранять evaluator identity.

Перед rollout задать gates, например «не ухудшать exact extraction correctness в ключевом slice» и «p95 в определённом workload ниже budget». Конкретные numbers определяет application, а не универсальный учебник. Для rollout/rollback policy нужно достаточно observations, чтобы не реагировать на один случай как на устойчивый trend.

## 7. Полный эксперимент и самопроверка

Сравнить no cache, exact cache и один semantic policy: один corpus snapshot/model, фиксированные queries с synonyms, negations, different numbers, permissions и changed-data cases. Измерить correctness reused outputs, accepted hit rate, latency/cost. Затем invalidation по document version и permission revoke.

1. Почему TTL не гарантия freshness? **Ответ:** underlying state может измениться до expiration.
2. Почему cosine similarity недостаточно? **Ответ:** тематическая близость не доказывает equality entities/constraints/intent.
3. Можно ли alarm на embedding drift считать quality regression? **Ответ:** это indicator; нужен task-specific evidence.
4. Чем application answer cache отличается от prompt cache? **Ответ:** возвращает готовый output вместо reuse intermediate prefix computation.

**Практика:** составить cache policy для documentation FAQ с key dependencies и forbidden-to-reuse cases. Проверить десять пар запросов, включая «включить/выключить», разные IDs и даты. Объяснить, где hit допустим и какое evidence нужно для этого решения.
