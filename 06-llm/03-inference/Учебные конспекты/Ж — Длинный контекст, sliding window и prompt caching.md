# Ж. Длинный контекст и повторное использование prefix

[Оглавление](README.md). Пункт C.4; KV memory из A.6 и отличие от E.3.

**Основа:** S5, строки 417–1034. Сохранён вопрос о prompt/context caching «на примере стандартного проекта и нескольких запросов». **Исправления C14:** возможности/TTL/тарифы провайдера не вечные факты, cached prefix не готовый answer, последующий query всё равно взаимодействует с prefix. **Углубление:** exact prefix examples, cache boundaries и оценка long-context quality.

## 1. Длина окна и полезное использование — разные свойства

Maximum context определяет supported input/total token limits по runtime/model contract. Оно не доказывает, что модель одинаково хорошо извлекает evidence из любой позиции и любой длины. Ограничения на input/output budgets и chat template special tokens тоже входят в фактическую длину.

Проверять task quality при разном положении evidence: начало, середина, конец; добавлять distractors и unanswerable queries. «Lost in the middle» — описанное в исследованиях поведение использования длинного context, а не гарантированный дефект каждой model. [Исследование long-context retrieval](https://arxiv.org/abs/2307.03172).

Новый positional scaling/extension method может позволить обработать более длинный input, но его accuracy требует проверки. Успешный allocation512k tokens не доказывает точность reasoning по512k. Chunking/retrieval и selection evidence могут быть полезнее передачи всего корпуса в каждый prompt.

## 2. Sliding window меняет доступ к истории

В локальном attention query читает ограниченные прошлые positions. Для window W стоимость взаимодействий и retained cache можно ограничить по architecture/runtime. Но данные вне window не доступны напрямую этому query, если отдельный mechanism не сохраняет/передаёт их.

Пример: W4, current position10 имеет разрешённые previous/current positions по конкретной convention, например7–10. Факт на позиции2 нельзя заново прочитать обычным local attention10. Более глубокие layers могут переносить информацию косвенно, но это не equivalent full history access.

Нельзя просто обрезать cache full-attention model и назвать её «бесплатно работающей с любым long context». Это меняет computation; модель могла не обучаться такой policy. Quantized KV уменьшает bytes, сохраняя больше positions с ошибкой; sliding window удаляет dependencies. Эти trade-offs разные.

## 3. Prompt caching на понятном проекте

Пусть документационный assistant получает:

```text
P = system rules + fixed tool definitions + документация версии v1
Q1 = вопрос об авторизации
Q2 = вопрос о формате ошибки
Q3 = вопрос о таймаутах.
```

Запрос1 обрабатывает $P+Q_{1}$, создавая K/V prefix. Запрос2 при exact совпадении P и compatible context может reuse его states и вычислить новый suffix Q2; затем генерирует новый answer. Запрос3 аналогично. Ни answer1, ни observation от инструмента1 автоматически не являются answer2.

Общий prefix должен быть **одинаковым в токенах и порядке**. Семантически сходная переписанная документация не идентичный prefix. Если случайный request ID поставлен в самом начале, совпадение может оборваться раньше большой документации. Если stable instructions/docs стоят раньше переменного вопроса, reusable portion больше.

При этом prompt ordering должен сохранять смысл/security/task quality: нельзя ради cache переносить важные context distinctions туда, где application их потеряет. Provider может иметь explicit cache objects или automatic prefix matching; granularity/min length/TTL/cache keys/charging зависят от актуального API/runtime.

Этот учебный пример описывает механизм без копирования устаревающего vendor snippet из чата. При реализации измерить reported cached tokens/hit rate и latency, сверив текущий contract. [Пример prefix cache design vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/).

## 4. Почему новые queries всё ещё читают prefix

Saved K/V избавляют от повторного вычисления **states prefix**, но новые query positions всё ещё attend к этим keys/values. Даже при100% reusable prefix длинная история может стоить bandwidth и decode attention. Cache hit не делает полный запрос вычислительно бесплатным.

Если P20k tokens,Q100tokens, reusable prefix20k, uncached prefill suffix100 — это число новых processed positions, а не100-token equivalent любой модели: suffix queries имеют доступ к20k cached keys. Output generation тоже остаётся. Реальные savings оцениваются end-to-end при workload.

Prefix sharing может увеличивать effective concurrency за счёт memory reuse. Но eviction и memory pressure снижают hit rate. Смена model/adapters, positions/template или некоторых runtime settings делает прежний state несовместимым. Cache key должен представлять computation identity, а tenant isolation/authorization задаются отдельным layer.

## 5. Context engineering и память приложения

Суммаризация старого диалога даёт новый текст меньшего размера, а не lossless KV compression. Summary может потерять числа, отрицания и provenance. Важно сохранять trusted state — например, выбранный file ID и ограничения — структурированно, а не только пересказом модели.

Retrieval выбирает evidence из внешнего corpus, prefix cache экономит повторение compute, answer cache возвращает готовый result. Их можно сочетать: stable P cached, retrieved snippets/query меняются, final answer при необходимости cached по application policy. Ошибка — смешать эти уровни и reuse ответ только потому, что prefix identical.

## 6. Вопросы и практика

1. Можно ли hit на одинаковом по смыслу, но другом тексте? **Ответ:** exact KV-prefix reuse требует согласованных tokens; semantic answer cache — другой механизм.
2. Что меняется при документации v1→v2 внутри P? **Ответ:** downstream prefix states обычно уже не equivalent, нужна invalidation/recompute соответствующей части.
3. Делает ли длинное окно retrieval ненужным? **Ответ:** нет; cost, distractors и evidence quality остаются.

**Практика:** выписать три prompts P+Q1/2/3 и отметить maximal common token prefix. Затем добавить timestamp в начало, в конец и только в metadata: объяснить различие potential cache reuse, не делая ложного обещания конкретного provider behavior. Оценить answer quality при переносе evidence по positions отдельно от speed.
