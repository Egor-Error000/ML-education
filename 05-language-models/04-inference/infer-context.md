# Контекст, sliding window и prefix caching

**Уровень:** общее ядро.

**Предпосылки:** [От prompt к обычному циклу генерации](<infer-generation.md>), [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Что меняется при ограничении истории и повторном использовании prefix?





## Карта вычисления и границы этапов

![Контекст, sliding window и prefix caching: входы, преобразования и выходы](<../../оформление/course/infer-context.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:7b19a1df83eafa4dc0 -->
<a id="u-7b19a1df83eafa4dc0"></a>

## Длина окна и полезное использование — разные свойства

<!-- /unit:7b19a1df83eafa4dc0 -->

<!-- unit:4836fefa83b6aa1180 -->
<a id="u-4836fefa83b6aa1180"></a>

Maximum context определяет supported input/total token limits по runtime/model contract. Оно не доказывает, что модель одинаково хорошо извлекает evidence из любой позиции и любой длины. Ограничения на input/output budgets и chat template special tokens тоже входят в фактическую длину.

<!-- /unit:4836fefa83b6aa1180 -->

<!-- unit:0fb4f50af5dad0d8a3 -->
<a id="u-0fb4f50af5dad0d8a3"></a>

Проверять task quality при разном положении evidence: начало, середина, конец; добавлять distractors и unanswerable queries. «Lost in the middle» — описанное в исследованиях поведение использования длинного context, а не гарантированный дефект каждой model. [Исследование long-context retrieval](https://arxiv.org/abs/2307.03172).

<!-- /unit:0fb4f50af5dad0d8a3 -->

<!-- unit:3f4cab1692e4dbeb74 -->
<a id="u-3f4cab1692e4dbeb74"></a>

Новый positional scaling/extension method может позволить обработать более длинный input, но его accuracy требует проверки. Успешный allocation512k tokens не доказывает точность reasoning по 512k. Chunking/retrieval и selection evidence могут быть полезнее передачи всего корпуса в каждый prompt.

<!-- /unit:3f4cab1692e4dbeb74 -->

<!-- unit:3a238a575223112734 -->
<a id="u-3a238a575223112734"></a>

### Исправляем размер cache из исходного C.4

<!-- /unit:3a238a575223112734 -->

<!-- unit:d69fd147c4aeb47185 -->
<a id="u-d69fd147c4aeb47185"></a>

В S5 для 80layers,8KVheads,head_dim128 и двухбайтовых K/V было получено 640KB/token. Прямое умножение даёт 2·80·8·128·2=327680bytes=320KiB/token: исходный результат завышен вдвое. При 8K=8192tokens получаем 2.5GiB, при 128K=131072tokens —40GiB. Для десяти таких sequences —400GiB до allocation/scale/workspace overhead. В decimal GB числа будут другими; нельзя смешивать KB/GB с KiB/GiB в одном расчёте.

<!-- /unit:d69fd147c4aeb47185 -->

<!-- unit:bbfadbc2b818fb01bd -->
<a id="u-bbfadbc2b818fb01bd"></a>

Weight storage и KV storage независимы. Quantized weights той же модели могут занимать меньше места, но каждый resident long-context request добавляет свой cache, если нет поддерживаемого sharing. Наоборот, общий большой prefix нескольких requests иногда хранится совместно. Поэтому примерB10×T128K —верхняя оценка без такого sharing, а не обязательная memory цифра любого service с десятью зарегистрированными пользователями.

<!-- /unit:bbfadbc2b818fb01bd -->

<!-- unit:43d4f18aa9c7dcedd6 -->
<a id="u-43d4f18aa9c7dcedd6"></a>

У full dense attention полный prefill имеет quadratic число взаимодействий по lengthN, а KV —linear число сохранённых vectors. На decode новая query взаимодействует с доступной history, поэтому его attention cost растёт примерно линейно по N, при прочих фиксированных dimensions. Формула «context вырос вдвое, весь запрос стал ровно вчетверо дороже» неверна для полной системы: linear projections/MLP, memory traffic, output length, batching и kernels также участвуют.

<!-- /unit:43d4f18aa9c7dcedd6 -->

<!-- unit:8b42c1378dad9f10da -->
<a id="u-8b42c1378dad9f10da"></a>

### RoPE: почему имеет значение расстояние

<!-- /unit:8b42c1378dad9f10da -->

<!-- unit:36dad593fb86b833a6 -->
<a id="u-36dad593fb86b833a6"></a>

В одной двумерной паре RoPE вращает q и k на углы iθ и jθ. Для стандартной rotation matrix R:

<!-- /unit:36dad593fb86b833a6 -->

<!-- unit:0883893e48d26a3505 -->
<a id="u-0883893e48d26a3505"></a>

$$
(R(i\theta)q)^\top(R(j\theta)k)=q^\top R((j-i)\theta)k.
$$

<!-- /unit:0883893e48d26a3505 -->

<!-- unit:4434c0d3bf6c2f7125 -->
<a id="u-4434c0d3bf6c2f7125"></a>

В scalar product возникает **relative displacement**. В полной head используются разные частоты для разных пар; сами q/k зависят от content и прежних layers. Поэтому identity не означает, что attention зависит только от расстояния и игнорирует текст. [RoFormer/RoPE](https://arxiv.org/abs/2104.09864).

<!-- /unit:4434c0d3bf6c2f7125 -->

<!-- unit:c2458b2fa77462eef6 -->
<a id="u-c2458b2fa77462eef6"></a>

При position interpolation p'=p/s переход из training window8K к planned32K даётs4: позиция 32000 отображается в 8000, соседние positions отличаются на 0.25 вместо 1. Это изменение фаз/разрешения, а не буквальное слияние всех соседних positions в одинаковый float. Интерполяция удерживает индексы в знакомом диапазоне, но меняет geometric relation, под которую weights учились. [Position Interpolation](https://arxiv.org/abs/2306.15595).

<!-- /unit:c2458b2fa77462eef6 -->

<!-- unit:cf85bdc4294af898a7 -->
<a id="u-cf85bdc4294af898a7"></a>

[YaRN](https://arxiv.org/abs/2309.00071) и NTK-inspired подходы меняют frequency/scale policy, пытаясь сохранить полезные расстояния на разных масштабах. Они не дают универсального рецепта «любой 8K checkpoint без обучения станет 32K, с LoRA станет 128K». Нужны actual training/extension recipe и evaluation коротких **и** длинных задач. Изменение RoPE parameters также меняет computation identity cache; прежние K/V не считаются совместимыми автоматически.

<!-- /unit:cf85bdc4294af898a7 -->

<!-- unit:8265eed0e37a1beb3f -->
<a id="u-8265eed0e37a1beb3f"></a>

## Sliding window меняет доступ к истории

<!-- /unit:8265eed0e37a1beb3f -->

<!-- unit:e1c4e23f0ae2b20691 -->
<a id="u-e1c4e23f0ae2b20691"></a>

В локальном attention query читает ограниченные прошлые positions. Для window W стоимость взаимодействий и retained cache можно ограничить по architecture/runtime. Но данные вне window не доступны напрямую этому query, если отдельный mechanism не сохраняет/передаёт их.

<!-- /unit:e1c4e23f0ae2b20691 -->

<!-- unit:2e093f3e4519f7c47d -->
<a id="u-2e093f3e4519f7c47d"></a>

Пример: W4, current position10 имеет разрешённые previous/current positions по конкретной convention, например 7–10. Факт на позиции 2 нельзя заново прочитать обычным local attention10. Более глубокие layers могут переносить информацию косвенно, но это не equivalent full history access.

<!-- /unit:2e093f3e4519f7c47d -->

<!-- unit:11b258aac8f814faac -->
<a id="u-11b258aac8f814faac"></a>

Нельзя просто обрезать cache full-attention model и назвать её «бесплатно работающей с любым long context». Это меняет computation; модель могла не обучаться такой policy. Quantized KV уменьшает bytes, сохраняя больше positions с ошибкой; sliding window удаляет dependencies. Эти trade-offs разные.

<!-- /unit:11b258aac8f814faac -->

<!-- unit:2fa50329a98d8e8cac -->
<a id="u-2fa50329a98d8e8cac"></a>

### Receptive field не равен гарантированному воспоминанию

<!-- /unit:2fa50329a98d8e8cac -->

<!-- unit:ec380cb9537546d984 -->
<a id="u-ec380cb9537546d984"></a>

Зафиксируем convention: окно W включает current token иW−1 предыдущих. ДляLlayers с таким attention graph максимальный путь по input positions может охватывать 1+L(W−1)tokens. ПриW4,L3 это 10positions. Это теоретическая достижимость через цепочку intermediate representations для фиксированного input, а не обещание сохранить каждый факт из этих десяти positions без потерь.

<!-- /unit:ec380cb9537546d984 -->

<!-- unit:e60d24e747af00180a -->
<a id="u-e60d24e747af00180a"></a>

Поэтому произведениеW×L из S5 не устанавливает supported context length модели и не эквивалентно full attention. Повторно прочитать точный старый source span и использовать косвенно перенесённую feature —разные возможности. Смешанные local/global layers, global tokens, dilated или block-sparse patterns дают другой graph. Каждый из них оценивают с actual mask, а не подставляют одинW для любой architecture.

<!-- /unit:e60d24e747af00180a -->

<!-- unit:58d174879e0f8cbdd3 -->
<a id="u-58d174879e0f8cbdd3"></a>

Если все layers допускают sliding cache с окномW, retained attention state на sequence может ограничиваться этим окном по layer. Но arbitrary cache truncation full-attention checkpoint меняет функцию. Quantization cache сохраняет больше history positions с численной ошибкой, eviction удаляет часть edges, summary создаёт новый текст. Эти способы сокращения затрат нельзя называть одним lossless context compression.

<!-- /unit:58d174879e0f8cbdd3 -->

<!-- unit:cd8c33721e71eb0709 -->
<a id="u-cd8c33721e71eb0709"></a>

### FlashAttention и Ring Attention решают другие части задачи

<!-- /unit:cd8c33721e71eb0709 -->

<!-- unit:54a738e68eca00ba33 -->
<a id="u-54a738e68eca00ba33"></a>

[FlashAttention](https://arxiv.org/abs/2205.14135) организует tiled exact attention и работу с промежуточными sums/maxima так, чтобы не хранить полнуюN×Nmatrix в HBM. При фиксированных прочих dimensions auxiliary storage может расти линейно поN, хотя arithmetic full attention остаётся quadratic. Это не превращение trained full-attention model в sliding-window model. Стабильная global softmax normalization при работе по tiles связана с формулой предыдущей главы serving.

<!-- /unit:54a738e68eca00ba33 -->

<!-- unit:bf59f2002086bfe38c -->
<a id="u-bf59f2002086bfe38c"></a>

[Ring Attention](https://arxiv.org/abs/2310.01889) распределяет sequence blocks между devices и организует обмен K/V с вычислением attention. Это помогает разместить и исполнять длинную sequence при suitable interconnect и реализации. Но добавление GPU не делает полный объём pair interactions линейным поN на всю систему. Нужно различать throughput, latency, суммарные FLOPs, память одного device и общий memory budget. Слово«почтилинейное» без указания оси scaling ничего не доказывает.

<!-- /unit:bf59f2002086bfe38c -->

<!-- unit:c83d97090098fa491b -->
<a id="u-c83d97090098fa491b"></a>

Mamba-подобная recurrent state архитектура хранит другую representation истории; это не runtime option, включаемая у любого Transformer. [Mamba](https://arxiv.org/abs/2312.00752) и hybrid[Jamba](https://arxiv.org/abs/2403.19887) иллюстрируют иные trade-offs. В S5 Jamba ошибочно приписывалась Mistral; работа принадлежитAI21. Если hybrid сохраняет некоторое число full-attention layers, у этой части остаётся quadratic full-prefill component. Обещание точного recall из любой длины не следует только из названияlinear-time.

<!-- /unit:c83d97090098fa491b -->

<!-- unit:e82cd0f8544e2b7522 -->
<a id="u-e82cd0f8544e2b7522"></a>

## Prompt caching на понятном проекте

<!-- /unit:e82cd0f8544e2b7522 -->

<!-- unit:e757b7883c9836e16d -->
<a id="u-e757b7883c9836e16d"></a>

Пусть документационный assistant получает:

<!-- /unit:e757b7883c9836e16d -->

<!-- unit:f07175b583a049162b -->
<a id="u-f07175b583a049162b"></a>

```text
P = system rules + fixed tool definitions + документация версии v1
Q1 = вопрос об авторизации
Q2 = вопрос о формате ошибки
Q3 = вопрос о таймаутах.
```

<!-- /unit:f07175b583a049162b -->

<!-- unit:4d7316b48d36bdc50b -->
<a id="u-4d7316b48d36bdc50b"></a>

Запрос 1 обрабатывает $P+Q_{1}$, создавая K/V prefix. Запрос 2 при exact совпадении P и compatible context может reuse его states и вычислить новый suffix Q2; затем генерирует новый answer. Запрос 3 аналогично. Ни answer1, ни observation от инструмента 1 автоматически не являются answer2.

<!-- /unit:4d7316b48d36bdc50b -->

<!-- unit:2336c909c806d1e923 -->
<a id="u-2336c909c806d1e923"></a>

Общий prefix должен быть **одинаковым в токенах и порядке**. Семантически сходная переписанная документация не идентичный prefix. Если случайный request ID поставлен в самом начале, совпадение может оборваться раньше большой документации. Если stable instructions/docs стоят раньше переменного вопроса, reusable portion больше.

<!-- /unit:2336c909c806d1e923 -->

<!-- unit:80ce15f8c303ca32d0 -->
<a id="u-80ce15f8c303ca32d0"></a>

При этом prompt ordering должен сохранять смысл/security/task quality: нельзя ради cache переносить важные context distinctions туда, где application их потеряет. Provider может иметь explicit cache objects или automatic prefix matching; granularity/min length/TTL/cache keys/charging зависят от актуального API/runtime.

<!-- /unit:80ce15f8c303ca32d0 -->

<!-- unit:b4a147d38c472bad08 -->
<a id="u-b4a147d38c472bad08"></a>

Этот учебный пример описывает механизм без копирования устаревающего vendor snippet из разбираемого примера. При реализации измерить reported cached tokens/hit rate и latency, сверив текущий contract. [Пример prefix cache design vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/).

<!-- /unit:b4a147d38c472bad08 -->

<!-- unit:6f5936c98d736dc5e5 -->
<a id="u-6f5936c98d736dc5e5"></a>

### Возвращаем исходный проект TechCorp

<!-- /unit:6f5936c98d736dc5e5 -->

<!-- unit:0f0ed2f5784acff9ec -->
<a id="u-0f0ed2f5784acff9ec"></a>

Ваш вопрос просил не только определение caching, но и несколько запросов в стандартном проекте. В S5 assistant работал с user-service,order-service,payment-service, их портами и общей документацией. Стабильные instructions занимали условно 8000tokens, documents15000; общий P=23000. Каждый отдельный вопрос Q добавлял ещё около 500tokens. Числа учебные: placeholders«здесь ещё 7500tokens» в коде сами не создают такой размер реального input.

<!-- /unit:0f0ed2f5784acff9ec -->

<!-- unit:18ebcf277b2462c7d3 -->
<a id="u-18ebcf277b2462c7d3"></a>

Первый вопрос: «Как устроен circuit breaker между сервисами?» Если подходящего cache нет, P и Q1 обрабатываются, states eligible prefix могут быть сохранены. Полученный ответA1 —новая generation, его содержание не является самим prefix cache. Второй вопрос: «Какие порты у payment-service и как подключиться из order-service?» При сохранённом совместимом P вычисление его состояний переиспользуется, но Q2 и новый ответA2 вычисляются заново. Третий вопрос о хранении секретов аналогичен только если cache ещё доступен и версия documents не изменилась.

<!-- /unit:18ebcf277b2462c7d3 -->

<!-- unit:50aad50f952cefabbd -->
<a id="u-50aad50f952cefabbd"></a>

В примере использовались **независимые** questions с однимP, а не обязательно диалог, где A1/Q1 включены передQ2. Если приложение добавило transcript, token prefix и lengths уже другие. Если жеP повторён, ответA1 не надо механически добавлять в новый запрос ради hit: это меняет evidence и может добавить нерелевантный текст. Сначала определяют intended context, затем находят его reusable часть.

<!-- /unit:50aad50f952cefabbd -->

<!-- unit:e7fd82f574bc4b30b8 -->
<a id="u-e7fd82f574bc4b30b8"></a>

### Изменение начала и изменение хвоста

<!-- /unit:e7fd82f574bc4b30b8 -->

<!-- unit:566fd41232fe84c3c8 -->
<a id="u-566fd41232fe84c3c8"></a>

Для запросов `[rules_v1, tools_v1, docs_v1, Q1]` и `[rules_v1, tools_v1, docs_v1, Q2]` общий prefix заканчивается перед различающимся question. При `[rules_v1, tools_v1, docs_v2, Q3]` earlier stable rules/tools ещё могут совпасть, но после изменившегося документа старые downstream states уже не эквивалентны. Это не обязательно полный miss от первого token.

<!-- /unit:566fd41232fe84c3c8 -->

<!-- unit:545066914eeb80f8a0 -->
<a id="u-545066914eeb80f8a0"></a>

Random request ID в начале может разрушить большой hit; тот же ID только в transport metadata не меняет model tokens, если template его не добавляет. ID в самом конце не разрушает предшествующий exact prefix. Разный порядок documents может менять и cache reuse, и качество ответа: canonical ordering полезен для воспроизводимости, но не повод игнорировать task-relevant ordering. Identity определяется реально поданным input и computation config, а не только визуальным равенством строк в editor.

<!-- /unit:545066914eeb80f8a0 -->

<!-- unit:f5b7bb2902da3c9edd -->
<a id="u-f5b7bb2902da3c9edd"></a>

Source называл обязательным побайтовое совпадение. Для KV принципиальны identical token/embedding prefix и совместимая computation semantics; serialization/template/tool definitions могут входить в это условие. Некоторые APIs дополнительно используют explicit cache IDs, block boundaries, scope или routing. Лишний пробел иногда меняет tokens уже в начале, но универсальное поведение любого provider по одной строке не выводится.

<!-- /unit:f5b7bb2902da3c9edd -->

<!-- unit:5678f592d768c00a0a -->
<a id="u-5678f592d768c00a0a"></a>

### Что изменяют RAG и персонализация

<!-- /unit:5678f592d768c00a0a -->

<!-- unit:fee39d22818ca5c9c5 -->
<a id="u-fee39d22818ca5c9c5"></a>

Динамический RAG не запрещает prefix caching вообще: stable rules/tools до retrieved documents могут повторяться. Разные questions к одному неизменному document bundle дают больший reusable prefix; разные document sets —меньший. Собрать один giant master set ради cache не всегда выгодно: растёт context, появляются distractors, меняется evidence selection. Он также должен соответствовать допустимому контексту пользователя, а не автоматически содержать все внутренние документы.

<!-- /unit:fee39d22818ca5c9c5 -->

<!-- unit:94b5ced05034d3eddd -->
<a id="u-94b5ced05034d3eddd"></a>

Персонализация в первом token дробит cache keys, но её перенос после stable rules не делает её неважной для приложения. Доступы и требуемые distinctions сохраняют, исполнение tools проверяет trusted state. Caching экономит вычисление уже **правильного** input, а не определяет, какой input пользователю следует разрешить или какие инструкции модель обязана игнорировать.

<!-- /unit:94b5ced05034d3eddd -->

<!-- unit:85c82b22be5882a858 -->
<a id="u-85c82b22be5882a858"></a>

Короткий prompt ниже provider threshold может оказаться неeligible. Предложение S5 искусственно «раздуть» его до порога не является общим optimization: дополнительные tokens тоже стоят памяти/compute и могут ухудшить task behavior. Добавлять полезные инструкции можно ради качества, но выгоду padding исключительно ради cache следует доказать расчётом, а не предполагать.

<!-- /unit:85c82b22be5882a858 -->

<!-- unit:b15d2e3bc314cd069d -->
<a id="u-b15d2e3bc314cd069d"></a>

## Почему новые queries всё ещё читают prefix

<!-- /unit:b15d2e3bc314cd069d -->

<!-- unit:2f946151a064bc537a -->
<a id="u-2f946151a064bc537a"></a>

Saved K/V избавляют от повторного вычисления **states prefix**, но новые query positions всё ещё attend к этим keys/values. Даже при 100% reusable prefix длинная история может стоить bandwidth и decode attention. Cache hit не делает полный запрос вычислительно бесплатным.

<!-- /unit:2f946151a064bc537a -->

<!-- unit:1f451bde150b15ae0b -->
<a id="u-1f451bde150b15ae0b"></a>

Если P20k tokens,Q100tokens, reusable prefix20k, uncached prefill suffix100 — это число новых processed positions, а не 100-token equivalent любой модели: suffix queries имеют доступ к 20k cached keys. Output generation тоже остаётся. Реальные savings оцениваются end-to-end при workload.

<!-- /unit:1f451bde150b15ae0b -->

<!-- unit:96d1a8eabeac9fc76c -->
<a id="u-96d1a8eabeac9fc76c"></a>

Prefix sharing может увеличивать effective concurrency за счёт memory reuse. Но eviction и memory pressure снижают hit rate. Смена model/adapters, positions/template или некоторых runtime settings делает прежний state несовместимым. Cache key должен представлять computation identity, а tenant isolation/authorization задаются отдельным layer.

<!-- /unit:96d1a8eabeac9fc76c -->

<!-- unit:64d0002b4729619c6d -->
<a id="u-64d0002b4729619c6d"></a>

### Hit не убирает чтение history новым suffix

<!-- /unit:64d0002b4729619c6d -->

<!-- unit:fe16bc3211dc91a092 -->
<a id="u-fe16bc3211dc91a092"></a>

Для P23000,Q500 full prefill строит states всех 23500positions. При compatible cached P новые Q positions всё ещё читают соответствующие cached keys/values. В dense causal attention их взаимодействия включают порядкаP·Q+Q(Q+1)/2 пар, до dimensions/heads. Для этих чисел это 11625250pairs. Вычисление стало существенно меньше повторного full-prefill attention, но это не то же самое, что model с context только 500tokens.

<!-- /unit:fe16bc3211dc91a092 -->

<!-- unit:7cd2f9632b53de87d6 -->
<a id="u-7cd2f9632b53de87d6"></a>

Новые output tokens тоже обращаются к P и растущему suffix. Поэтому cache hit прежде всего экономит повтор prefix prefill, а не гарантирует ускорения всех decode steps в ту же долю. Memory sharing, transfer/offload и locality могут дать другие эффекты, которые измеряются отдельно. Если cache надо загрузить с другой memory tier, hit имеет свою latency.

<!-- /unit:7cd2f9632b53de87d6 -->

<!-- unit:13775ae76dbda09d57 -->
<a id="u-13775ae76dbda09d57"></a>

### Счётчики и TTL из исходного кода

<!-- /unit:13775ae76dbda09d57 -->

<!-- unit:271b4cc5760634c6d7 -->
<a id="u-271b4cc5760634c6d7"></a>

В S5 были поля cache_creation/cache_read/non-cached input и времена 8.7/2.1/1.9s. Это иллюстративные outputs assistant, не результаты пользовательского запуска. Более того, output length между первыми двумя ответами различалась 340 и 280tokens: одно end-to-end время не выделяет чистый prefill effect. Для такого вывода нужны TTFT/prefill measurements при сопоставимых условиях, включая очередь, network и warmup.

<!-- /unit:271b4cc5760634c6d7 -->

<!-- unit:39f840ae3feb996d1e -->
<a id="u-39f840ae3feb996d1e"></a>

При logging нужно определить, означает ли total input уже сумму cached и uncached components, иначе cached tokens можно посчитать дважды. Request hit rate«хотя бы часть prefix прочитана» отличается от доли всех input tokens, которые были cached. Частичный hit на первые 1000tokens при 23000eligible tokens не равен полному reuse. Creation0/read0 также не доказывает, что новый cache создан: он мог быть disabled, ineligible или не сохранён.

<!-- /unit:39f840ae3feb996d1e -->

<!-- unit:9c2ce186dfb42dd89e -->
<a id="u-9c2ce186dfb42dd89e"></a>

TTL/минимальная длина/cache scope и billing зависят от актуального model/provider contract. Третий вопрос«ещё через 5minutes» из S5 не гарантирует hit на boundary expiry. Нужно знать, от какого события отсчитывается TTL, продлевается ли он, бывают ли eviction/routing misses и explicit control. Один cached request не даёт бессрочной гарантии на следующие. Таблицы старых vendor values не считаются текущей спецификацией.

<!-- /unit:9c2ce186dfb42dd89e -->

<!-- unit:d7a561cd070927fbb6 -->
<a id="u-d7a561cd070927fbb6"></a>

КлассCachedAssistant сохранял Python strings и HTTP client, но не локальные KV tensors. Connection pooling может уменьшать transport overhead, а provider prefix cache —вычисления модели; это разные механизмы. Новый client не обязан означать miss, старый —не обязан гарантировать hit. `hash(system_prompt + docs)` пригоден разве что как грубый local label: Python hash не обязан совпадать между процессами, concatenation не различает границы blocks, и model/template revision в нём отсутствуют. Для устойчивого monitoring identity нужен canonical manifest и declared digest; server cache identity определяется его собственным contract.

<!-- /unit:d7a561cd070927fbb6 -->

<!-- unit:7504e54c00343e173e -->
<a id="u-7504e54c00343e173e"></a>

## Context engineering и память приложения

<!-- /unit:7504e54c00343e173e -->

<!-- unit:acd635f3f9a29730bb -->
<a id="u-acd635f3f9a29730bb"></a>

Суммаризация старого диалога даёт новый текст меньшего размера, а не lossless KV compression. Summary может потерять числа, отрицания и provenance. Важно сохранять trusted state — например, выбранный file ID и ограничения — структурированно, а не только пересказом модели.

<!-- /unit:acd635f3f9a29730bb -->

<!-- unit:aa8d856dfb63e1b522 -->
<a id="u-aa8d856dfb63e1b522"></a>

Retrieval выбирает evidence из внешнего corpus, prefix cache экономит повторение compute, answer cache возвращает готовый result. Их можно сочетать: stable P cached, retrieved snippets/query меняются, final answer при необходимости cached по application policy. Ошибка — смешать эти уровни и reuse ответ только потому, что prefix identical.

<!-- /unit:aa8d856dfb63e1b522 -->

<!-- unit:38db3c03538b3bffd1 -->
<a id="u-38db3c03538b3bffd1"></a>

### Исправляем арифметику повторных запросов

<!-- /unit:38db3c03538b3bffd1 -->

<!-- unit:baa18ee10207d7a9ba -->
<a id="u-baa18ee10207d7a9ba"></a>

Для binary модели «полный hit или полный miss» пусть N1000requests, P23000eligible tokens, Q500uncached suffix и hit rate0.9. Тогда 100misses и 900hits дают 2.3Mtokens создания prefix,20.7Mtokens чтения и 0.5Mнового suffix. В S5 одновременно предполагались 90%hit, лишь 3creations и 1000reads —это не один согласованный experiment. Если ровно 3misses из 1000, полный hit rate99.7%, а reads997.

<!-- /unit:baa18ee10207d7a9ba -->

<!-- unit:0dfe92ed4fc03153b5 -->
<a id="u-0dfe92ed4fc03153b5"></a>

Пусть unit rates prefix write/read/обычной обработки равныc_w,c_r,c_u. Тогда input cost в этой declared модели:

<!-- /unit:0dfe92ed4fc03153b5 -->

<!-- unit:8a0cdcd8e12050eb23 -->
<a id="u-8a0cdcd8e12050eb23"></a>

$$
C=N\left(P\big((1-h)c_w+h c_r\big)+Q c_u\right).
$$

<!-- /unit:8a0cdcd8e12050eb23 -->

<!-- unit:2272ef5fb227a5509d -->
<a id="u-2272ef5fb227a5509d"></a>

Output cost добавляется отдельно. Для безкэшевого случая input=N(P+Q)c_u. Приc_w=1.25c_u,c_r=0.1c_u caching prefix выгоден в ожидании только еслиh>(1.25−1)/(1.25−0.1)≈0.21739. Это собственный toy calculation, а не актуальный тариф. Малое число повторов, частые expiry и платное хранение меняют результат. Если miss оплачивается иначе, formula тоже меняется по contract.

<!-- /unit:2272ef5fb227a5509d -->

<!-- unit:ad4cd895be7df46bbe -->
<a id="u-ad4cd895be7df46bbe"></a>

### Quality проверяют независимо от cache economy

<!-- /unit:ad4cd895be7df46bbe -->

<!-- unit:ce95de4d8b949e5df8 -->
<a id="u-ce95de4d8b949e5df8"></a>

В recall experiment сохраняют известный факт и ответ, меняют его position при сопоставимой total length, добавляют distractors и unanswerable cases. Затем отдельно проверяют multi-document reasoning и правильность citations. Найденная needle не доказывает полноценного понимания всей codebase. [Lost in the Middle](https://arxiv.org/abs/2307.03172) описывает empirical pattern, а не математическую неизбежность любой attention architecture.

<!-- /unit:ce95de4d8b949e5df8 -->

<!-- unit:b82c34bbec2b8f2e5d -->
<a id="u-b82c34bbec2b8f2e5d"></a>

Для проекта полезны две независимые оси: quality нужного context bundle и reuse/cost того же bundle. Retrieval может повысить evidence quality и уменьшить input, summary —потерять важное отрицание, а длинный cached corpus —быть дешёвым при повторе, но всё равно вводить модель в заблуждение. Проверяют вопросы с точными numbers/версией документа, противоречивыми источниками и отсутствующим ответом. Дальше измеряют cached tokens, TTFT/decode latency и total cost, сохраняя raw outputs для разбора ошибок.

<!-- /unit:b82c34bbec2b8f2e5d -->

<!-- unit:e3555eb60b9b6da5a7 -->
<a id="u-e3555eb60b9b6da5a7"></a>

Упражнение из исходного вопроса теперь конкретно: подготовить manifest stableP и триQ, найти maximal common token prefix, затем изменить docs version, место timestamp и порядок retrieval snippets. Предсказать, какая часть потенциально reusable, проверить фактические counters выбранного runtime и отдельно оценить correctness. Не требуется объявлять successful cache hit или deploy без такого наблюдения.

<!-- /unit:e3555eb60b9b6da5a7 -->

<!-- unit:843505721ddd097225 -->
<a id="u-843505721ddd097225"></a>

## Вопросы и практика

<!-- /unit:843505721ddd097225 -->

<!-- unit:5594a1f86325e67a01 -->
<a id="u-5594a1f86325e67a01"></a>

1. Можно ли hit на одинаковом по смыслу, но другом тексте? **Ответ:** exact KV-prefix reuse требует согласованных tokens; semantic answer cache — другой механизм.
2. Что меняется при документации v1→v2 внутри P? **Ответ:** downstream prefix states обычно уже не equivalent, нужна invalidation/recompute соответствующей части.
3. Делает ли длинное окно retrieval ненужным? **Ответ:** нет; cost, distractors и evidence quality остаются.

<!-- /unit:5594a1f86325e67a01 -->

<!-- unit:c48540f555be7613a8 -->
<a id="u-c48540f555be7613a8"></a>

**Практика:** выписать три prompts P+Q1/2/3 и отметить maximal common token prefix. Затем добавить timestamp в начало, в конец и только в metadata: объяснить различие potential cache reuse, не делая ложного обещания конкретного provider behavior. Оценить answer quality при переносе evidence по positions отдельно от speed.

<!-- /unit:c48540f555be7613a8 -->

<!-- unit:772f398c43178275c4 -->
<a id="u-772f398c43178275c4"></a>

**Схема курса:** request + evidence → context budget/position policy → full window or sliding window → cache-key check → reused prefix or recompute → position-controlled evaluation

<!-- /unit:772f398c43178275c4 -->

<!-- unit:aa17137ed195c0026a -->
<a id="u-aa17137ed195c0026a"></a>

<!-- source-depth-inference-Ж -->

<!-- /unit:aa17137ed195c0026a -->

## Результат главы и переход дальше

Ответьте на вопрос: **Что меняется при ограничении истории и повторном использовании prefix?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
