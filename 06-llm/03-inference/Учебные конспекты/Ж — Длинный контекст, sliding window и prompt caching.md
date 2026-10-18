# Ж. Длинный контекст и повторное использование prefix

[Оглавление](README.md). Пункт C.4; KV memory из A.6 и отличие от E.3.

**Основа:** S5, строки 417–1034. Сохранён вопрос о prompt/context caching «на примере стандартного проекта и нескольких запросов». **Исправления C14/C22:** возможности/TTL/тарифы провайдера не вечные факты, cached prefix не готовый answer, последующий query всё равно взаимодействует с prefix. **Углубление:** exact prefix examples, cache boundaries и оценка long-context quality.

## 1. Длина окна и полезное использование — разные свойства

Maximum context определяет supported input/total token limits по runtime/model contract. Оно не доказывает, что модель одинаково хорошо извлекает evidence из любой позиции и любой длины. Ограничения на input/output budgets и chat template special tokens тоже входят в фактическую длину.

Проверять task quality при разном положении evidence: начало, середина, конец; добавлять distractors и unanswerable queries. «Lost in the middle» — описанное в исследованиях поведение использования длинного context, а не гарантированный дефект каждой model. [Исследование long-context retrieval](https://arxiv.org/abs/2307.03172).

Новый positional scaling/extension method может позволить обработать более длинный input, но его accuracy требует проверки. Успешный allocation512k tokens не доказывает точность reasoning по 512k. Chunking/retrieval и selection evidence могут быть полезнее передачи всего корпуса в каждый prompt.

### Исправляем размер cache из исходного C.4

В S5 для 80layers,8KVheads,head_dim128 и двухбайтовых K/V было получено 640KB/token. Прямое умножение даёт 2·80·8·128·2=327680bytes=320KiB/token: исходный результат завышен вдвое. При 8K=8192tokens получаем 2.5GiB, при 128K=131072tokens —40GiB. Для десяти таких sequences —400GiB до allocation/scale/workspace overhead. В decimal GB числа будут другими; нельзя смешивать KB/GB с KiB/GiB в одном расчёте.

Weight storage и KV storage независимы. Quantized weights той же модели могут занимать меньше места, но каждый resident long-context request добавляет свой cache, если нет поддерживаемого sharing. Наоборот, общий большой prefix нескольких requests иногда хранится совместно. Поэтому примерB10×T128K —верхняя оценка без такого sharing, а не обязательная memory цифра любого service с десятью зарегистрированными пользователями.

У full dense attention полный prefill имеет quadratic число взаимодействий по lengthN, а KV —linear число сохранённых vectors. На decode новая query взаимодействует с доступной history, поэтому его attention cost растёт примерно линейно по N, при прочих фиксированных dimensions. Формула «context вырос вдвое, весь запрос стал ровно вчетверо дороже» неверна для полной системы: linear projections/MLP, memory traffic, output length, batching и kernels также участвуют.

### RoPE: почему имеет значение расстояние

В одной двумерной паре RoPE вращает q и k на углы iθ и jθ. Для стандартной rotation matrix R:

$$
(R(i\theta)q)^\top(R(j\theta)k)=q^\top R((j-i)\theta)k.
$$

В scalar product возникает **relative displacement**. В полной head используются разные частоты для разных пар; сами q/k зависят от content и прежних layers. Поэтому identity не означает, что attention зависит только от расстояния и игнорирует текст. [RoFormer/RoPE](https://arxiv.org/abs/2104.09864).

При position interpolation p'=p/s переход из training window8K к planned32K даётs4: позиция 32000 отображается в 8000, соседние positions отличаются на 0.25 вместо 1. Это изменение фаз/разрешения, а не буквальное слияние всех соседних positions в одинаковый float. Интерполяция удерживает индексы в знакомом диапазоне, но меняет geometric relation, под которую weights учились. [Position Interpolation](https://arxiv.org/abs/2306.15595).

[YaRN](https://arxiv.org/abs/2309.00071) и NTK-inspired подходы меняют frequency/scale policy, пытаясь сохранить полезные расстояния на разных масштабах. Они не дают универсального рецепта «любой 8K checkpoint без обучения станет 32K, с LoRA станет 128K». Нужны actual training/extension recipe и evaluation коротких **и** длинных задач. Изменение RoPE parameters также меняет computation identity cache; прежние K/V не считаются совместимыми автоматически.

## 2. Sliding window меняет доступ к истории

В локальном attention query читает ограниченные прошлые positions. Для window W стоимость взаимодействий и retained cache можно ограничить по architecture/runtime. Но данные вне window не доступны напрямую этому query, если отдельный mechanism не сохраняет/передаёт их.

Пример: W4, current position10 имеет разрешённые previous/current positions по конкретной convention, например 7–10. Факт на позиции 2 нельзя заново прочитать обычным local attention10. Более глубокие layers могут переносить информацию косвенно, но это не equivalent full history access.

Нельзя просто обрезать cache full-attention model и назвать её «бесплатно работающей с любым long context». Это меняет computation; модель могла не обучаться такой policy. Quantized KV уменьшает bytes, сохраняя больше positions с ошибкой; sliding window удаляет dependencies. Эти trade-offs разные.

### Receptive field не равен гарантированному воспоминанию

Зафиксируем convention: окно W включает current token иW−1 предыдущих. ДляLlayers с таким attention graph максимальный путь по input positions может охватывать 1+L(W−1)tokens. ПриW4,L3 это 10positions. Это теоретическая достижимость через цепочку intermediate representations для фиксированного input, а не обещание сохранить каждый факт из этих десяти positions без потерь.

Поэтому произведениеW×L из S5 не устанавливает supported context length модели и не эквивалентно full attention. Повторно прочитать точный старый source span и использовать косвенно перенесённую feature —разные возможности. Смешанные local/global layers, global tokens, dilated или block-sparse patterns дают другой graph. Каждый из них оценивают с actual mask, а не подставляют одинW для любой architecture.

Если все layers допускают sliding cache с окномW, retained attention state на sequence может ограничиваться этим окном по layer. Но arbitrary cache truncation full-attention checkpoint меняет функцию. Quantization cache сохраняет больше history positions с численной ошибкой, eviction удаляет часть edges, summary создаёт новый текст. Эти способы сокращения затрат нельзя называть одним lossless context compression.

### FlashAttention и Ring Attention решают другие части задачи

[FlashAttention](https://arxiv.org/abs/2205.14135) организует tiled exact attention и работу с промежуточными sums/maxima так, чтобы не хранить полнуюN×Nmatrix в HBM. При фиксированных прочих dimensions auxiliary storage может расти линейно поN, хотя arithmetic full attention остаётся quadratic. Это не превращение trained full-attention model в sliding-window model. Стабильная global softmax normalization при работе по tiles связана с формулой предыдущей главы serving.

[Ring Attention](https://arxiv.org/abs/2310.01889) распределяет sequence blocks между devices и организует обмен K/V с вычислением attention. Это помогает разместить и исполнять длинную sequence при suitable interconnect и реализации. Но добавление GPU не делает полный объём pair interactions линейным поN на всю систему. Нужно различать throughput, latency, суммарные FLOPs, память одного device и общий memory budget. Слово«почтилинейное» без указания оси scaling ничего не доказывает.

Mamba-подобная recurrent state архитектура хранит другую representation истории; это не runtime option, включаемая у любого Transformer. [Mamba](https://arxiv.org/abs/2312.00752) и hybrid[Jamba](https://arxiv.org/abs/2403.19887) иллюстрируют иные trade-offs. В S5 Jamba ошибочно приписывалась Mistral; работа принадлежитAI21. Если hybrid сохраняет некоторое число full-attention layers, у этой части остаётся quadratic full-prefill component. Обещание точного recall из любой длины не следует только из названияlinear-time.

## 3. Prompt caching на понятном проекте

Пусть документационный assistant получает:

```text
P = system rules + fixed tool definitions + документация версии v1
Q1 = вопрос об авторизации
Q2 = вопрос о формате ошибки
Q3 = вопрос о таймаутах.
```

Запрос 1 обрабатывает $P+Q_{1}$, создавая K/V prefix. Запрос 2 при exact совпадении P и compatible context может reuse его states и вычислить новый suffix Q2; затем генерирует новый answer. Запрос 3 аналогично. Ни answer1, ни observation от инструмента 1 автоматически не являются answer2.

Общий prefix должен быть **одинаковым в токенах и порядке**. Семантически сходная переписанная документация не идентичный prefix. Если случайный request ID поставлен в самом начале, совпадение может оборваться раньше большой документации. Если stable instructions/docs стоят раньше переменного вопроса, reusable portion больше.

При этом prompt ordering должен сохранять смысл/security/task quality: нельзя ради cache переносить важные context distinctions туда, где application их потеряет. Provider может иметь explicit cache objects или automatic prefix matching; granularity/min length/TTL/cache keys/charging зависят от актуального API/runtime.

Этот учебный пример описывает механизм без копирования устаревающего vendor snippet из чата. При реализации измерить reported cached tokens/hit rate и latency, сверив текущий contract. [Пример prefix cache design vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/).

### Возвращаем исходный проект TechCorp

Ваш вопрос просил не только определение caching, но и несколько запросов в стандартном проекте. В S5 assistant работал с user-service,order-service,payment-service, их портами и общей документацией. Стабильные instructions занимали условно 8000tokens, documents15000; общий P=23000. Каждый отдельный вопрос Q добавлял ещё около 500tokens. Числа учебные: placeholders«здесь ещё 7500tokens» в коде сами не создают такой размер реального input.

Первый вопрос: «Как устроен circuit breaker между сервисами?» Если подходящего cache нет, P и Q1 обрабатываются, states eligible prefix могут быть сохранены. Полученный ответA1 —новая generation, его содержание не является самим prefix cache. Второй вопрос: «Какие порты у payment-service и как подключиться из order-service?» При сохранённом совместимом P вычисление его состояний переиспользуется, но Q2 и новый ответA2 вычисляются заново. Третий вопрос о хранении секретов аналогичен только если cache ещё доступен и версия documents не изменилась.

В примере использовались **независимые** questions с однимP, а не обязательно диалог, где A1/Q1 включены передQ2. Если приложение добавило transcript, token prefix и lengths уже другие. Если жеP повторён, ответA1 не надо механически добавлять в новый запрос ради hit: это меняет evidence и может добавить нерелевантный текст. Сначала определяют intended context, затем находят его reusable часть.

### Изменение начала и изменение хвоста

Для запросов `[rules_v1, tools_v1, docs_v1, Q1]` и `[rules_v1, tools_v1, docs_v1, Q2]` общий prefix заканчивается перед различающимся question. При `[rules_v1, tools_v1, docs_v2, Q3]` earlier stable rules/tools ещё могут совпасть, но после изменившегося документа старые downstream states уже не эквивалентны. Это не обязательно полный miss от первого token.

Random request ID в начале может разрушить большой hit; тот же ID только в transport metadata не меняет model tokens, если template его не добавляет. ID в самом конце не разрушает предшествующий exact prefix. Разный порядок documents может менять и cache reuse, и качество ответа: canonical ordering полезен для воспроизводимости, но не повод игнорировать task-relevant ordering. Identity определяется реально поданным input и computation config, а не только визуальным равенством строк в editor.

Source называл обязательным побайтовое совпадение. Для KV принципиальны identical token/embedding prefix и совместимая computation semantics; serialization/template/tool definitions могут входить в это условие. Некоторые APIs дополнительно используют explicit cache IDs, block boundaries, scope или routing. Лишний пробел иногда меняет tokens уже в начале, но универсальное поведение любого provider по одной строке не выводится.

### Что изменяют RAG и персонализация

Динамический RAG не запрещает prefix caching вообще: stable rules/tools до retrieved documents могут повторяться. Разные questions к одному неизменному document bundle дают больший reusable prefix; разные document sets —меньший. Собрать один giant master set ради cache не всегда выгодно: растёт context, появляются distractors, меняется evidence selection. Он также должен соответствовать допустимому контексту пользователя, а не автоматически содержать все внутренние документы.

Персонализация в первом token дробит cache keys, но её перенос после stable rules не делает её неважной для приложения. Доступы и требуемые distinctions сохраняют, исполнение tools проверяет trusted state. Caching экономит вычисление уже **правильного** input, а не определяет, какой input пользователю следует разрешить или какие инструкции модель обязана игнорировать.

Короткий prompt ниже provider threshold может оказаться неeligible. Предложение S5 искусственно «раздуть» его до порога не является общим optimization: дополнительные tokens тоже стоят памяти/compute и могут ухудшить task behavior. Добавлять полезные инструкции можно ради качества, но выгоду padding исключительно ради cache следует доказать расчётом, а не предполагать.

## 4. Почему новые queries всё ещё читают prefix

Saved K/V избавляют от повторного вычисления **states prefix**, но новые query positions всё ещё attend к этим keys/values. Даже при 100% reusable prefix длинная история может стоить bandwidth и decode attention. Cache hit не делает полный запрос вычислительно бесплатным.

Если P20k tokens,Q100tokens, reusable prefix20k, uncached prefill suffix100 — это число новых processed positions, а не 100-token equivalent любой модели: suffix queries имеют доступ к 20k cached keys. Output generation тоже остаётся. Реальные savings оцениваются end-to-end при workload.

Prefix sharing может увеличивать effective concurrency за счёт memory reuse. Но eviction и memory pressure снижают hit rate. Смена model/adapters, positions/template или некоторых runtime settings делает прежний state несовместимым. Cache key должен представлять computation identity, а tenant isolation/authorization задаются отдельным layer.

### Hit не убирает чтение history новым suffix

Для P23000,Q500 full prefill строит states всех 23500positions. При compatible cached P новые Q positions всё ещё читают соответствующие cached keys/values. В dense causal attention их взаимодействия включают порядкаP·Q+Q(Q+1)/2 пар, до dimensions/heads. Для этих чисел это 11625250pairs. Вычисление стало существенно меньше повторного full-prefill attention, но это не то же самое, что model с context только 500tokens.

Новые output tokens тоже обращаются к P и растущему suffix. Поэтому cache hit прежде всего экономит повтор prefix prefill, а не гарантирует ускорения всех decode steps в ту же долю. Memory sharing, transfer/offload и locality могут дать другие эффекты, которые измеряются отдельно. Если cache надо загрузить с другой memory tier, hit имеет свою latency.

### Счётчики и TTL из исходного кода

В S5 были поля cache_creation/cache_read/non-cached input и времена 8.7/2.1/1.9s. Это иллюстративные outputs assistant, не результаты пользовательского запуска. Более того, output length между первыми двумя ответами различалась 340 и 280tokens: одно end-to-end время не выделяет чистый prefill effect. Для такого вывода нужны TTFT/prefill measurements при сопоставимых условиях, включая очередь, network и warmup.

При logging нужно определить, означает ли total input уже сумму cached и uncached components, иначе cached tokens можно посчитать дважды. Request hit rate«хотя бы часть prefix прочитана» отличается от доли всех input tokens, которые были cached. Частичный hit на первые 1000tokens при 23000eligible tokens не равен полному reuse. Creation0/read0 также не доказывает, что новый cache создан: он мог быть disabled, ineligible или не сохранён.

TTL/минимальная длина/cache scope и billing зависят от актуального model/provider contract. Третий вопрос«ещё через 5minutes» из S5 не гарантирует hit на boundary expiry. Нужно знать, от какого события отсчитывается TTL, продлевается ли он, бывают ли eviction/routing misses и explicit control. Один cached request не даёт бессрочной гарантии на следующие. Таблицы старых vendor values не считаются текущей спецификацией.

КлассCachedAssistant сохранял Python strings и HTTP client, но не локальные KV tensors. Connection pooling может уменьшать transport overhead, а provider prefix cache —вычисления модели; это разные механизмы. Новый client не обязан означать miss, старый —не обязан гарантировать hit. `hash(system_prompt + docs)` пригоден разве что как грубый local label: Python hash не обязан совпадать между процессами, concatenation не различает границы blocks, и model/template revision в нём отсутствуют. Для устойчивого monitoring identity нужен canonical manifest и declared digest; server cache identity определяется его собственным contract.

## 5. Context engineering и память приложения

Суммаризация старого диалога даёт новый текст меньшего размера, а не lossless KV compression. Summary может потерять числа, отрицания и provenance. Важно сохранять trusted state — например, выбранный file ID и ограничения — структурированно, а не только пересказом модели.

Retrieval выбирает evidence из внешнего corpus, prefix cache экономит повторение compute, answer cache возвращает готовый result. Их можно сочетать: stable P cached, retrieved snippets/query меняются, final answer при необходимости cached по application policy. Ошибка — смешать эти уровни и reuse ответ только потому, что prefix identical.

### Исправляем арифметику повторных запросов

Для binary модели «полный hit или полный miss» пусть N1000requests, P23000eligible tokens, Q500uncached suffix и hit rate0.9. Тогда 100misses и 900hits дают 2.3Mtokens создания prefix,20.7Mtokens чтения и 0.5Mнового suffix. В S5 одновременно предполагались 90%hit, лишь 3creations и 1000reads —это не один согласованный experiment. Если ровно 3misses из 1000, полный hit rate99.7%, а reads997.

Пусть unit rates prefix write/read/обычной обработки равныc_w,c_r,c_u. Тогда input cost в этой declared модели:

$$
C=N\left(P\big((1-h)c_w+h c_r\big)+Q c_u\right).
$$

Output cost добавляется отдельно. Для безкэшевого случая input=N(P+Q)c_u. Приc_w=1.25c_u,c_r=0.1c_u caching prefix выгоден в ожидании только еслиh>(1.25−1)/(1.25−0.1)≈0.21739. Это собственный toy calculation, а не актуальный тариф. Малое число повторов, частые expiry и платное хранение меняют результат. Если miss оплачивается иначе, formula тоже меняется по contract.

### Quality проверяют независимо от cache economy

В recall experiment сохраняют известный факт и ответ, меняют его position при сопоставимой total length, добавляют distractors и unanswerable cases. Затем отдельно проверяют multi-document reasoning и правильность citations. Найденная needle не доказывает полноценного понимания всей codebase. [Lost in the Middle](https://arxiv.org/abs/2307.03172) описывает empirical pattern, а не математическую неизбежность любой attention architecture.

Для проекта полезны две независимые оси: quality нужного context bundle и reuse/cost того же bundle. Retrieval может повысить evidence quality и уменьшить input, summary —потерять важное отрицание, а длинный cached corpus —быть дешёвым при повторе, но всё равно вводить модель в заблуждение. Проверяют вопросы с точными numbers/версией документа, противоречивыми источниками и отсутствующим ответом. Дальше измеряют cached tokens, TTFT/decode latency и total cost, сохраняя raw outputs для разбора ошибок.

Упражнение из исходного вопроса теперь конкретно: подготовить manifest stableP и триQ, найти maximal common token prefix, затем изменить docs version, место timestamp и порядок retrieval snippets. Предсказать, какая часть потенциально reusable, проверить фактические counters выбранного runtime и отдельно оценить correctness. Не требуется объявлять successful cache hit или deploy без такого наблюдения.

## 6. Вопросы и практика

1. Можно ли hit на одинаковом по смыслу, но другом тексте? **Ответ:** exact KV-prefix reuse требует согласованных tokens; semantic answer cache — другой механизм.
2. Что меняется при документации v1→v2 внутри P? **Ответ:** downstream prefix states обычно уже не equivalent, нужна invalidation/recompute соответствующей части.
3. Делает ли длинное окно retrieval ненужным? **Ответ:** нет; cost, distractors и evidence quality остаются.

**Практика:** выписать три prompts P+Q1/2/3 и отметить maximal common token prefix. Затем добавить timestamp в начало, в конец и только в metadata: объяснить различие potential cache reuse, не делая ложного обещания конкретного provider behavior. Оценить answer quality при переносе evidence по positions отдельно от speed.

<!-- source-depth-inference-Ж -->
