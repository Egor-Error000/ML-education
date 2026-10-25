# Кэш ответов, monitoring и качество

**Уровень:** основа ветки.

**Предпосылки:** [От prompt к обычному циклу генерации](<infer-generation.md>), [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>), [Serving, batching и PagedAttention](<infer-serving.md>)

**Используют:** Завершение выбранной ветки: самостоятельный эксперимент по практике этой главы.

**Учебный вопрос:** Как доказать допустимость reuse и заметить регрессию качества?

## Карта вычисления и границы этапов

![Кэш ответов, monitoring и качество: входы, преобразования и выходы](<../../оформление/course/infer-monitoring.svg>)

Ключ кеша должен учитывать все данные, влияющие на результат. Мониторинг связывает качество и ресурсы с версией модели и конфигурацией, иначе регрессии нельзя локализовать.

## Какой именно результат сохраняется

KV/prefix (уже доступный префикс) cache (кеш повторно используемых вычислений) хранит intermediate states, ускоряя повторный compute. Answer cache хранит final response/structured result и может вообще пропустить generation. Tool-result cache хранит результат внешней операции, например snapshot документа. Semantic cache ищет похожий query (запрос), чтобы переиспользовать answer/result по принятой policy.

Эти уровни имеют разные validity conditions. Одно и то же prefix P с разными вопросами допускает KV reuse, но ответы различаются. Одинаковый вопрос с новыми underlying data может требовать нового ответа даже при identical text. Одинаковый tool (инструмент приложения) query в другом authorized scope не должен возвращать чужие данные.



### Возвращаемся к трём вопросам из разбираемого примера о prompt (входной запрос или инструкция) caching

Рассмотрим общий контекст, который содержит правила и документацию проекта, а вопросы менялись: о circuit breaker, о портах payment-service и подключении из order-service, затем о secrets. Повторяемый prefix позволяет переиспользовать часть **вычисления** при поддержке model runtime (среда исполнения модели)/provider. Он не делает три ответа одинаковыми. Сохранить первый готовый ответ и вернуть его на второй вопрос было бы ошибкой даже при идеальном prefix hit.

Наоборот, два полностью одинаковых вопроса могут требовать разных ответов, если обновилась документация или изменился разрешённый scope. Поэтому cache выбирается по тому, какой объект сохраняется. У prefix cache проверяется совместимость model computation, у answer cache — допустимость готового результата в текущей прикладной ситуации. Эти проверки связаны, но не совпадают.

Tool-result cache занимает промежуточное место. Сохранённый snapshot документа может быть переиспользован при той же версии и правах, а новый вопрос всё равно потребует другого чтения/ответа. Сохранённый result действия не заменяет исполнение нового действия. Строка «готово» в cache не доказывает, что требуемая операция произошла для нынешнего request.

### Cache меняет не только время, но иногда и распределение ответов

Для stochastic generation одинаковый prompt может давать разные допустимые формулировки. Answer cache фиксирует один ранее полученный output. Если задача требует независимые samples, это изменение semantics. В учебной двухответной модели P(A)=0.7: у десяти независимых вызовов вероятность «все A» равна $0.7^{10}\approx0.028248$, а при одном первом sample и девяти cache returns —0.7. Marginal первого ответа тот же, совместное поведение другое.

Для FAQ несколько корректных формулировок могут быть взаимозаменяемыми по policy, и возврат первой допустим. Для генерации разнообразных идей это уже не та же задача. Включить temperature (температура распределения выбора)/seed (начальное состояние генератора случайных чисел) в key полезно только в рамках выбранного contract; одинаковый seed сам по себе не обещает bitwise воспроизводимость всех backend execution paths.

Если первый сохранённый ответ ошибочен, ошибка способна повториться на множестве hits. Это делает provenance и проверку сохранённого результата особенно важными. Проверить один cached answer1000 раз —не то же самое, что оценить 1000 независимо сгенерированных ответов.

## Exact cache key и computation identity

Пример key components: normalized input, user/tenant или permission scope, model/adapter revision, system/template (шаблон сериализации входа) version, decoding policy, corpus/index snapshot, tool result versions и relevant locale/time assumptions. Это не универсальная обязательная структура: включается всё, что меняет допустимый результат выбранного cache.

Пусть запрос «какая версия API?» вчера отвечался по docs v1, сегодня docs v2. Key только по строке вернёт stale answer. Если answer зависит от current date или account plan, эти dependencies должны быть versioned/invalidated либо query не следует cache таким способом.

Canonicalization помогает exact hits: упорядочить незначимые whitespace/JSON keys, если это действительно сохраняет semantics. Но нельзя удалять отрицания, numbers, entity IDs или порядок instructions ради hit rate. «Платёж разрешён» и «платёж не разрешён» отличаются одним словом и противоположным смыслом.

TTL ограничивает возраст записи, но не доказывает correctness до expiration: данные могут измениться через секунду. Event/version invalidation точнее привязывает cache к изменениям. Нужны policy deletion/revocation и handling entries с ошибками; temporary tool timeout не должен стать долгим cached «ничего не найдено» без осознанного negative-cache TTL.

### Ключ описывает выбранную эквивалентность, а hash только кодирует её

Представим функцию ответа $y=F(x;\theta,p,d,a)$: x —input, θ —model/adapter revision, p —prompt/decoding policy, d —данные/tools snapshot, a —authorization/locale и другие существенные условия. Для exact application cache выбирают, какие входы считаются взаимозаменяемыми по нужной задаче. Затем детерминированно сериализуют соответствующие поля и получают digest.

```text
schema_version
task_type + normalized_input
model_revision + adapter_revision
template_revision + decoding_policy
data_snapshot + tool_dependencies
permission_scope_revision + relevant_locale/time_policy
```

Это учебная схема, а не требование включать все поля всегда. Если ответ не зависит от locale, поле не добавляет смысла. Если зависит от текущего состояния документа, игнорировать его revision нельзя. Hash не исправляет пропущенную dependency: два разных допустимых результата попадут под один и тот же намеренно неполный key.

Сериализация должна сохранять границы полей. Простая конкатенация «ab»+«c» и «a»+«bc» даёт одну строку «abc». Structured encoding или длины полей устраняют эту неоднозначность до hashing. Равный digest затем всё равно используется в контексте выбранного storage/collision protocol; длинный hash не доказывает правильность semantics.

Canonicalization тоже условна. Для application function над JSON object порядок независимых keys может быть несущественным. Для literal model prompt перестановка текста меняет token (токен) sequence, даже если человек считает содержание похожим. Удаление whitespace из кода или списков может менять смысл. Поэтому normalize следует только те различия, которые объявлены несущественными для результата, а не всё, что повышает hit rate.

### TTL, eviction и dependency invalidation — три разные причины удаления

TTL ограничивает срок жизни записи по принятой политике. Eviction освобождает место, например из-за memory budget. Invalidation говорит, что запись больше не соответствует dependency или policy. Запись может быть свежей по TTL, но invalid после изменения документа; может быть valid, но вытеснена из памяти; может истечь по времени при неизменных данных.

Пусть запись создана в t=0 с TTL60s, а документ изменился в t=1. Без иной проверки cache способен возвращать старый ответ до expiration. В t=30 запись моложе TTL, но это не доказывает её актуальность. Для snapshot-based задачи ответ по старой версии может быть корректен, если именно она запрошена; для «последней версии» —другой contract.

Negative cache должен различать подтверждённый no-results и временный timeout. Если search вернул empty set для конкретного corpus snapshot, это один результат. Если network failure превращён в такое же `[]`, приложение потеряло причину отсутствия evidence. Долгий cache такого значения закрепит ошибку как будто это факт «документов нет». Status, completeness и зависимости сохраняют вместе с payload.

Stale-while-revalidate возвращает прежнюю запись, пока новое вычисление идёт фоном. Это может быть допустимая freshness policy, но не автоматическое ускорение без изменения semantics. Нужен предел допустимого возраста и правило для dependencies, где stale result неприемлем. Expired по TTL и запрещённый текущим scope result нельзя смешивать в один «немного старый ответ».

### Гонка «удалили старое, затем старое вернулось»

В t0 worker A читает docs v7 и начинает generation. В t1 публикуется v8, cache key только по вопросу удаляется. В t2 A завершает вычисление по v7 и снова записывает старый answer под тем же key. Однократное delete не предотвратило stale resurrection.

Версионный key привязывает answer к snapshot, который реально использован: результат v7 записывается как v7 и не становится результатом v8. Нельзя в момент публикации просто взять «текущую» revision v8 и присвоить её ответу, созданному из v7. При policy «отвечать по самой новой версии» дополнительно определяют, что делать с изменением dependencies во время активного request: повторить, вернуть явно указанный snapshot или иной согласованный путь.

Общий учебный протокол: зафиксировать dependency identity → искать соответствующий key → проверить допустимость использования → на miss вычислить по этому snapshot → опубликовать с тем же identity по выбранному concurrency protocol. Между разными storage systems простая проверка перед write не создаёт универсальную атомарную transaction. Требуемая consistency и revocation semantics должны быть свойством всей схемы, а не обещанием cache helper.

При изменении прав key со старым permission revision может физически остаться, но не должен считаться разрешением на выдачу данных. Актуальная authorization policy проверяется в нужной точке read/return. Сам факт, что ответ когда-то был виден одному пользователю, не делает его доступным другому.

## Semantic similarity не равна взаимозаменяемости ответа

Dense embedding (векторное представление) может сблизить «удали черновик 42» и «покажи черновик 42», «лимит 100» и «лимит 1000». Для retrieval (поиск кандидатов) это похожие темы, для answer reuse — разные задачи. Threshold cosine0.95 не является доказательством одинакового смысла или authorization.

Учебный safe pattern для read-only FAQ: filter by scope/version/task type → exact entity/number/negation checks → semantic candidates → separate acceptance rule → reuse verified answer с provenance, либо обычный generation. Для действий с side effects final answer cache не означает, что действие уже выполнено в новом контексте.

Оценивать semantic-cache precision: доля accepted hits, где reused answer допустим; accepted hit rate — какую долю requests обслужили reuse; recall допустимых возможностей reuse требует отдельной разметки; latency (задержка отдельного запроса)/cost и stale/error rate. Если высокая цена false hit, лучше меньший hit rate. Human audit должен включать near-threshold и counterexample queries, а не только очевидные synonyms.

### Candidate retrieval ещё не решение reuse

Semantic search находит близкие сохранённые queries. Это этап предложения кандидата, аналог first-stage retrieval. Acceptance проверяет, можно ли вернуть **этот** answer для текущего query. Вопросы «включить feature (признак) X?» и «как выключить feature X?» могут быть тематически близки, но намерение различается. Даже совпадение entity X не устраняет противоположный intent.

Проверки чисел, IDs, отрицаний и task type полезны как части policy. Они не исчерпывают смысл: одинаковые слова могут относиться к разным условиям или версиям. Если используется дополнительный judge equivalence, его false accepts/rejects измеряют отдельно. Высокий cosine не становится вероятностью валидности answer только потому, что выбран красивый threshold0.95.

Полезно различать два свойства: query-answer compatibility и correctness самого сохранённого answer. Два queries могут быть эквивалентны, но исходный answer неверен. Тогда reuse повторяет ту же ошибку. Проверять лишь перефразирование вопроса недостаточно для end-to-end quality.

### Similarity threshold не создаёт отношение эквивалентности

Возьмём unit vectors под углами 0°,10°,20°. Cosine соседей≈0.984808, крайних≈0.939693. При threshold0.98 A близок B, B близок C, A не близок C. Это искусственная геометрическая иллюстрация, не измерение реального encoder (кодировщик входа). Если объявлять новый accepted query представителем следующего и строить цепочку aliases, можно постепенно уйти от исходного вопроса/answer.

Поэтому cache equivalence нельзя выводить из connected components threshold graph без дополнительной модели задачи. Сравнение с конкретной исходной записью и её dependencies отличается от транзитивного расширения класса «похожих». Та же проблема обсуждалась для duplicates в BERT-разделе; здесь ошибка приводит к reuse неправильного output.

### Четыре denominators вместо одного hit rate

Пусть из 1000 requests semantic lookup нашёл candidates для 600, acceptance пропустил 500, из них 495 reused answers признаны допустимыми. Lookup candidate rate=0.6, accepted hit rate=0.5, accepted-hit precision=495/500=0.99. Пять false hits —1% accepted hits и 0.5% всех requests. Эти два процента нельзя незаметно подменять друг другом.

Recall reuse определяется относительно **всех действительно допустимых возможностей reuse**, а не числа найденных candidates. Он требует соответствующей разметки или sampling (случайный выбор по распределению) evaluation. По одному log accepted/rejected нельзя узнать, сколько полезных candidates поиск не нашёл. Также lookup miss, rejection, stale dependency и wrong answer —разные причины fallback.

Threshold выбирают на разрешённой validation (проверочная часть для выбора решения) выборке с counterexamples, затем фиксируют. Если каждый найденный false accept исправлять и сразу считать по тем же cases итоговый score, получается regression training set, а не независимая оценка. Нужны и известные трудные пары, и свежая representative sample.

## Когда caching вообще выгоден

Пусть cache lookup5ms, обычный answer1000ms, accepted hit rate h, hit validation ещё 5ms. Упрощённое expected time $5+h\cdot 5+\left(1-h\right)\cdot 1000$. При h0.5 получим 507.5ms; при h0 —1005ms, cache только добавил overhead. При false hits скорость не считается успешным решением: quality constraint первична.

Аналогично cost включает lookup/vector storage/validation/maintenance. Semantic judge на каждый query может съесть значительную часть экономии. Сначала измерить actual repetition и sensitivity to stale results; затем сравнивать exact cache, semantic cache и no cache на одном workload.

Stampede: много simultaneous misses одного expensive key создают повторную генерацию. Single-flight/request coalescing может объединить работу, если inputs/scope equivalent. Cancellation одного клиента не должна неправильно отменить результат, нужный другим. Distributed invalidation/concurrency требуют своей semantics.

### Вывод break-even с явно заданными затратами

В исходной простой модели lookup L=5ms, hit validation V=5ms, fresh answer T=1000ms, accepted hit fraction h. Среднее $E=L+hV+(1-h)T$. Cache быстрее baseline (базовый вариант сравнения) T, если

$$
h(T-V)>L,\qquad h>\frac{L}{T-V}\quad\text{при }T>V.
$$

Для этих чисел порог h≈0.005025, то есть около 0.503%. Это вывод лишь для объявленной модели с validation cost на accepted hits и одинаковой fresh cost. При h=0.5 получается 507.5ms; при h=0 cache добавляет 5ms. Реальные rejected candidates также могут требовать дорогую проверку.

Пусть candidate находится с вероятностью c, принимается с условной вероятностью a, так что h=ca. Если validation V оплачивается каждому candidate, включая rejected, то $E=L+cV+(1-ca)T$. При c=0.6,a=0.8,V=50ms,L=5ms,T=1000ms получаем 555ms. Игнорирование rejected validations занизило бы стоимость.

Если fresh miss queries в среднем длиннее cached queries, один общий T уже недостаточен. Раздельно оценивают costs по slices и по тому же request workload. Provider token accounting, storage, embedding lookup, judge и maintenance добавляют свои единицы. Учебные milliseconds не являются measurements конкретного API или железа.

### Request hit rate не равен доле сэкономленных tokens (токены)

Из 1000 requests800 коротких cache hits по 100 prompt tokens,200 misses по 5000. Если сравнить с no-cache на том же workload, потенциально пропущено 80000 из 1080000 prompt tokens, около 7.41%, хотя request hit rate80%. Output tokens и другие charges пока вообще не учтены. Поэтому «80% hits» не означает автоматически «80% меньше счёт».

И tail latency может вести себя иначе, чем mean. Для 90 hits по 10ms и 10 misses по 2000ms mean209ms, но nearest-rank p95=2000ms. Средний ответ стал быстрым, а выбранный tail percentile всё ещё определяется misses. Эти числа —учебная fixed sample с объявленным percentile convention.

### Single-flight объединяет работу, а не превращает все misses в готовые hits

Пусть 20 equivalent requests одновременно пришли на пустой cache. Без coordination все могут начать generation. Single-flight выбирает одного producer, остальные ждут общий result. Число upstream generations может уменьшиться 20→1, то есть на 95%, но initial ready-cache hit rate остаётся 0: запись ещё не существовала. Coalesced waiters следует учитывать отдельной metric.

Это не обещает ускорить latency каждого request в 20 раз: параллельные вычисления могли батчиться, а waiters всё равно ждут producer. Выгода зависит от ресурса и workload. Отмена одного waiter не обязана отменять computation, которое нужно другим. Future завершается успехом или error; in-flight registry очищается по всем terminal paths, иначе последующие requests могут ждать навсегда.

В distributed lease варианте owner A может задержаться дольше TTL, после чего lease получит B. Поздний A не должен безусловно удалить lock B. Сравнение owner token при release объяснено в [Redis distributed-locks pattern](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/). Но защита unlock не делает любой поздний write корректным: publication результата требует своего generation/version protocol. Random ownership token и упорядоченная версия записи решают разные задачи.

### Storage budget и eviction тоже входят в эксперимент

Для 100000 entries с answer payload2KiB только тексты занимают 204800000 bytes. Query vectors ширины 768 float32 добавят 307200000 bytes, даже без index overhead. Если key digest32 bytes и metadata64 bytes на entry, эти компоненты добавят ещё 3.2 и 6.4 MB decimal. Сумма перечисленного —521.6 MB до allocator/index/replicas и хранения дополнительных dependencies.

Число requests не равно числу entries: много repeats одного key может сохранять одну запись, а почти все уникальные вопросы быстро увеличивают footprint при низком reuse. TTL регулирует age, memory eviction —место, semantic acceptance —правильность. Одной настройкой TTL нельзя оптимизировать все три свойства сразу.

## Мониторинг технического состояния и качества

Operational metrics: requests, queue age, TTFT/inter-token/end-to-end p50/p95, timeout/error/OOM, input/output lengths, cache hit/reject/stale rates, version distribution и cost. Их следует привязывать к workload slices: language, input length, tenant/task type, retrieved source quality.

Quality metrics: task correctness, schema (схема допустимых данных) validity, wrong-tool/args, retrieval relevance, evidence/citation support, false abstention/refusal и successful completion. Ни latency, ни user thumbs-up в одиночку не доказывают correctness. Feedback имеет selection bias: довольные/недовольные пользователи оставляют оценки с разной вероятностью.

Delayed labels (метки) требуют оценивать quality после появления результата, сохраняя связь с version/time cohort. Drift (изменение распределения или зависимости со временем) input embeddings (векторные представления)/length/language может быть ранним indicator, но не равен доказанному падению task quality. Concept drift может ухудшить correctness без большого сдвига внешней формы запросов.

### Связать quality с конкретной траекторией и версиями

Для разбора request нужны IDs и версии: model/adapter, template, corpus/index, cache policy, selected entry/dependencies, tools, evaluator. У cache hit может вообще не быть нового model forward (прямой проход вычисления); если считать только model spans, такие ответы исчезнут из end-to-end статистики. У agent request может быть несколько model calls и tool waits; TTFT каждого call и время до первого пользовательского ответа —разные величины.

Monitoring counters должны сообщать, что именно считают. Candidates, accepted hits, coalesced waiters, expired entries, invalidated entries и fresh generations различаются. Метрика «cache hit» без определения способна вырасти после переименования waiters, хотя фактическое поведение не изменилось. Аналогично costs привязывают к token categories конкретного execution contract, не объявляя любой cache read бесплатным.

### У generative задачи могут быть проверяемые истины без единственной формулировки

План говорит об отсутствии одного правильного ответа в generative системах. Это полезное отличие от одного class ID, но не отсутствие проверяемых требований вообще. «Timeout равен 5s» и «Ожидание прекращается через 5 секунд» могут быть двумя корректными формулировками. «50 секунд» лексически похоже, но неверно относительно тех же docs.

Для extraction можно проверить schema, IDs и значения; для code —заявленные tests, помня об их coverage; для summary —faithfulness и покрытие существенных фактов по rubric. JSON validity не доказывает истинность поля. Наличие citation не доказывает, что именно этот source поддерживает claim. Поэтому качество оценивают по отдельным наблюдаемым свойствам вместо одного расплывчатого «ответ понравился».

### Почему сравнивать hits и misses напрямую нечестно

Представим workload пополам: frequent easy queries и rare hard. No-cache quality на них 99% и 90%, общий результат 94.5%. Cache возвращает easy answers с quality98%, misses на hard остаются 90%; итог 94.0%. Если сравнить только общий cached score94% с miss score90%, кажется, что cache улучшил качество. На том же workload он ухудшил его на 0.5 процентного пункта.

Нужен counterfactual no-cache baseline на сопоставимых queries, а не сравнение разных по трудности подмножеств. Shadow generation на sample cache hits может дать такую проверку, но тратит compute и имеет собственный protocol. Changed-data и permission cases проверяют отдельно: обычные synonyms не измеряют stale reuse.

### Повторяемая ошибка и delayed ground truth (эталонная разметка)

Если один cached answer возвращён 1000 раз, его оценка 1000 пользователями создаёт зависимые observations. Для крайних toy условий пусть вероятность неверного первого answer p=0.01. У 1000 independent generations число ошибок имеет variance $1000p(1-p)=9.9$. При одном first sample, повторённом 1000 раз, variance числа ошибок $1000^2p(1-p)=9900$ при том же expected count10. Cache усиливает корреляцию исхода, а не обязательно меняет его marginal expectation.

Поэтому audit может учитывать request-weighted impact и отдельно unique answer/cache-key groups. Оба нужны: одна ошибка, показанная 1000 раз, имеет большой пользовательский эффект, но не является 1000 независимыми доказательствами качества генератора. Similarity drift и user feedback не заменяют такую оценку.

Ground truth тоже может приходить позже. Для ответа, созданного по docs v7, важно различать correctness относительно тогдашнего snapshot и freshness относительно уже опубликованной v8. Сохраняемый cohort/version context (доступный модели контекст) позволяет не объявить исторически правильный ответ ошибкой только из-за смены evaluator data, и одновременно увидеть реальные stale returns после обновления.

## Regression suite и judge

Фиксированный curated набор ловит известные failures; свежая случайная sample production показывает новые. Оба нужны: static benchmark может загрязняться tuning, а только fresh sample плохо воспроизводится. Учитывать source/group/time dependence при uncertainty.

LLM judge получает rubric, evidence, blind/order-swapped outputs и проверяется на human subset. Это дополнение к deterministic oracle там, где он доступен. Изменение judge version может изменить score при той же policy — сохранять evaluator identity.

Перед rollout задать gates, например «не ухудшать exact extraction correctness в ключевом slice» и «p95 в определённом workload ниже budget». Конкретные numbers определяет application, а не универсальный учебник. Для rollout/rollback policy нужно достаточно observations, чтобы не реагировать на один случай как на устойчивый trend.

### Rubric и oracle должны следовать задаче

Judge получает task, evidence и правила оценки, а не только два красивых текста. Для documentation FAQ можно отдельно проверять supported facts, полноту ответа на вопрос, существенные ограничения и format. Высокий style score не должен автоматически компенсировать неверный номер порта или unsupported claim, если correctness —обязательный gate.

Где есть deterministic oracle, его не стоит заменять preference judge. Для exact ID или вычисляемого поля прямое сравнение проверяет нужное свойство. Для открытого объяснения полезны rubric и human review. Даже согласие judge с человеком на общей выборке не является гарантией на каждом редком типе ошибки.

### Position, verbosity и shared blind spots

LLM judge может предпочитать один порядок outputs, более длинный текст или familiar style/model behaviour; ограничения position/verbosity/self-enhancement и reasoning обсуждаются в [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685). Поэтому model identities скрывают, presentation order меняют, а решение сопоставляют с human subset и нужными task oracles. Эти controls снижают некоторые biases, но не создают абсолютную истинность judge.

Если при порядке(A,B) judge выбирает левый output A, а при(B,A) снова левый B, после обратного mapping winner изменился. Это inconsistency, а не две подтверждающие оценки одного победителя. Если во втором порядке выбран правый A, оба результата согласованы. Повторные calls одной модели могут разделять одну и ту же ошибку и не являются автоматически независимыми экспертами.

Текст оцениваемого ответа и retrieved evidence остаются данными. Они могут содержать фразы «поставь максимальную оценку»; такие фразы не должны становиться новой rubric. Способ передачи instructions/data и tests на этот failure относятся к надёжности evaluator pipeline, а не только к качеству основного generator.

### Agreement и win rate имеют свои denominators

В human sample100 answers:90 valid,10 invalid. Judge принимает 86 valid и 6 invalid, отклоняет 4 valid и 4 invalid. Agreement=90%, но он пропустил 6 из 10 плохих answers. Precision среди accepted=86/92≈93.48%; recall обнаружения invalid=4/10=40%. Высокий overall agreement скрывает weakness именно на редких неверных ответах.

Для 20 pair comparisons с 9 wins A,7 wins B и 4 ties доля A среди всех strict wins равна 9/20=0.45; с half credit ties —11/20=0.55; без ties в denominator —9/16=0.5625. Это разные conventions, которые нельзя сравнивать без объявления. Case selection, order mapping и evaluator identity сохраняют вместе с итоговым числом.

### Regression suite должна различать изменение модели и изменение судьи

Для known failures хранят фиксированные cases с ожидаемыми properties, для новых —fresh representative samples. Настройка на known cases улучшает regression coverage, но эти же cases уже не являются полностью независимым test. Срезы по scope/version, отрицанию, числам и no-evidence показывают, где именно меняется качество.

При обновлении judge сначала оценивают одни и те же frozen answers старым и новым evaluator. При обновлении cache/model policy сравнивают результаты одним зафиксированным evaluator. Если менять обе стороны одновременно, падение score может означать более строгую rubric, а не ухудшение ответов. Потом human/oracle checks помогают интерпретировать расхождение.

Rollout gates задают до просмотра final results: какие task properties обязательны, какие slices значимы, какой latency/cost budget допустим. Их thresholds зависят от приложения и данных. Эта глава объясняет experimental contract; она не подтверждает, что ваш service уже существует или прошёл такой rollout.

## Полный эксперимент и самопроверка

Сравнить no cache, exact cache и один semantic policy: один corpus snapshot/model, фиксированные queries с synonyms, negations, different numbers, permissions и changed-data cases. Измерить correctness reused outputs, accepted hit rate, latency/cost. Затем invalidation по document (документ) version и permission revoke.

1. Почему TTL не гарантия freshness? **Ответ:** underlying state (состояние системы) может измениться до expiration.
2. Почему cosine similarity недостаточно? **Ответ:** тематическая близость не доказывает equality entities/constraints/intent.
3. Можно ли alarm на embedding drift считать quality regression? **Ответ:** это indicator; нужен task-specific evidence.
4. Чем application answer cache отличается от prompt cache? **Ответ:** возвращает готовый output вместо reuse intermediate prefix computation.

5. Сохраняет ли answer cache независимость stochastic samples? **Ответ:** нет. Повторно возвращается один sample; вероятность совместных событий и корреляция ошибок меняются.
6. Исправит ли длинный hash пропущенную data revision? **Ответ:** нет. Он кодирует переданные fields; пропущенная dependency остаётся пропущенной.
7. Почему deletion при обновлении недостаточно? **Ответ:** старый producer может завершиться позже и снова записать stale output под тем же key. Нужен version/publication protocol.
8. Равен ли candidate rate accepted hit rate? **Ответ:** нет. Candidate может быть rejected, stale или несовместимым по scope; accepted-hit precision имеет ещё другой denominator.
9. Почему 80% request hits не означают 80% экономии tokens? **Ответ:** hits могут быть короткими, misses длинными; экономия взвешивается фактическими единицами работы и затрат.
10. Делает ли single-flight все waiting requests ready-cache hits? **Ответ:** нет. Они ждут ещё не готовое общее вычисление; coalescing учитывают отдельно.
11. Достаточно ли 90% judge agreement для проверки редких неверных ответов? **Ответ:** нет. Нужно смотреть confusion по важным outcomes; в примере обнаружено лишь 40% invalid answers.
12. Можно ли сравнить quality hits и misses как две равные группы? **Ответ:** без контроля difficulty и selection —нет. Нужен сопоставимый no-cache baseline и отдельные slices.

**Практика:** составить cache policy для documentation FAQ с key dependencies и forbidden-to-reuse cases. Проверить десять пар запросов, включая «включить/выключить», разные IDs и даты. Объяснить, где hit допустим и какое evidence нужно для этого решения.

## Результат главы и переход дальше

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
