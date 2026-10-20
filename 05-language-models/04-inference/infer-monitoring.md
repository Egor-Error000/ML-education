# Кэш ответов, monitoring и качество

**Уровень:** основа ветки.

**Предпосылки:** [От prompt к обычному циклу генерации](<infer-generation.md>), [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>), [Serving, batching и PagedAttention](<infer-serving.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как доказать допустимость reuse и заметить регрессию качества?





## Карта вычисления и границы этапов

![Кэш ответов, monitoring и качество: входы, преобразования и выходы](<../../оформление/course/infer-monitoring.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:efd55ba56c37b1e9b2 -->
<a id="u-efd55ba56c37b1e9b2"></a>

## Какой именно результат сохраняется

<!-- /unit:efd55ba56c37b1e9b2 -->

<!-- unit:3fb592133125f90249 -->
<a id="u-3fb592133125f90249"></a>

KV/prefix cache хранит intermediate states, ускоряя повторный compute. Answer cache хранит final response/structured result и может вообще пропустить generation. Tool-result cache хранит результат внешней операции, например snapshot документа. Semantic cache ищет похожий query, чтобы переиспользовать answer/result по принятой policy.

<!-- /unit:3fb592133125f90249 -->

<!-- unit:311f873d8cbc0df758 -->
<a id="u-311f873d8cbc0df758"></a>

Эти уровни имеют разные validity conditions. Одно и то же prefix P с разными вопросами допускает KV reuse, но ответы различаются. Одинаковый вопрос с новыми underlying data может требовать нового ответа даже при identical text. Одинаковый tool query в другом authorized scope не должен возвращать чужие данные.

<!-- /unit:311f873d8cbc0df758 -->

<!-- unit:6d2baf750674827cb9 -->
<a id="u-6d2baf750674827cb9"></a>

<!-- source-depth-inference-К -->

<!-- /unit:6d2baf750674827cb9 -->

<!-- unit:e9d8327e985e9c0df7 -->
<a id="u-e9d8327e985e9c0df7"></a>

### Возвращаемся к трём вопросам из разбираемого примера о prompt caching

<!-- /unit:e9d8327e985e9c0df7 -->

<!-- unit:a9e6f6b702289e1bd3 -->
<a id="u-a9e6f6b702289e1bd3"></a>

В S5 один общий context содержал правила и документацию проекта, а вопросы менялись: о circuit breaker, о портах payment-service и подключении из order-service, затем о secrets. Повторяемый prefix позволяет переиспользовать часть **вычисления** при поддержке model runtime/provider. Он не делает три ответа одинаковыми. Сохранить первый готовый ответ и вернуть его на второй вопрос было бы ошибкой даже при идеальном prefix hit.

<!-- /unit:a9e6f6b702289e1bd3 -->

<!-- unit:e664b73a1e0eec3cd7 -->
<a id="u-e664b73a1e0eec3cd7"></a>

Наоборот, два полностью одинаковых вопроса могут требовать разных ответов, если обновилась документация или изменился разрешённый scope. Поэтому cache выбирается по тому, какой объект сохраняется. У prefix cache проверяется совместимость model computation, у answer cache — допустимость готового результата в текущей прикладной ситуации. Эти проверки связаны, но не совпадают.

<!-- /unit:e664b73a1e0eec3cd7 -->

<!-- unit:becba53bb1b0563cd8 -->
<a id="u-becba53bb1b0563cd8"></a>

Tool-result cache занимает промежуточное место. Сохранённый snapshot документа может быть переиспользован при той же версии и правах, а новый вопрос всё равно потребует другого чтения/ответа. Сохранённый result действия не заменяет исполнение нового действия. Строка «готово» в cache не доказывает, что требуемая операция произошла для нынешнего request.

<!-- /unit:becba53bb1b0563cd8 -->

<!-- unit:c28011c9ef02cf232f -->
<a id="u-c28011c9ef02cf232f"></a>

### Cache меняет не только время, но иногда и распределение ответов

<!-- /unit:c28011c9ef02cf232f -->

<!-- unit:f82c8f1b75dc16a789 -->
<a id="u-f82c8f1b75dc16a789"></a>

Для stochastic generation одинаковый prompt может давать разные допустимые формулировки. Answer cache фиксирует один ранее полученный output. Если задача требует независимые samples, это изменение semantics. В учебной двухответной модели P(A)=0.7: у десяти независимых вызовов вероятность «все A» равна $0.7^{10}\approx0.028248$, а при одном первом sample и девяти cache returns —0.7. Marginal первого ответа тот же, совместное поведение другое.

<!-- /unit:f82c8f1b75dc16a789 -->

<!-- unit:ef1b95079bf688d015 -->
<a id="u-ef1b95079bf688d015"></a>

Для FAQ несколько корректных формулировок могут быть взаимозаменяемыми по policy, и возврат первой допустим. Для генерации разнообразных идей это уже не та же задача. Включить temperature/seed в key полезно только в рамках выбранного contract; одинаковый seed сам по себе не обещает bitwise воспроизводимость всех backend execution paths.

<!-- /unit:ef1b95079bf688d015 -->

<!-- unit:455d1a359617c27105 -->
<a id="u-455d1a359617c27105"></a>

Если первый сохранённый ответ ошибочен, ошибка способна повториться на множестве hits. Это делает provenance и проверку сохранённого результата особенно важными. Проверить один cached answer1000 раз —не то же самое, что оценить 1000 независимо сгенерированных ответов.

<!-- /unit:455d1a359617c27105 -->

<!-- unit:63292a48384190bcb6 -->
<a id="u-63292a48384190bcb6"></a>

## Exact cache key и computation identity

<!-- /unit:63292a48384190bcb6 -->

<!-- unit:b2f62aa8d63fa7f58b -->
<a id="u-b2f62aa8d63fa7f58b"></a>

Пример key components: normalized input, user/tenant или permission scope, model/adapter revision, system/template version, decoding policy, corpus/index snapshot, tool result versions и relevant locale/time assumptions. Это не универсальная обязательная структура: включается всё, что меняет допустимый результат выбранного cache.

<!-- /unit:b2f62aa8d63fa7f58b -->

<!-- unit:ff904f3935e913f47b -->
<a id="u-ff904f3935e913f47b"></a>

Пусть запрос «какая версия API?» вчера отвечался по docs v1, сегодня docs v2. Key только по строке вернёт stale answer. Если answer зависит от current date или account plan, эти dependencies должны быть versioned/invalidated либо query не следует cache таким способом.

<!-- /unit:ff904f3935e913f47b -->

<!-- unit:b38ded58c1a26aff23 -->
<a id="u-b38ded58c1a26aff23"></a>

Canonicalization помогает exact hits: упорядочить незначимые whitespace/JSON keys, если это действительно сохраняет semantics. Но нельзя удалять отрицания, numbers, entity IDs или порядок instructions ради hit rate. «Платёж разрешён» и «платёж не разрешён» отличаются одним словом и противоположным смыслом.

<!-- /unit:b38ded58c1a26aff23 -->

<!-- unit:8248e676a50f6ae687 -->
<a id="u-8248e676a50f6ae687"></a>

TTL ограничивает возраст записи, но не доказывает correctness до expiration: данные могут измениться через секунду. Event/version invalidation точнее привязывает cache к изменениям. Нужны policy deletion/revocation и handling entries с ошибками; temporary tool timeout не должен стать долгим cached «ничего не найдено» без осознанного negative-cache TTL.

<!-- /unit:8248e676a50f6ae687 -->

<!-- unit:906155935b154e0f71 -->
<a id="u-906155935b154e0f71"></a>

### Ключ описывает выбранную эквивалентность, а hash только кодирует её

<!-- /unit:906155935b154e0f71 -->

<!-- unit:c28b38339b6db026aa -->
<a id="u-c28b38339b6db026aa"></a>

Представим функцию ответа $y=F(x;\theta,p,d,a)$: x —input, θ —model/adapter revision, p —prompt/decoding policy, d —данные/tools snapshot, a —authorization/locale и другие существенные условия. Для exact application cache выбирают, какие входы считаются взаимозаменяемыми по нужной задаче. Затем детерминированно сериализуют соответствующие поля и получают digest.

<!-- /unit:c28b38339b6db026aa -->

<!-- unit:a8f3f78f7249da9fb4 -->
<a id="u-a8f3f78f7249da9fb4"></a>

```text
schema_version
task_type + normalized_input
model_revision + adapter_revision
template_revision + decoding_policy
data_snapshot + tool_dependencies
permission_scope_revision + relevant_locale/time_policy
```

<!-- /unit:a8f3f78f7249da9fb4 -->

<!-- unit:f4771e66b160e628e6 -->
<a id="u-f4771e66b160e628e6"></a>

Это учебная схема, а не требование включать все поля всегда. Если ответ не зависит от locale, поле не добавляет смысла. Если зависит от текущего состояния документа, игнорировать его revision нельзя. Hash не исправляет пропущенную dependency: два разных допустимых результата попадут под один и тот же намеренно неполный key.

<!-- /unit:f4771e66b160e628e6 -->

<!-- unit:ef62fa9cb347702362 -->
<a id="u-ef62fa9cb347702362"></a>

Сериализация должна сохранять границы полей. Простая конкатенация «ab»+«c» и «a»+«bc» даёт одну строку «abc». Structured encoding или длины полей устраняют эту неоднозначность до hashing. Равный digest затем всё равно используется в контексте выбранного storage/collision protocol; длинный hash не доказывает правильность semantics.

<!-- /unit:ef62fa9cb347702362 -->

<!-- unit:ba2154c710875757ab -->
<a id="u-ba2154c710875757ab"></a>

Canonicalization тоже условна. Для application function над JSON object порядок независимых keys может быть несущественным. Для literal model prompt перестановка текста меняет token sequence, даже если человек считает содержание похожим. Удаление whitespace из кода или списков может менять смысл. Поэтому normalize следует только те различия, которые объявлены несущественными для результата, а не всё, что повышает hit rate.

<!-- /unit:ba2154c710875757ab -->

<!-- unit:07ab6fc6c4244a70c3 -->
<a id="u-07ab6fc6c4244a70c3"></a>

### TTL, eviction и dependency invalidation — три разные причины удаления

<!-- /unit:07ab6fc6c4244a70c3 -->

<!-- unit:5269350cd8e93354ff -->
<a id="u-5269350cd8e93354ff"></a>

TTL ограничивает срок жизни записи по принятой политике. Eviction освобождает место, например из-за memory budget. Invalidation говорит, что запись больше не соответствует dependency или policy. Запись может быть свежей по TTL, но invalid после изменения документа; может быть valid, но вытеснена из памяти; может истечь по времени при неизменных данных.

<!-- /unit:5269350cd8e93354ff -->

<!-- unit:cbdfd357f100eebec3 -->
<a id="u-cbdfd357f100eebec3"></a>

Пусть запись создана в t=0 с TTL60s, а документ изменился в t=1. Без иной проверки cache способен возвращать старый ответ до expiration. В t=30 запись моложе TTL, но это не доказывает её актуальность. Для snapshot-based задачи ответ по старой версии может быть корректен, если именно она запрошена; для «последней версии» —другой contract.

<!-- /unit:cbdfd357f100eebec3 -->

<!-- unit:67d5892fef439dedf1 -->
<a id="u-67d5892fef439dedf1"></a>

Negative cache должен различать подтверждённый no-results и временный timeout. Если search вернул empty set для конкретного corpus snapshot, это один результат. Если network failure превращён в такое же `[]`, приложение потеряло причину отсутствия evidence. Долгий cache такого значения закрепит ошибку как будто это факт «документов нет». Status, completeness и зависимости сохраняют вместе с payload.

<!-- /unit:67d5892fef439dedf1 -->

<!-- unit:8faadc73bf21fe74cc -->
<a id="u-8faadc73bf21fe74cc"></a>

Stale-while-revalidate возвращает прежнюю запись, пока новое вычисление идёт фоном. Это может быть допустимая freshness policy, но не автоматическое ускорение без изменения semantics. Нужен предел допустимого возраста и правило для dependencies, где stale result неприемлем. Expired по TTL и запрещённый текущим scope result нельзя смешивать в один «немного старый ответ».

<!-- /unit:8faadc73bf21fe74cc -->

<!-- unit:fd7018cbbb14316a97 -->
<a id="u-fd7018cbbb14316a97"></a>

### Гонка «удалили старое, затем старое вернулось»

<!-- /unit:fd7018cbbb14316a97 -->

<!-- unit:23dede2896b1e97a43 -->
<a id="u-23dede2896b1e97a43"></a>

В t0 worker A читает docs v7 и начинает generation. В t1 публикуется v8, cache key только по вопросу удаляется. В t2 A завершает вычисление по v7 и снова записывает старый answer под тем же key. Однократное delete не предотвратило stale resurrection.

<!-- /unit:23dede2896b1e97a43 -->

<!-- unit:828c0aa1e4aa757674 -->
<a id="u-828c0aa1e4aa757674"></a>

Версионный key привязывает answer к snapshot, который реально использован: результат v7 записывается как v7 и не становится результатом v8. Нельзя в момент публикации просто взять «текущую» revision v8 и присвоить её ответу, созданному из v7. При policy «отвечать по самой новой версии» дополнительно определяют, что делать с изменением dependencies во время активного request: повторить, вернуть явно указанный snapshot или иной согласованный путь.

<!-- /unit:828c0aa1e4aa757674 -->

<!-- unit:5879da4785e67a8280 -->
<a id="u-5879da4785e67a8280"></a>

Общий учебный протокол: зафиксировать dependency identity → искать соответствующий key → проверить допустимость использования → на miss вычислить по этому snapshot → опубликовать с тем же identity по выбранному concurrency protocol. Между разными storage systems простая проверка перед write не создаёт универсальную атомарную transaction. Требуемая consistency и revocation semantics должны быть свойством всей схемы, а не обещанием cache helper.

<!-- /unit:5879da4785e67a8280 -->

<!-- unit:92f50badb4f41bb4c4 -->
<a id="u-92f50badb4f41bb4c4"></a>

При изменении прав key со старым permission revision может физически остаться, но не должен считаться разрешением на выдачу данных. Актуальная authorization policy проверяется в нужной точке read/return. Сам факт, что ответ когда-то был виден одному пользователю, не делает его доступным другому.

<!-- /unit:92f50badb4f41bb4c4 -->

<!-- unit:3dbcfc20d5ffc5d1d4 -->
<a id="u-3dbcfc20d5ffc5d1d4"></a>

## Semantic similarity не равна взаимозаменяемости ответа

<!-- /unit:3dbcfc20d5ffc5d1d4 -->

<!-- unit:26a646da07be9193ac -->
<a id="u-26a646da07be9193ac"></a>

Dense embedding может сблизить «удали черновик 42» и «покажи черновик 42», «лимит 100» и «лимит 1000». Для retrieval это похожие темы, для answer reuse — разные задачи. Threshold cosine0.95 не является доказательством одинакового смысла или authorization.

<!-- /unit:26a646da07be9193ac -->

<!-- unit:116d43213d8308c5f6 -->
<a id="u-116d43213d8308c5f6"></a>

Учебный safe pattern для read-only FAQ: filter by scope/version/task type → exact entity/number/negation checks → semantic candidates → separate acceptance rule → reuse verified answer с provenance, либо обычный generation. Для действий с side effects final answer cache не означает, что действие уже выполнено в новом контексте.

<!-- /unit:116d43213d8308c5f6 -->

<!-- unit:3932f65edde1e760ff -->
<a id="u-3932f65edde1e760ff"></a>

Оценивать semantic-cache precision: доля accepted hits, где reused answer допустим; accepted hit rate — какую долю requests обслужили reuse; recall допустимых возможностей reuse требует отдельной разметки; latency/cost и stale/error rate. Если высокая цена false hit, лучше меньший hit rate. Human audit должен включать near-threshold и counterexample queries, а не только очевидные synonyms.

<!-- /unit:3932f65edde1e760ff -->

<!-- unit:e35e971bad1309107c -->
<a id="u-e35e971bad1309107c"></a>

### Candidate retrieval ещё не решение reuse

<!-- /unit:e35e971bad1309107c -->

<!-- unit:b147da83527d6d2f96 -->
<a id="u-b147da83527d6d2f96"></a>

Semantic search находит близкие сохранённые queries. Это этап предложения кандидата, аналог first-stage retrieval. Acceptance проверяет, можно ли вернуть **этот** answer для текущего query. Вопросы «включить feature X?» и «как выключить feature X?» могут быть тематически близки, но намерение различается. Даже совпадение entity X не устраняет противоположный intent.

<!-- /unit:b147da83527d6d2f96 -->

<!-- unit:5e8254fcd4374887a6 -->
<a id="u-5e8254fcd4374887a6"></a>

Проверки чисел, IDs, отрицаний и task type полезны как части policy. Они не исчерпывают смысл: одинаковые слова могут относиться к разным условиям или версиям. Если используется дополнительный judge equivalence, его false accepts/rejects измеряют отдельно. Высокий cosine не становится вероятностью валидности answer только потому, что выбран красивый threshold0.95.

<!-- /unit:5e8254fcd4374887a6 -->

<!-- unit:a6eab65e9e59f63142 -->
<a id="u-a6eab65e9e59f63142"></a>

Полезно различать два свойства: query-answer compatibility и correctness самого сохранённого answer. Два queries могут быть эквивалентны, но исходный answer неверен. Тогда reuse повторяет ту же ошибку. Проверять лишь перефразирование вопроса недостаточно для end-to-end quality.

<!-- /unit:a6eab65e9e59f63142 -->

<!-- unit:e489154b8bbd2d2f3c -->
<a id="u-e489154b8bbd2d2f3c"></a>

### Similarity threshold не создаёт отношение эквивалентности

<!-- /unit:e489154b8bbd2d2f3c -->

<!-- unit:aca43ed27a7ce07f51 -->
<a id="u-aca43ed27a7ce07f51"></a>

Возьмём unit vectors под углами 0°,10°,20°. Cosine соседей≈0.984808, крайних≈0.939693. При threshold0.98 A близок B, B близок C, A не близок C. Это искусственная геометрическая иллюстрация, не измерение реального encoder. Если объявлять новый accepted query представителем следующего и строить цепочку aliases, можно постепенно уйти от исходного вопроса/answer.

<!-- /unit:aca43ed27a7ce07f51 -->

<!-- unit:742fe69b3be3fde886 -->
<a id="u-742fe69b3be3fde886"></a>

Поэтому cache equivalence нельзя выводить из connected components threshold graph без дополнительной модели задачи. Сравнение с конкретной исходной записью и её dependencies отличается от транзитивного расширения класса «похожих». Та же проблема обсуждалась для duplicates в BERT-разделе; здесь ошибка приводит к reuse неправильного output.

<!-- /unit:742fe69b3be3fde886 -->

<!-- unit:2519b20f63af79c681 -->
<a id="u-2519b20f63af79c681"></a>

### Четыре denominators вместо одного hit rate

<!-- /unit:2519b20f63af79c681 -->

<!-- unit:a886a6185143fc87f6 -->
<a id="u-a886a6185143fc87f6"></a>

Пусть из 1000 requests semantic lookup нашёл candidates для 600, acceptance пропустил 500, из них 495 reused answers признаны допустимыми. Lookup candidate rate=0.6, accepted hit rate=0.5, accepted-hit precision=495/500=0.99. Пять false hits —1% accepted hits и 0.5% всех requests. Эти два процента нельзя незаметно подменять друг другом.

<!-- /unit:a886a6185143fc87f6 -->

<!-- unit:b01de758c87b1a2b54 -->
<a id="u-b01de758c87b1a2b54"></a>

Recall reuse определяется относительно **всех действительно допустимых возможностей reuse**, а не числа найденных candidates. Он требует соответствующей разметки или sampling evaluation. По одному log accepted/rejected нельзя узнать, сколько полезных candidates поиск не нашёл. Также lookup miss, rejection, stale dependency и wrong answer —разные причины fallback.

<!-- /unit:b01de758c87b1a2b54 -->

<!-- unit:251283020761575fc2 -->
<a id="u-251283020761575fc2"></a>

Threshold выбирают на разрешённой validation выборке с counterexamples, затем фиксируют. Если каждый найденный false accept исправлять и сразу считать по тем же cases итоговый score, получается regression training set, а не независимая оценка. Нужны и известные трудные пары, и свежая representative sample.

<!-- /unit:251283020761575fc2 -->

<!-- unit:6741b7b665cb385abd -->
<a id="u-6741b7b665cb385abd"></a>

## Когда caching вообще выгоден

<!-- /unit:6741b7b665cb385abd -->

<!-- unit:d3e9c2a9821d1bd20b -->
<a id="u-d3e9c2a9821d1bd20b"></a>

Пусть cache lookup5ms, обычный answer1000ms, accepted hit rate h, hit validation ещё 5ms. Упрощённое expected time $5+h\cdot 5+\left(1-h\right)\cdot 1000$. При h0.5 получим 507.5ms; при h0 —1005ms, cache только добавил overhead. При false hits скорость не считается успешным решением: quality constraint первична.

<!-- /unit:d3e9c2a9821d1bd20b -->

<!-- unit:d5af954ce959a53376 -->
<a id="u-d5af954ce959a53376"></a>

Аналогично cost включает lookup/vector storage/validation/maintenance. Semantic judge на каждый query может съесть значительную часть экономии. Сначала измерить actual repetition и sensitivity to stale results; затем сравнивать exact cache, semantic cache и no cache на одном workload.

<!-- /unit:d5af954ce959a53376 -->

<!-- unit:06d726d274cba60e90 -->
<a id="u-06d726d274cba60e90"></a>

Stampede: много simultaneous misses одного expensive key создают повторную генерацию. Single-flight/request coalescing может объединить работу, если inputs/scope equivalent. Cancellation одного клиента не должна неправильно отменить результат, нужный другим. Distributed invalidation/concurrency требуют своей semantics.

<!-- /unit:06d726d274cba60e90 -->

<!-- unit:cd1f69fdf76ee76b14 -->
<a id="u-cd1f69fdf76ee76b14"></a>

### Вывод break-even с явно заданными затратами

<!-- /unit:cd1f69fdf76ee76b14 -->

<!-- unit:9f88b83f3fb915b759 -->
<a id="u-9f88b83f3fb915b759"></a>

В исходной простой модели lookup L=5ms, hit validation V=5ms, fresh answer T=1000ms, accepted hit fraction h. Среднее $E=L+hV+(1-h)T$. Cache быстрее baseline T, если

<!-- /unit:9f88b83f3fb915b759 -->

<!-- unit:aa9d76ff006f5c6701 -->
<a id="u-aa9d76ff006f5c6701"></a>

$$
h(T-V)>L,\qquad h>\frac{L}{T-V}\quad\text{при }T>V.
$$

<!-- /unit:aa9d76ff006f5c6701 -->

<!-- unit:b77d58348b9002f127 -->
<a id="u-b77d58348b9002f127"></a>

Для этих чисел порог h≈0.005025, то есть около 0.503%. Это вывод лишь для объявленной модели с validation cost на accepted hits и одинаковой fresh cost. При h=0.5 получается 507.5ms; при h=0 cache добавляет 5ms. Реальные rejected candidates также могут требовать дорогую проверку.

<!-- /unit:b77d58348b9002f127 -->

<!-- unit:fda901ea395f01b187 -->
<a id="u-fda901ea395f01b187"></a>

Пусть candidate находится с вероятностью c, принимается с условной вероятностью a, так что h=ca. Если validation V оплачивается каждому candidate, включая rejected, то $E=L+cV+(1-ca)T$. При c=0.6,a=0.8,V=50ms,L=5ms,T=1000ms получаем 555ms. Игнорирование rejected validations занизило бы стоимость.

<!-- /unit:fda901ea395f01b187 -->

<!-- unit:3e13ac1988719b6158 -->
<a id="u-3e13ac1988719b6158"></a>

Если fresh miss queries в среднем длиннее cached queries, один общий T уже недостаточен. Раздельно оценивают costs по slices и по тому же request workload. Provider token accounting, storage, embedding lookup, judge и maintenance добавляют свои единицы. Учебные milliseconds не являются measurements конкретного API или железа.

<!-- /unit:3e13ac1988719b6158 -->

<!-- unit:e64c09f6e871d205ae -->
<a id="u-e64c09f6e871d205ae"></a>

### Request hit rate не равен доле сэкономленных tokens

<!-- /unit:e64c09f6e871d205ae -->

<!-- unit:1f1d011796ac39aa96 -->
<a id="u-1f1d011796ac39aa96"></a>

Из 1000 requests800 коротких cache hits по 100 prompt tokens,200 misses по 5000. Если сравнить с no-cache на том же workload, потенциально пропущено 80000 из 1080000 prompt tokens, около 7.41%, хотя request hit rate80%. Output tokens и другие charges пока вообще не учтены. Поэтому «80% hits» не означает автоматически «80% меньше счёт».

<!-- /unit:1f1d011796ac39aa96 -->

<!-- unit:ff9a5882e9b416d5f0 -->
<a id="u-ff9a5882e9b416d5f0"></a>

И tail latency может вести себя иначе, чем mean. Для 90 hits по 10ms и 10 misses по 2000ms mean209ms, но nearest-rank p95=2000ms. Средний ответ стал быстрым, а выбранный tail percentile всё ещё определяется misses. Эти числа —учебная fixed sample с объявленным percentile convention.

<!-- /unit:ff9a5882e9b416d5f0 -->

<!-- unit:69538c3e05e14b831e -->
<a id="u-69538c3e05e14b831e"></a>

### Single-flight объединяет работу, а не превращает все misses в готовые hits

<!-- /unit:69538c3e05e14b831e -->

<!-- unit:bacf6dc0d335d8f91a -->
<a id="u-bacf6dc0d335d8f91a"></a>

Пусть 20 equivalent requests одновременно пришли на пустой cache. Без coordination все могут начать generation. Single-flight выбирает одного producer, остальные ждут общий result. Число upstream generations может уменьшиться 20→1, то есть на 95%, но initial ready-cache hit rate остаётся 0: запись ещё не существовала. Coalesced waiters следует учитывать отдельной metric.

<!-- /unit:bacf6dc0d335d8f91a -->

<!-- unit:73a7b8d0e52efacc5f -->
<a id="u-73a7b8d0e52efacc5f"></a>

Это не обещает ускорить latency каждого request в 20 раз: параллельные вычисления могли батчиться, а waiters всё равно ждут producer. Выгода зависит от ресурса и workload. Отмена одного waiter не обязана отменять computation, которое нужно другим. Future завершается успехом или error; in-flight registry очищается по всем terminal paths, иначе последующие requests могут ждать навсегда.

<!-- /unit:73a7b8d0e52efacc5f -->

<!-- unit:ac3297822d96e2add6 -->
<a id="u-ac3297822d96e2add6"></a>

В distributed lease варианте owner A может задержаться дольше TTL, после чего lease получит B. Поздний A не должен безусловно удалить lock B. Сравнение owner token при release объяснено в [Redis distributed-locks pattern](https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/). Но защита unlock не делает любой поздний write корректным: publication результата требует своего generation/version protocol. Random ownership token и упорядоченная версия записи решают разные задачи.

<!-- /unit:ac3297822d96e2add6 -->

<!-- unit:5cda749b4fc5c68475 -->
<a id="u-5cda749b4fc5c68475"></a>

### Storage budget и eviction тоже входят в эксперимент

<!-- /unit:5cda749b4fc5c68475 -->

<!-- unit:31bafe84c274b74116 -->
<a id="u-31bafe84c274b74116"></a>

Для 100000 entries с answer payload2KiB только тексты занимают 204800000 bytes. Query vectors ширины 768 float32 добавят 307200000 bytes, даже без index overhead. Если key digest32 bytes и metadata64 bytes на entry, эти компоненты добавят ещё 3.2 и 6.4 MB decimal. Сумма перечисленного —521.6 MB до allocator/index/replicas и хранения дополнительных dependencies.

<!-- /unit:31bafe84c274b74116 -->

<!-- unit:1bda73e6cd7e7aa69b -->
<a id="u-1bda73e6cd7e7aa69b"></a>

Число requests не равно числу entries: много repeats одного key может сохранять одну запись, а почти все уникальные вопросы быстро увеличивают footprint при низком reuse. TTL регулирует age, memory eviction —место, semantic acceptance —правильность. Одной настройкой TTL нельзя оптимизировать все три свойства сразу.

<!-- /unit:1bda73e6cd7e7aa69b -->

<!-- unit:8fe7202bb8b3a40318 -->
<a id="u-8fe7202bb8b3a40318"></a>

## Мониторинг технического состояния и качества

<!-- /unit:8fe7202bb8b3a40318 -->

<!-- unit:15e20372fac45e28b3 -->
<a id="u-15e20372fac45e28b3"></a>

Operational metrics: requests, queue age, TTFT/inter-token/end-to-end p50/p95, timeout/error/OOM, input/output lengths, cache hit/reject/stale rates, version distribution и cost. Их следует привязывать к workload slices: language, input length, tenant/task type, retrieved source quality.

<!-- /unit:15e20372fac45e28b3 -->

<!-- unit:ccd6e91e0d097b6fab -->
<a id="u-ccd6e91e0d097b6fab"></a>

Quality metrics: task correctness, schema validity, wrong-tool/args, retrieval relevance, evidence/citation support, false abstention/refusal и successful completion. Ни latency, ни user thumbs-up в одиночку не доказывают correctness. Feedback имеет selection bias: довольные/недовольные пользователи оставляют оценки с разной вероятностью.

<!-- /unit:ccd6e91e0d097b6fab -->

<!-- unit:9853875c15f960b012 -->
<a id="u-9853875c15f960b012"></a>

Delayed labels требуют оценивать quality после появления результата, сохраняя связь с version/time cohort. Drift input embeddings/length/language может быть ранним indicator, но не равен доказанному падению task quality. Concept drift может ухудшить correctness без большого сдвига внешней формы запросов.

<!-- /unit:9853875c15f960b012 -->

<!-- unit:ea9bda43917c8fe338 -->
<a id="u-ea9bda43917c8fe338"></a>

### Связать quality с конкретной траекторией и версиями

<!-- /unit:ea9bda43917c8fe338 -->

<!-- unit:0ae816bad6e32d388e -->
<a id="u-0ae816bad6e32d388e"></a>

Для разбора request нужны IDs и версии: model/adapter, template, corpus/index, cache policy, selected entry/dependencies, tools, evaluator. У cache hit может вообще не быть нового model forward; если считать только model spans, такие ответы исчезнут из end-to-end статистики. У agent request может быть несколько model calls и tool waits; TTFT каждого call и время до первого пользовательского ответа —разные величины.

<!-- /unit:0ae816bad6e32d388e -->

<!-- unit:94ba52677c5b17228e -->
<a id="u-94ba52677c5b17228e"></a>

Monitoring counters должны сообщать, что именно считают. Candidates, accepted hits, coalesced waiters, expired entries, invalidated entries и fresh generations различаются. Метрика «cache hit» без определения способна вырасти после переименования waiters, хотя фактическое поведение не изменилось. Аналогично costs привязывают к token categories конкретного execution contract, не объявляя любой cache read бесплатным.

<!-- /unit:94ba52677c5b17228e -->

<!-- unit:f2d2db12be1423a7ec -->
<a id="u-f2d2db12be1423a7ec"></a>

### У generative задачи могут быть проверяемые истины без единственной формулировки

<!-- /unit:f2d2db12be1423a7ec -->

<!-- unit:7f98ce5c54ed0a2c2a -->
<a id="u-7f98ce5c54ed0a2c2a"></a>

План говорит об отсутствии одного правильного ответа в generative системах. Это полезное отличие от одного class ID, но не отсутствие проверяемых требований вообще. «Timeout равен 5s» и «Ожидание прекращается через 5 секунд» могут быть двумя корректными формулировками. «50 секунд» лексически похоже, но неверно относительно тех же docs.

<!-- /unit:7f98ce5c54ed0a2c2a -->

<!-- unit:ec55ca2a56be936e2b -->
<a id="u-ec55ca2a56be936e2b"></a>

Для extraction можно проверить schema, IDs и значения; для code —заявленные tests, помня об их coverage; для summary —faithfulness и покрытие существенных фактов по rubric. JSON validity не доказывает истинность поля. Наличие citation не доказывает, что именно этот source поддерживает claim. Поэтому качество оценивают по отдельным наблюдаемым свойствам вместо одного расплывчатого «ответ понравился».

<!-- /unit:ec55ca2a56be936e2b -->

<!-- unit:f82e1c34654c9aaa22 -->
<a id="u-f82e1c34654c9aaa22"></a>

### Почему сравнивать hits и misses напрямую нечестно

<!-- /unit:f82e1c34654c9aaa22 -->

<!-- unit:40e71105a59ad15d00 -->
<a id="u-40e71105a59ad15d00"></a>

Представим workload пополам: frequent easy queries и rare hard. No-cache quality на них 99% и 90%, общий результат 94.5%. Cache возвращает easy answers с quality98%, misses на hard остаются 90%; итог 94.0%. Если сравнить только общий cached score94% с miss score90%, кажется, что cache улучшил качество. На том же workload он ухудшил его на 0.5 процентного пункта.

<!-- /unit:40e71105a59ad15d00 -->

<!-- unit:2759f5a83dd4a99488 -->
<a id="u-2759f5a83dd4a99488"></a>

Нужен counterfactual no-cache baseline на сопоставимых queries, а не сравнение разных по трудности подмножеств. Shadow generation на sample cache hits может дать такую проверку, но тратит compute и имеет собственный protocol. Changed-data и permission cases проверяют отдельно: обычные synonyms не измеряют stale reuse.

<!-- /unit:2759f5a83dd4a99488 -->

<!-- unit:c6d775913863669748 -->
<a id="u-c6d775913863669748"></a>

### Повторяемая ошибка и delayed ground truth

<!-- /unit:c6d775913863669748 -->

<!-- unit:0f1b0c37a5c672ee5b -->
<a id="u-0f1b0c37a5c672ee5b"></a>

Если один cached answer возвращён 1000 раз, его оценка 1000 пользователями создаёт зависимые observations. Для крайних toy условий пусть вероятность неверного первого answer p=0.01. У 1000 independent generations число ошибок имеет variance $1000p(1-p)=9.9$. При одном first sample, повторённом 1000 раз, variance числа ошибок $1000^2p(1-p)=9900$ при том же expected count10. Cache усиливает корреляцию исхода, а не обязательно меняет его marginal expectation.

<!-- /unit:0f1b0c37a5c672ee5b -->

<!-- unit:64d921486baf03720d -->
<a id="u-64d921486baf03720d"></a>

Поэтому audit может учитывать request-weighted impact и отдельно unique answer/cache-key groups. Оба нужны: одна ошибка, показанная 1000 раз, имеет большой пользовательский эффект, но не является 1000 независимыми доказательствами качества генератора. Similarity drift и user feedback не заменяют такую оценку.

<!-- /unit:64d921486baf03720d -->

<!-- unit:04c0b34d92dfa49662 -->
<a id="u-04c0b34d92dfa49662"></a>

Ground truth тоже может приходить позже. Для ответа, созданного по docs v7, важно различать correctness относительно тогдашнего snapshot и freshness относительно уже опубликованной v8. Сохраняемый cohort/version context позволяет не объявить исторически правильный ответ ошибкой только из-за смены evaluator data, и одновременно увидеть реальные stale returns после обновления.

<!-- /unit:04c0b34d92dfa49662 -->

<!-- unit:0504d83395ff813019 -->
<a id="u-0504d83395ff813019"></a>

## Regression suite и judge

<!-- /unit:0504d83395ff813019 -->

<!-- unit:226d97b72e7367940c -->
<a id="u-226d97b72e7367940c"></a>

Фиксированный curated набор ловит известные failures; свежая случайная sample production показывает новые. Оба нужны: static benchmark может загрязняться tuning, а только fresh sample плохо воспроизводится. Учитывать source/group/time dependence при uncertainty.

<!-- /unit:226d97b72e7367940c -->

<!-- unit:1c47a61bc3fa9905fd -->
<a id="u-1c47a61bc3fa9905fd"></a>

LLM judge получает rubric, evidence, blind/order-swapped outputs и проверяется на human subset. Это дополнение к deterministic oracle там, где он доступен. Изменение judge version может изменить score при той же policy — сохранять evaluator identity.

<!-- /unit:1c47a61bc3fa9905fd -->

<!-- unit:387de4b7ebbb5261a5 -->
<a id="u-387de4b7ebbb5261a5"></a>

Перед rollout задать gates, например «не ухудшать exact extraction correctness в ключевом slice» и «p95 в определённом workload ниже budget». Конкретные numbers определяет application, а не универсальный учебник. Для rollout/rollback policy нужно достаточно observations, чтобы не реагировать на один случай как на устойчивый trend.

<!-- /unit:387de4b7ebbb5261a5 -->

<!-- unit:1b62ada066f674681a -->
<a id="u-1b62ada066f674681a"></a>

### Rubric и oracle должны следовать задаче

<!-- /unit:1b62ada066f674681a -->

<!-- unit:981c470205e32cbb22 -->
<a id="u-981c470205e32cbb22"></a>

Judge получает task, evidence и правила оценки, а не только два красивых текста. Для documentation FAQ можно отдельно проверять supported facts, полноту ответа на вопрос, существенные ограничения и format. Высокий style score не должен автоматически компенсировать неверный номер порта или unsupported claim, если correctness —обязательный gate.

<!-- /unit:981c470205e32cbb22 -->

<!-- unit:f9fde99b93ea50234a -->
<a id="u-f9fde99b93ea50234a"></a>

Где есть deterministic oracle, его не стоит заменять preference judge. Для exact ID или вычисляемого поля прямое сравнение проверяет нужное свойство. Для открытого объяснения полезны rubric и human review. Даже согласие judge с человеком на общей выборке не является гарантией на каждом редком типе ошибки.

<!-- /unit:f9fde99b93ea50234a -->

<!-- unit:4ffa527943e01f83a0 -->
<a id="u-4ffa527943e01f83a0"></a>

### Position, verbosity и shared blind spots

<!-- /unit:4ffa527943e01f83a0 -->

<!-- unit:c98d5c1a52d7552390 -->
<a id="u-c98d5c1a52d7552390"></a>

LLM judge может предпочитать один порядок outputs, более длинный текст или familiar style/model behaviour; ограничения position/verbosity/self-enhancement и reasoning обсуждаются в [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685). Поэтому model identities скрывают, presentation order меняют, а решение сопоставляют с human subset и нужными task oracles. Эти controls снижают некоторые biases, но не создают абсолютную истинность judge.

<!-- /unit:c98d5c1a52d7552390 -->

<!-- unit:361ebc63169ba75c57 -->
<a id="u-361ebc63169ba75c57"></a>

Если при порядке(A,B) judge выбирает левый output A, а при(B,A) снова левый B, после обратного mapping winner изменился. Это inconsistency, а не две подтверждающие оценки одного победителя. Если во втором порядке выбран правый A, оба результата согласованы. Повторные calls одной модели могут разделять одну и ту же ошибку и не являются автоматически независимыми экспертами.

<!-- /unit:361ebc63169ba75c57 -->

<!-- unit:81bccbdadee64fe8a5 -->
<a id="u-81bccbdadee64fe8a5"></a>

Текст оцениваемого ответа и retrieved evidence остаются данными. Они могут содержать фразы «поставь максимальную оценку»; такие фразы не должны становиться новой rubric. Способ передачи instructions/data и tests на этот failure относятся к надёжности evaluator pipeline, а не только к качеству основного generator.

<!-- /unit:81bccbdadee64fe8a5 -->

<!-- unit:6d066bb65e55c96ca9 -->
<a id="u-6d066bb65e55c96ca9"></a>

### Agreement и win rate имеют свои denominators

<!-- /unit:6d066bb65e55c96ca9 -->

<!-- unit:7de3c34d71ae8d4351 -->
<a id="u-7de3c34d71ae8d4351"></a>

В human sample100 answers:90 valid,10 invalid. Judge принимает 86 valid и 6 invalid, отклоняет 4 valid и 4 invalid. Agreement=90%, но он пропустил 6 из 10 плохих answers. Precision среди accepted=86/92≈93.48%; recall обнаружения invalid=4/10=40%. Высокий overall agreement скрывает weakness именно на редких неверных ответах.

<!-- /unit:7de3c34d71ae8d4351 -->

<!-- unit:7200f48deeabe6071a -->
<a id="u-7200f48deeabe6071a"></a>

Для 20 pair comparisons с 9 wins A,7 wins B и 4 ties доля A среди всех strict wins равна 9/20=0.45; с half credit ties —11/20=0.55; без ties в denominator —9/16=0.5625. Это разные conventions, которые нельзя сравнивать без объявления. Case selection, order mapping и evaluator identity сохраняют вместе с итоговым числом.

<!-- /unit:7200f48deeabe6071a -->

<!-- unit:8ae847d441b43dd7ca -->
<a id="u-8ae847d441b43dd7ca"></a>

### Regression suite должна различать изменение модели и изменение судьи

<!-- /unit:8ae847d441b43dd7ca -->

<!-- unit:f8ce8472e0f139fe99 -->
<a id="u-f8ce8472e0f139fe99"></a>

Для known failures хранят фиксированные cases с ожидаемыми properties, для новых —fresh representative samples. Настройка на known cases улучшает regression coverage, но эти же cases уже не являются полностью независимым test. Срезы по scope/version, отрицанию, числам и no-evidence показывают, где именно меняется качество.

<!-- /unit:f8ce8472e0f139fe99 -->

<!-- unit:e30e30ae378511fd6a -->
<a id="u-e30e30ae378511fd6a"></a>

При обновлении judge сначала оценивают одни и те же frozen answers старым и новым evaluator. При обновлении cache/model policy сравнивают результаты одним зафиксированным evaluator. Если менять обе стороны одновременно, падение score может означать более строгую rubric, а не ухудшение ответов. Потом human/oracle checks помогают интерпретировать расхождение.

<!-- /unit:e30e30ae378511fd6a -->

<!-- unit:6ba59fae887149b8c1 -->
<a id="u-6ba59fae887149b8c1"></a>

Rollout gates задают до просмотра final results: какие task properties обязательны, какие slices значимы, какой latency/cost budget допустим. Их thresholds зависят от приложения и данных. Эта глава объясняет experimental contract; она не подтверждает, что ваш service уже существует или прошёл такой rollout.

<!-- /unit:6ba59fae887149b8c1 -->

<!-- unit:5ed58022ed1495dca7 -->
<a id="u-5ed58022ed1495dca7"></a>

## Полный эксперимент и самопроверка

<!-- /unit:5ed58022ed1495dca7 -->

<!-- unit:beef3489eba4cf1274 -->
<a id="u-beef3489eba4cf1274"></a>

Сравнить no cache, exact cache и один semantic policy: один corpus snapshot/model, фиксированные queries с synonyms, negations, different numbers, permissions и changed-data cases. Измерить correctness reused outputs, accepted hit rate, latency/cost. Затем invalidation по document version и permission revoke.

<!-- /unit:beef3489eba4cf1274 -->

<!-- unit:47e029d26f13c1d813 -->
<a id="u-47e029d26f13c1d813"></a>

1. Почему TTL не гарантия freshness? **Ответ:** underlying state может измениться до expiration.
2. Почему cosine similarity недостаточно? **Ответ:** тематическая близость не доказывает equality entities/constraints/intent.
3. Можно ли alarm на embedding drift считать quality regression? **Ответ:** это indicator; нужен task-specific evidence.
4. Чем application answer cache отличается от prompt cache? **Ответ:** возвращает готовый output вместо reuse intermediate prefix computation.

<!-- /unit:47e029d26f13c1d813 -->

<!-- unit:f2c4fc300d06f95e82 -->
<a id="u-f2c4fc300d06f95e82"></a>

5. Сохраняет ли answer cache независимость stochastic samples? **Ответ:** нет. Повторно возвращается один sample; вероятность совместных событий и корреляция ошибок меняются.
6. Исправит ли длинный hash пропущенную data revision? **Ответ:** нет. Он кодирует переданные fields; пропущенная dependency остаётся пропущенной.
7. Почему deletion при обновлении недостаточно? **Ответ:** старый producer может завершиться позже и снова записать stale output под тем же key. Нужен version/publication protocol.
8. Равен ли candidate rate accepted hit rate? **Ответ:** нет. Candidate может быть rejected, stale или несовместимым по scope; accepted-hit precision имеет ещё другой denominator.
9. Почему 80% request hits не означают 80% экономии tokens? **Ответ:** hits могут быть короткими, misses длинными; экономия взвешивается фактическими единицами работы и затрат.
10. Делает ли single-flight все waiting requests ready-cache hits? **Ответ:** нет. Они ждут ещё не готовое общее вычисление; coalescing учитывают отдельно.
11. Достаточно ли 90% judge agreement для проверки редких неверных ответов? **Ответ:** нет. Нужно смотреть confusion по важным outcomes; в примере обнаружено лишь 40% invalid answers.
12. Можно ли сравнить quality hits и misses как две равные группы? **Ответ:** без контроля difficulty и selection —нет. Нужен сопоставимый no-cache baseline и отдельные slices.

<!-- /unit:f2c4fc300d06f95e82 -->

<!-- unit:289b9889a7c22a976c -->
<a id="u-289b9889a7c22a976c"></a>

**Практика:** составить cache policy для documentation FAQ с key dependencies и forbidden-to-reuse cases. Проверить десять пар запросов, включая «включить/выключить», разные IDs и даты. Объяснить, где hit допустим и какое evidence нужно для этого решения.

<!-- /unit:289b9889a7c22a976c -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как доказать допустимость reuse и заметить регрессию качества?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
