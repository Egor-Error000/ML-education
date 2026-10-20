# Оценка retrieval, ответов и task success

**Уровень:** основа ветки.

**Предпосылки:** [RAG: BM25, dense, hybrid и multi-vector retrieval](<app-rag.md>)

**Используют:** [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>)

**Учебный вопрос:** Как локализовать ошибку приложения и измерить стоимость успешной задачи?





## Карта вычисления и границы этапов

![Оценка retrieval, ответов и task success: входы, преобразования и выходы](<../../оформление/course/app-evaluation.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:58ff1c7e84f524bc03 -->
<a id="u-58ff1c7e84f524bc03"></a>

## Как разделить ошибки

<!-- /unit:58ff1c7e84f524bc03 -->

<!-- unit:08fd06b6ef90750bea -->
<a id="u-08fd06b6ef90750bea"></a>

Retrieval failure: evidence не найдено. Context selection failure: найдено, но не передано/обрезано. Generation failure: evidence есть, ответ неверный. Tool failure: wrong tool/args или ошибка исполнения. State failure: потерян результат/ID, повторено действие. Такие категории позволяют выбрать intervention точнее, чем «агент тупит».

<!-- /unit:08fd06b6ef90750bea -->

<!-- unit:b3f4ce5e4f9d54cee8 -->
<a id="u-b3f4ce5e4f9d54cee8"></a>

Oracle context experiment подаёт gold snippet вместо retrieval. Если ответ теперь correct, bottleneck вероятно upstream; если нет — prompt/model/evidence interpretation требуют разбора. JSON validity и tool HTTP success — локальные diagnostics, final task success оценивается отдельно.

<!-- /unit:b3f4ce5e4f9d54cee8 -->

<!-- unit:32865d03932378da9d -->
<a id="u-32865d03932378da9d"></a>

### Разбираем одну задачу до terminal outcome

<!-- /unit:32865d03932378da9d -->

<!-- unit:90bee6e1c8dedea1af -->
<a id="u-90bee6e1c8dedea1af"></a>

Учебная задача: «найди timeout payment-service v8 и процитируй». Trusted state разрешает только проект P. Первый поиск возвращает snippet v7 с timeout 2 s. Это observation реального поиска, но ещё не ответ на вопрос v8. Модель предлагает уточнить version, второй поиск возвращает v8 с timeout 5 s и ID `P/pay/v8#timeout`. После чтения полного условия ответ указывает 5 s и ограничение применимости. Validator проверяет наличие ID среди observations и нужную version; содержательную корректность числа сверяют с текстом.

<!-- /unit:90bee6e1c8dedea1af -->

<!-- unit:37ce7d0241f5ed3f2e -->
<a id="u-37ce7d0241f5ed3f2e"></a>

Если второй поиск пустой, ответ «в v8 тоже 2 s» не следует из первого результата. Корректно сказать, что найдена только v7 и значение для v8 не установлено. Если tool timeout, в state хранится ошибка исполнения, а не no-results: мы не знаем, существовал ли документ. Если найденный ID недоступен, он не становится разрешённым из-за уверенности модели. Эти три случая требуют разных observations и разных дальнейших решений.

<!-- /unit:37ce7d0241f5ed3f2e -->

<!-- unit:9c17e650df60dd7368 -->
<a id="u-9c17e650df60dd7368"></a>

Полезная trace-запись содержит исходную задачу, revision prompt/model/tool registry, model-call index, call ID, validated args, scope decision, result/error, evidence IDs и terminal reason. Длинный лог всех строк сам по себе не заменяет связи между ними. Сводный статус success при незавершённом pending call скрывает ошибку жизненного цикла.

<!-- /unit:9c17e650df60dd7368 -->

<!-- unit:76b433936ab4f25b71 -->
<a id="u-76b433936ab4f25b71"></a>

### Oracle experiments и локальные метрики

<!-- /unit:76b433936ab4f25b71 -->

<!-- unit:850e9885c51ee6ecb3 -->
<a id="u-850e9885c51ee6ecb3"></a>

Чтобы проверить retrieval, вручную подставляют нужный v8 snippet. Если модель всё равно отвечает 2 s, проблема не исчерпывается поиском. Если теперь ответ верный, upstream становится основным подозреваемым, но один успешный пример не доказывает, что generator всегда надёжен. Затем отдельно проверяют context packaging: возможно, retriever уже находил v8, но truncation удалял именно его.

<!-- /unit:850e9885c51ee6ecb3 -->

<!-- unit:58727f83d2dbe6b41f -->
<a id="u-58727f83d2dbe6b41f"></a>

На 100 учебных задачах 90 calls могут быть schema-valid, 80 —к правильному tool, 70 —успешно исполненными, 60 —дающими нужный evidence и 55 —завершёнными корректным ответом. Если это вложенные множества одной трассы на задачу, validity 90% и end-to-end success 55% совместимы. Условный success после найденного evidence равен 55/60≈91.67%. При нескольких calls на задачу denominators уже нужно определять отдельно: долю валидных calls нельзя напрямую считать долей успешных tasks.

<!-- /unit:58727f83d2dbe6b41f -->

<!-- unit:b11bb3a3eb2c4be5f5 -->
<a id="u-b11bb3a3eb2c4be5f5"></a>

Число retries полезно вместе с причинами. Меньше retries бывает от хорошего первого выбора, а бывает от преждевременного отказа. Больше evidence бывает от полноценного поиска, а бывает от дублей. Сравнивают outcome и затраты по типам задач, включая no-results, conflicting versions, malformed call и exhausted budget. Такая проверка помогает выбрать изменение prompt/template/retriever/runtime по месту ошибки, а не менять всё приложение сразу.

<!-- /unit:b11bb3a3eb2c4be5f5 -->

<!-- unit:630390e42ffd0e51ef -->
<a id="u-630390e42ffd0e51ef"></a>

## Независимый evaluation set

<!-- /unit:630390e42ffd0e51ef -->

<!-- unit:37bd90d9cb5db33c3c -->
<a id="u-37bd90d9cb5db33c3c"></a>

Нужны questions с user intent/constraints, допустимыми sources/versions, qrels для retrieval и rubric/answers для end-to-end. Для tool tasks — initial state, allowed actions, expected final state и oracle/checks. Train/eval разделяются по source/group/time согласно deployment task, а не только random rows.

<!-- /unit:37bd90d9cb5db33c3c -->

<!-- unit:fa91821da795c9ce30 -->
<a id="u-fa91821da795c9ce30"></a>

Включить answerable/unanswerable, exact terminology/paraphrases, rare entities, misleading/stale documents, contradictory sources и permissions. Если проверить только простые FAQ с буквальным совпадением query/doc, получится узкая уверенность, а не доказательство general quality.

<!-- /unit:fa91821da795c9ce30 -->

<!-- unit:86f80f65e9df9d9a23 -->
<a id="u-86f80f65e9df9d9a23"></a>

Gold answers тоже проверяются: несколько formulations могут быть допустимы, а source может обновиться. Eval snapshot/revision нужно сохранять, чтобы model regression не путать с изменением test ground truth.

<!-- /unit:86f80f65e9df9d9a23 -->

<!-- unit:b60486331f326bf330 -->
<a id="u-b60486331f326bf330"></a>

**Схема курса:** fixed query set + qrels/oracle → retrieval metrics → evidence coverage → answer/citation correctness → end-to-end task success + cost/error slice

<!-- /unit:b60486331f326bf330 -->

<!-- unit:5071b254bfbb3b06f1 -->
<a id="u-5071b254bfbb3b06f1"></a>

<!-- source-depth-applications-Д -->

<!-- /unit:5071b254bfbb3b06f1 -->

<!-- unit:6b81c7ed29215ad2d7 -->
<a id="u-6b81c7ed29215ad2d7"></a>

### Единица оценки: query, chunk, claim, call или задача

<!-- /unit:6b81c7ed29215ad2d7 -->

<!-- unit:748ccb7fa396ae04a5 -->
<a id="u-748ccb7fa396ae04a5"></a>

В S6 раздельная evaluation мотивируется просто: плохой final answer не сообщает, где произошёл сбой. Но и набор отдельных процентов без общих единиц не объясняет результат. Один user task может содержать два вопроса, три searches, пять passages и десять factual claims. Accuracy по calls и success по tasks имеют разные denominators. Сначала определяют единицу решения и критерий успеха, затем собирают labels.

<!-- /unit:748ccb7fa396ae04a5 -->

<!-- unit:658f74b302c1fcb47b -->
<a id="u-658f74b302c1fcb47b"></a>

Для документационного запроса полезен record: original question, service/version/time constraints, permitted scope, answerability, достаточные evidence spans, обязательные claims и правила допустимых формулировок. Для tool task добавляют initial environment state, разрешённые операции и expected final state. Raw transcript сохраняют как trace; gold label не следует строить только по красивому финальному сообщению.

<!-- /unit:658f74b302c1fcb47b -->

<!-- unit:9292d08bf9bb3c5395 -->
<a id="u-9292d08bf9bb3c5395"></a>

«Где timeout v8?» и «сравни timeout v7/v8 и объясни причину изменения» —разные задачи. Во второй нужен набор evidence и объяснение, а не один найденный scalar. Наличие хотя бы одного relevant chunk не означает покрытие обоих версий/причины. Dataset должен отражать реальный класс вопросов, включая сложность и недостающие данные, а не только повторять короткий training FAQ.

<!-- /unit:9292d08bf9bb3c5395 -->

<!-- unit:3c37f36d9dc2bb18b3 -->
<a id="u-3c37f36d9dc2bb18b3"></a>

### Qrels и разметка answerability

<!-- /unit:3c37f36d9dc2bb18b3 -->

<!-- unit:f81ca426e225798aa0 -->
<a id="u-f81ca426e225798aa0"></a>

Qrels связывают query с оценкой relevance document/chunk. Нужно определить granularity: relevant parent document может содержать ответ, но конкретный child chunk —не содержать. Для graded labels можно различать sufficient evidence, полезный background и unrelated content. Числа grades получают смысл из rubric; выбор gain function для nDCG задают отдельно.

<!-- /unit:f81ca426e225798aa0 -->

<!-- unit:03f94628ffc3b13820 -->
<a id="u-03f94628ffc3b13820"></a>

Отсутствие qrel не равно установленной нерелевантности. Если annotator увидел только top10 старого BM25, новый dense retriever может найти хороший passage вне этого пула и получить формальный penalty. Pooling нескольких systems, независимый human review и аудит disagreements уменьшают этот blind spot. Не надо незаметно разметить только победы новой системы после просмотра её score: annotation policy должна быть симметричной.

<!-- /unit:03f94628ffc3b13820 -->

<!-- unit:4473c521709ae15e00 -->
<a id="u-4473c521709ae15e00"></a>

Answerability относится к объявленному доступному corpus/scope/revision. Документ мог существовать в общей базе, но быть недоступным конкретному user; для grounded ответа это другая постановка. Отсутствие evidence после tool timeout нельзя разметить как «ответа в corpus нет»: результат поиска не получен. Нужны отдельные labels для unknown execution и established unanswerable.

<!-- /unit:4473c521709ae15e00 -->

<!-- unit:f385713889012d252c -->
<a id="u-f385713889012d252c"></a>

В source предлагаются ручная разметка, LLM judge и implicit clicks. Каждый способ имеет ошибки. Click зависит от позиции выдачи, UI и привычки пользователя, а отсутствие click не доказывает irrelevance. Judge может оценивать правдоподобие вместо соответствия source. Human annotator тоже может пропустить условие; полезны инструкции, примеры пограничных случаев, повторная разметка subset и adjudication спорных labels.

<!-- /unit:f385713889012d252c -->

<!-- unit:6efdb1e9a7bc450b24 -->
<a id="u-6efdb1e9a7bc450b24"></a>

### Holdout, contamination и что значит независимость

<!-- /unit:6efdb1e9a7bc450b24 -->

<!-- unit:3ccfa7ed2b857c993f -->
<a id="u-3ccfa7ed2b857c993f"></a>

Random split paraphrases одного вопроса легко оставляет почти ту же задачу в train и test. Split по документам помогает, если будущая задача —новые documents; split по времени —если важны будущие revisions. Разные customers/task families могут требовать group split. Для нескольких questions одного source ответы часто зависимы; это важно и при статистическом сравнении.

<!-- /unit:3ccfa7ed2b857c993f -->

<!-- unit:11b0d048f2caa6d74d -->
<a id="u-11b0d048f2caa6d74d"></a>

S3 предупреждает о benchmark contamination. Совпадения и memorization действительно могут завысить score, но высокая оценка сама по себе не доказывает ни reasoning, ни leakage. Свежее задание уменьшает часть риска, а не гарантирует отсутствие повторного использования при последующем подборе prompt/adapter. Скрытый final holdout нужно отделять от development set, на котором принимаются решения.

<!-- /unit:11b0d048f2caa6d74d -->

<!-- unit:9c7e7573ad836c06ce -->
<a id="u-9c7e7573ad836c06ce"></a>

«Инвертировать benchmark и посмотреть, сломалась ли модель» не является надёжным бинарным тестом на memorization. Обратная задача может быть недоопределена или объективно труднее: по одному ответу 3 нельзя однозначно восстановить исходное уравнение. Такие perturbations полезны как дополнительные probes с собственным oracle, но не доказывают причину изменения качества автоматически.

<!-- /unit:9c7e7573ad836c06ce -->

<!-- unit:1cce522c53d05c4728 -->
<a id="u-1cce522c53d05c4728"></a>

## Retrieval metrics

<!-- /unit:1cce522c53d05c4728 -->

<!-- unit:e902e7b1a5fde64045 -->
<a id="u-e902e7b1a5fde64045"></a>

При наборе relevant documents $R_q$ и первых $k$ найденных $S_{q,k}$:

<!-- /unit:e902e7b1a5fde64045 -->

<!-- unit:c99081c6af033904ca -->
<a id="u-c99081c6af033904ca"></a>

$$
\operatorname{Recall@}k=\frac{|S_{q,k}\cap R_q|}{|R_q|},\qquad
\operatorname{Precision@}k=\frac{|S_{q,k}\cap R_q|}{k}.
$$

<!-- /unit:c99081c6af033904ca -->

<!-- unit:d9a895cedad71c03f3 -->
<a id="u-d9a895cedad71c03f3"></a>

Если qrels содержит два relevant chunks, а top3 нашёл один, recall $1/2$, precision $1/3$. Если нужна хотя бы одна достаточная passage, отдельная hit@k измеряет наличие такого passage; это не та же самая доля всех relevant chunks.

<!-- /unit:d9a895cedad71c03f3 -->

<!-- unit:4d889aac7ee0e69517 -->
<a id="u-4d889aac7ee0e69517"></a>

Reciprocal rank — $1/r$, где $r$ rank первого relevant item; при отсутствии — 0 по выбранному cutoff. MRR усредняет по queries. При ranks $(1,3,\infty)$ получаем:

<!-- /unit:4d889aac7ee0e69517 -->

<!-- unit:493bf1f47439349316 -->
<a id="u-493bf1f47439349316"></a>

$$
\operatorname{MRR}=\frac{1+\frac13+0}{3}=\frac49\approx0.4444.
$$

<!-- /unit:493bf1f47439349316 -->

<!-- unit:4febf9bac683c91320 -->
<a id="u-4febf9bac683c91320"></a>

Для graded relevance используют DCG/nDCG. Объявим convention:

<!-- /unit:4febf9bac683c91320 -->

<!-- unit:e7b4eb2f931b5a22e1 -->
<a id="u-e7b4eb2f931b5a22e1"></a>

$$
\operatorname{DCG@}k=\sum_{i=1}^{k}\frac{2^{g_i}-1}{\log_2(i+1)},\qquad
\operatorname{nDCG@}k=\frac{\operatorname{DCG@}k}{\operatorname{IDCG@}k}.
$$

<!-- /unit:e7b4eb2f931b5a22e1 -->

<!-- unit:cbcaead7619112f8e6 -->
<a id="u-cbcaead7619112f8e6"></a>

Для grades $(2,0,1)$ DCG $3+0+1/2=3.5$, ideal order $(2,1,0)$ даёт $3+1/\log_2 3\approx3.63093$, nDCG около $0.96394$. Если denominator нулевой, handling queries без relevant results определяется protocol.

<!-- /unit:cbcaead7619112f8e6 -->

<!-- unit:483d781236f0268a10 -->
<a id="u-483d781236f0268a10"></a>

Incomplete qrels не доказывает irrelevance каждого неразмеченного document. Human audit retrieval disagreements полезен, особенно при смене lexical/dense method.

<!-- /unit:483d781236f0268a10 -->

<!-- unit:63d8d5413c8ba6edf4 -->
<a id="u-63d8d5413c8ba6edf4"></a>

### Hit, recall и достаточность evidence

<!-- /unit:63d8d5413c8ba6edf4 -->

<!-- unit:042e7196883b3f2acf -->
<a id="u-042e7196883b3f2acf"></a>

В обычном binary hit@k проверяется наличие хотя бы одного relevant item, по выбранному определению relevance. Для RAG можно дополнительно определить sufficient-evidence hit, когда найдено достаточно материала для ответа. Эти определения нельзя подменять: multi-hop вопрос требует двух passages A и B, top-k содержит только A. Binary relevant-item hit=1, recall по двум passages=1/2, но sufficient-set success=0.

<!-- /unit:042e7196883b3f2acf -->

<!-- unit:b4bba9cfa06f622e72 -->
<a id="u-b4bba9cfa06f622e72"></a>

Если после parent mapping пять child hits относятся к одному document, оценка unique sources отличается от vector/chunk hit count. Требуется дедупликация по объявленной identity. Дублировать релевантный chunk в выдаче десять раз нельзя считать десятью найденными элементами gold set. У разных versions тоже должна быть однозначная политика identity и relevance.

<!-- /unit:b4bba9cfa06f622e72 -->

<!-- unit:29396172cc2c53ac7a -->
<a id="u-29396172cc2c53ac7a"></a>

Если система вернула меньше k результатов, denominator precision выбирают заранее: фиксированный k штрафует отсутствие выдачи, returned-count precision отвечает другому вопросу. Для qrels с R=0 recall по обычной формуле не определён; protocol может исключать такие queries из retrieval recall и оценивать abstention отдельно. Молча присвоить 0 или 1 меняет смысл macro average.

<!-- /unit:29396172cc2c53ac7a -->

<!-- unit:fd9859be23a1a709d8 -->
<a id="u-fd9859be23a1a709d8"></a>

### Macro и micro отвечают на разные вопросы

<!-- /unit:fd9859be23a1a709d8 -->

<!-- unit:2f5bf032875dcadb38 -->
<a id="u-2f5bf032875dcadb38"></a>

Пусть Q1 имеет один relevant document и найден он; Q2 —девять relevant documents, найден один. Per-query recalls1 и 1/9. Macro recall —$(1+1/9)/2=5/9\approx0.555556$. Micro recall по всем gold items —$(1+1)/(1+9)=0.2$. Это не ошибка вычислений: первая величина одинаково взвешивает queries, вторая —gold items.

<!-- /unit:2f5bf032875dcadb38 -->

<!-- unit:5dc3c7e184797b1156 -->
<a id="u-5dc3c7e184797b1156"></a>

Для user-facing задач macro часто полезна как среднее по запросам; для покрытия массива evidence micro сообщает другое. Если query difficulty и число relevant passages различаются, один average скрывает это. В отчёте указывают aggregation и slices: exact IDs, paraphrases, rare entities, versions, multi-hop и unanswerable.

<!-- /unit:5dc3c7e184797b1156 -->

<!-- unit:a9ed5c4e79d92d4f5a -->
<a id="u-a9ed5c4e79d92d4f5a"></a>

### MRR не средняя позиция и не полнота выдачи

<!-- /unit:a9ed5c4e79d92d4f5a -->

<!-- unit:b41fb46a8c5f2abb6a -->
<a id="u-b41fb46a8c5f2abb6a"></a>

В S6 MRR описан как «на каком месте в среднем первый релевантный». Точнее: усредняется reciprocal rank, а не rank. Для рангов 1,3 и отсутствия по cutoff MRR4/9, как посчитано выше. Инверсия среднего и среднее инверсий —разные операции; отсутствие не следует подменять произвольно огромным rank без изменения protocol.

<!-- /unit:b41fb46a8c5f2abb6a -->

<!-- unit:0d96b696abf4557551 -->
<a id="u-0d96b696abf4557551"></a>

Если обе системы ставят первый relevant passage на rank1, MRR у обеих 1, даже когда вторая теряет все остальные нужные documents. Для двух relevant items на positions1 и 3 average precision при denominator2 равна $(1+2/3)/2=5/6$. Для positions1 и 4 AP=(1+2/4)/2=0.75, хотя RR остаётся 1. MAP усредняет AP по queries; cutoff/handling incompleteness нужно объявить. [Учебник information retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) рассматривает эти различные способы оценки ranked results.

<!-- /unit:0d96b696abf4557551 -->

<!-- unit:bd68f49dceee01e255 -->
<a id="u-bd68f49dceee01e255"></a>

### nDCG: labels, gain и идеальный порядок

<!-- /unit:bd68f49dceee01e255 -->

<!-- unit:1faaa2f40fdab284af -->
<a id="u-1faaa2f40fdab284af"></a>

В формуле этой главы grade2 даёт gain3, grade1 —gain1. Удвоение grade не означает удвоение gain. Другой protocol может использовать linear gain g; тогда численное nDCG изменится. Поэтому одна строка «nDCG=0.9» без cutoff, gain convention и qrels недостаточна для воспроизведения.

<!-- /unit:1faaa2f40fdab284af -->

<!-- unit:047c170e21744377a6 -->
<a id="u-047c170e21744377a6"></a>

IDCG строится по допустимому идеальному порядку gold grades на том же cutoff. Если использовать maximum только среди уже retrieved items, система, пропустившая лучший document, может выглядеть неоправданно хорошо. При incomplete qrels идеальность ограничена известными labels; это ещё одна причина аудитировать новые retrieval methods.

<!-- /unit:047c170e21744377a6 -->

<!-- unit:5b7212bc7972cae5a2 -->
<a id="u-5b7212bc7972cae5a2"></a>

Сортировка релевантных items выше обычно повышает DCG, но metric не знает, что два отдельных low-grade passages вместе дают достаточное доказательство. Для multi-hop/context packing полезны дополнительные coverage/set-based checks. nDCG по candidate list не заменяет оценку того, что generator реально прочитал.

<!-- /unit:5b7212bc7972cae5a2 -->

<!-- unit:8199be5d45a9bbf17f -->
<a id="u-8199be5d45a9bbf17f"></a>

## Correctness, groundedness и citations

<!-- /unit:8199be5d45a9bbf17f -->

<!-- unit:cd7ac115bce005cfca -->
<a id="u-cd7ac115bce005cfca"></a>

**Correctness:** отвечает ли result на вопрос верно. **Groundedness:** поддерживают ли переданные sources claims. **Citation correctness:** ведут ли ссылки к нужным passages и подтверждают ли конкретный claim. **Coverage/completeness:** не пропущены ли существенные части ответа.

<!-- /unit:cd7ac115bce005cfca -->

<!-- unit:a43500a7795188c206 -->
<a id="u-a43500a7795188c206"></a>

Модель может дать правильный факт из памяти с неправильной citation; grounded answer по устаревшему документу может быть неверным для текущего вопроса; несколько citations могут не поддерживать главный вывод. Эти cases требуют разных labels, а не одного «есть ссылки».

<!-- /unit:a43500a7795188c206 -->

<!-- unit:f533a7490cf9ef1f18 -->
<a id="u-f533a7490cf9ef1f18"></a>

Unknown/abstention оценивается с учётом answerability. Отказ без причины на answerable question — ошибка coverage; уверенный ответ без evidence на unanswerable — hallucination/unsupported claim. Нельзя улучшить factual score, просто отказываясь на всех сложных задачах, если denominator скрывает отказы.

<!-- /unit:f533a7490cf9ef1f18 -->

<!-- unit:910666ce3b0200ab7f -->
<a id="u-910666ce3b0200ab7f"></a>

### Четыре разных свойства одного ответа

<!-- /unit:910666ce3b0200ab7f -->

<!-- unit:2283756b7f78dbb1ab -->
<a id="u-2283756b7f78dbb1ab"></a>

Допустим, запрос —timeout payment-service v8. Текущий source D8 говорит 5s, старый D7 —2s. Ответ «5s» из памяти может быть correct, но citation D7 его не поддерживает. Ответ «2s» с точной citation D7 faithful старому passage, но не соответствует v8. Ответ «5s, это уменьшает latency на 30%» добавляет неподтверждённый claim, даже если первое число правильно.

<!-- /unit:2283756b7f78dbb1ab -->

<!-- unit:1eba518c3388571bdd -->
<a id="u-1eba518c3388571bdd"></a>

Groundedness оценивает поддержку утверждений переданным context, factual/task correctness —правильность по задаче и применимому truth, citation correctness —правильность конкретной связи claim→source. Наличие URL и успешный HTTP response проверяют только часть provenance. Link может вести на нужный документ, но не на passage с доказательством, или быть недоступным в scope пользователя.

<!-- /unit:1eba518c3388571bdd -->

<!-- unit:9139194a4ea5069e20 -->
<a id="u-9139194a4ea5069e20"></a>

Completeness задаётся required aspects. Для вопроса о Transformer «что такое attention» не покрывает обучение, masks, positions, residual path и отличия CV variants, если они требуются задачей. Длинный ответ также не гарантирует полноту: можно повторить одно определение вместо объяснения механизмов. Rubric должна проверять перечисленные aspects и связь между ними.

<!-- /unit:9139194a4ea5069e20 -->

<!-- unit:d4b1e88c9dc9765d53 -->
<a id="u-d4b1e88c9dc9765d53"></a>

### Claim-level citations и denominators

<!-- /unit:d4b1e88c9dc9765d53 -->

<!-- unit:b9bb16cb796ed2792b -->
<a id="u-b9bb16cb796ed2792b"></a>

Пусть в ответе пять factual assertions; четыре снабжены citation, но только три пары действительно подтверждаются указанным source. Citation correctness по отмеченным pairs=3/4=0.75. Доля всех assertions с корректной cited support=3/5=0.6. Процент наличия citation4/5=0.8 —ещё третья величина. Нужно также решить, как учитывать несколько citations у одного claim и одну citation у нескольких claims.

<!-- /unit:b9bb16cb796ed2792b -->

<!-- unit:6186b5deeb4f31f64e -->
<a id="u-6186b5deeb4f31f64e"></a>

Отдельно оценим completeness: gold requires A,B,C,D, а ответ содержит A,B,C,E,F. Покрыто 3/4 required facts; два дополнительных claims могут быть верными либо unsupported, но не заменяют отсутствующий D. Claim count не следует оптимизировать как самостоятельную цель. Если annotator разбил одну длинную фразу на десять мелких утверждений, denominator изменится —granularity rubric должна быть стабильной.

<!-- /unit:6186b5deeb4f31f64e -->

<!-- unit:7d641a32e3fe2f8e12 -->
<a id="u-7d641a32e3fe2f8e12"></a>

Для механических полей можно проверить actual source ID, version и string/number span. Для entailment часто требуется содержательная оценка: source «до 5 секунд при условии X» не равен безусловному «всегда 5 секунд». Перевод единиц/арифметический вывод может быть корректным при отсутствии буквального совпадения; validator должен учитывать допустимые transformations, а не требовать только exact strings.

<!-- /unit:7d641a32e3fe2f8e12 -->

<!-- unit:dfaf1ba461e610675e -->
<a id="u-dfaf1ba461e610675e"></a>

### Abstention, coverage и цена уверенной ошибки

<!-- /unit:dfaf1ba461e610675e -->

<!-- unit:23c8f05bcd0b4a207b -->
<a id="u-23c8f05bcd0b4a207b"></a>

Учебный набор 100 задач:80 answerable и 20 unanswerable в данном corpus/scope. Система отвечает на 70:65 из answerable и 5 из unanswerable. Из ответов 60 correct; остальные 10 incorrect/unsupported. Conditional correctness среди выданных ответов 60/70≈85.71%, coverage70%. На 15 оставшихся unanswerable система корректно отказалась.

<!-- /unit:23c8f05bcd0b4a207b -->

<!-- unit:134a188228fb904193 -->
<a id="u-134a188228fb904193"></a>

Если protocol считает успехом correct answer или правильный отказ на unanswerable, total task success=(60+15)/100=75%. На answerable корректно решено 60/80=75%; доля корректных отказов на unanswerable15/20=75%. Это не та же metric, что 85.71% среди ответов. Всегда отказывающая система имела бы 20% task success в этом наборе, а conditional answer correctness была бы не определена.

<!-- /unit:134a188228fb904193 -->

<!-- unit:72dc24f25204b4fdd5 -->
<a id="u-72dc24f25204b4fdd5"></a>

Изменение порога ответа меняет coverage и error rate среди принятых ответов. Высокая precision при маленьком coverage может быть подходящим trade-off, если такова задача, но её нельзя скрыто выдавать за универсальное улучшение. Confidence score модели требует калибровки на сопоставимых данных; красивая фраза «уверенность 0.99» не даёт статистической гарантии.

<!-- /unit:72dc24f25204b4fdd5 -->

<!-- unit:60ba39382f5da4dec1 -->
<a id="u-60ba39382f5da4dec1"></a>

## Oracle context и разложение pipeline failures

<!-- /unit:60ba39382f5da4dec1 -->

<!-- unit:3330f53df56cf79887 -->
<a id="u-3330f53df56cf79887"></a>

Сравнить no retrieval, actual retrieved context и gold/oracle context при одинаковом generator/config. Если actual fail, oracle correct — evidence pipeline вероятный bottleneck. Если oracle тоже fail — причина может быть в instructions, reasoning, context interpretation или task specification.

<!-- /unit:3330f53df56cf79887 -->

<!-- unit:19a74c39e018644b02 -->
<a id="u-19a74c39e018644b02"></a>

Нужно проверить и context assembly: relevant document может быть в top20, но не попасть в final prompt из-за reranker, dedup или token truncation. Поэтому сохранять candidate IDs/ranks, selected snippets и фактический model input. Успех retriever по большому $k$ не доказывает, что generator увидел evidence.

<!-- /unit:19a74c39e018644b02 -->

<!-- unit:0cd6a4b7d4e978136c -->
<a id="u-0cd6a4b7d4e978136c"></a>

Для tools отделять selected-tool accuracy, argument correctness, execution success и final task success. Валидный call к search может быть бесполезным; HTTP200 не гарантирует правильное изменение состояния; модель могла верно выполнить action, но неверно сообщить result.

<!-- /unit:0cd6a4b7d4e978136c -->

<!-- unit:25dddc8bbb17f269f1 -->
<a id="u-25dddc8bbb17f269f1"></a>

### Воронка evidence без предположения независимости

<!-- /unit:25dddc8bbb17f269f1 -->

<!-- unit:c33c52ee9866ebe015 -->
<a id="u-c33c52ee9866ebe015"></a>

Пусть из 100 задач нужный passage найден в candidates у 90, после filtering/rerank/packing реально передан у 80, grounded correct answer получен у 72. Conditional rates:90/100,80/90,72/80. Их произведение 0.72 —точное разложение для этих вложенных success events; независимость stages не нужна. Но если correctness допускает правильный ответ из памяти без evidence, множество outcomes уже не обязано быть вложенным, и такая воронка не описывает все correct answers.

<!-- /unit:c33c52ee9866ebe015 -->

<!-- unit:c351d2bd0d28975af9 -->
<a id="u-c351d2bd0d28975af9"></a>

Сохраняют source extraction, candidate IDs/scores, eligible/selected IDs, actual prompt и final labels. Если passage был найден, но исчез после truncation, менять encoder не обязательно. Если его нет даже в exact search, нужно различить отсутствие в corpus, representation mismatch и query intent. Если passage есть в prompt, но нарушено условие версии, проблема может быть в interpretation или неоднозначном packaging.

<!-- /unit:c351d2bd0d28975af9 -->

<!-- unit:b8c98bc3f07979807f -->
<a id="u-b8c98bc3f07979807f"></a>

### Oracle context —контролируемое вмешательство

<!-- /unit:b8c98bc3f07979807f -->

<!-- unit:8d24a086665df0cf3a -->
<a id="u-8d24a086665df0cf3a"></a>

Сравнивают no retrieval, actual context и oracle passages при одинаковых generator/template/decoding и сопоставимом formatting. Oracle не должен содержать скрытую готовую эталонную формулировку там, где реальный источник даёт только premises. Изменение количества distractors и length тоже влияет на generation; «дали один идеальный sentence вместо 20pages» —несколько изменений одновременно.

<!-- /unit:8d24a086665df0cf3a -->

<!-- unit:6eb1f94b19ac501abb -->
<a id="u-6eb1f94b19ac501abb"></a>

На 100 paired queries actual success60, oracle80. Например,55 успешны в обоих режимах,25 исправились только с oracle,5 были успешны только actual,15 неуспешны в обоих. Net gain20points скрывает 30 changed outcomes. Не все 25 улучшений можно механически назвать «только ошибка retriever»: oracle также изменил context composition. Эти cases просматривают по trace, а не присваивают одну причину из итогового процента.

<!-- /unit:6eb1f94b19ac501abb -->

<!-- unit:9517f6d966bfcdf15f -->
<a id="u-9517f6d966bfcdf15f"></a>

Для packing ablation оставляют те же retrieved items, но гарантированно включают relevant passage. Для generator ablation фиксируют одни observations и меняют модель/инструкцию. Для retriever ablation фиксируют corpus и downstream protocol. Изолированное изменение помогает интерпретации, хотя interactions stages тоже существуют.

<!-- /unit:9517f6d966bfcdf15f -->

<!-- unit:b6722d69850ae6e4d0 -->
<a id="u-b6722d69850ae6e4d0"></a>

### Tool/environment evaluation

<!-- /unit:b6722d69850ae6e4d0 -->

<!-- unit:ba91b85391792be222 -->
<a id="u-ba91b85391792be222"></a>

У call оценивают name, аргументы, scope, допустимость момента вызова и обработку observation. У task —конечное состояние и то, верно ли сообщён результат. Запись `create_report` может пройти schema, выполниться с timeout после успешного effect и завершиться повтором, создавшим duplicate. Если oracle проверяет только «хотя бы один report есть», он пропустит эту ошибку.

<!-- /unit:ba91b85391792be222 -->

<!-- unit:c09931e7f8e236c019 -->
<a id="u-c09931e7f8e236c019"></a>

Read-only stub позволяет детерминированно подставить no-results, timeout, denied и inconsistent-version results. Но happy-path stub, всегда возвращающий готовый ответ, не оценивает recovery. Для mutation задают initial state и инварианты: нужная операция совершена, лишние эффекты отсутствуют, request/intent identity не потеряна. Проверка final prose —дополнительная, а не замена environment oracle.

<!-- /unit:c09931e7f8e236c019 -->

<!-- unit:03cd5997e2c958a29b -->
<a id="u-03cd5997e2c958a29b"></a>

Loop cap и terminal status также входят в outcome. Невыполненная задача после max_steps —failure/partial по rubric, а не автоматический success из-за достижения END. Сессии, которые упали до ответа, включаются в task denominator; удаление exceptions из evaluation создаёт искусственно высокую точность.

<!-- /unit:03cd5997e2c958a29b -->

<!-- unit:a6d818fe48eb8bc9ec -->
<a id="u-a6d818fe48eb8bc9ec"></a>

## Стоимость, latency и reproducibility

<!-- /unit:a6d818fe48eb8bc9ec -->

<!-- unit:45bcd4377aca48a7f6 -->
<a id="u-45bcd4377aca48a7f6"></a>

Считать total input/output tokens, retrieval/reranker/model/tool calls, retries и latency components. Для набора задач полезна стоимость успешного решения:

<!-- /unit:45bcd4377aca48a7f6 -->

<!-- unit:97f0e1010442edd26f -->
<a id="u-97f0e1010442edd26f"></a>

$$
C_{\mathrm{success}}=
\frac{\text{общая стоимость всех попыток}}{\text{число успешно решённых задач}}.
$$

<!-- /unit:97f0e1010442edd26f -->

<!-- unit:06e19a3be6f8c2580b -->
<a id="u-06e19a3be6f8c2580b"></a>

Если 100 attempts стоили 20 условных units и 80 успешны, получается $0.25$ unit/success. Деление только стоимости successful attempts исключило бы wasted work и занизило реальную цену. При нуле successes ratio не определён/бесконечен по report convention, а не равен нулю.

<!-- /unit:06e19a3be6f8c2580b -->

<!-- unit:388e7662a25b8a548d -->
<a id="u-388e7662a25b8a548d"></a>

Зафиксировать model/template/tokenizer, corpus/index revision, encoder/pooling, reranker, retrieval settings, prompt budget, decoding и tool stubs/environment. Не приписывать улучшение model, если одновременно расширили corpus и дали вдвое больше calls.

<!-- /unit:388e7662a25b8a548d -->

<!-- unit:8f7c2bb4bbc14b0e2e -->
<a id="u-8f7c2bb4bbc14b0e2e"></a>

### First attempt и успех за бюджет

<!-- /unit:8f7c2bb4bbc14b0e2e -->

<!-- unit:4af4110515604710d1 -->
<a id="u-4af4110515604710d1"></a>

Success на первой попытке и после максимум трёх attempts —разные показатели. Даже IID model с p=0.2 даёт probability хотя бы одного успеха за 3 попытки $1-0.8^3=0.488$. Это выигрыш test-time compute, а не доказательство, что веса стали лучше. В реальности attempts могут быть зависимыми, а verifier неидеальным, поэтому formula не подставляется автоматически.

<!-- /unit:4af4110515604710d1 -->

<!-- unit:09359996f3fa1bc956 -->
<a id="u-09359996f3fa1bc956"></a>

Если сгенерировано n=10 candidates и oracle считает c=2 правильными, при равномерном выборе subset из k=3 без возвращения доля subset, содержащих хотя бы один correct, равна $1-\binom{8}{3}/\binom{10}{3}=8/15\approx0.533333$. Это finite-pool подсчёт, не то же самое, что гарантированная способность реального verifier выбрать correct candidate. Для применения pass@k нужно объявить sampling/selection protocol и доступность oracle.

<!-- /unit:09359996f3fa1bc956 -->

<!-- unit:08c19dd18548b7a09f -->
<a id="u-08c19dd18548b7a09f"></a>

На 100 user tasks может быть 140 model attempts,80 task successes и суммарная цена 20units. Тогда cost per successful task0.25, attempts per task1.4. Делиться нужно на 80 успешных задач, а не на число отдельных «хороших messages». В цену входят failed attempts, tools/retrieval/reranking, verifier и retries по выбранному accounting. Если сравнивать системы, обе получают одинаковое budget правило или явно различаются на cost-quality curve.

<!-- /unit:08c19dd18548b7a09f -->

<!-- unit:ed587628f7b14270b4 -->
<a id="u-ed587628f7b14270b4"></a>

### Latency: все сессии и явная quantile convention

<!-- /unit:ed587628f7b14270b4 -->

<!-- unit:1cd10d66551a8b5294 -->
<a id="u-1cd10d66551a8b5294"></a>

Допустим,95 из 100 задач завершаются за 1s, пять —за 10s. Mean1.45s. При nearest-rank p95 берёт 95-й sorted result и равен 1s, p99 —10s. Другая interpolation convention может дать другое число на этой границе; она указывается в protocol. Один p95 способен скрыть ровно такой tail, поэтому смотрят несколько quantiles и timeout/failure rates.

<!-- /unit:1cd10d66551a8b5294 -->

<!-- unit:8c89cdf5d5fbfde014 -->
<a id="u-8c89cdf5d5fbfde014"></a>

Если пять долгих запросов завершились timeout, убрать их из latency статистики и сказать «всё всегда 1s» —selection bias. User-visible ожидание и failure остаются частью опыта. Для streaming отдельно измеряют time-to-first-token и full-answer latency; ранний token не означает раннее завершение reasoning/tool task. Для parallel stages wall time и сумма resource time также различаются.

<!-- /unit:8c89cdf5d5fbfde014 -->

<!-- unit:78beecad5b71ef5157 -->
<a id="u-78beecad5b71ef5157"></a>

### Замороженный experiment record

<!-- /unit:78beecad5b71ef5157 -->

<!-- unit:e870738097bd28db07 -->
<a id="u-e870738097bd28db07"></a>

Фиксируются dataset/qrels revision и split, source snapshot/ACL policy, encoder/query/document normalization, index/search parameters, fusion/reranker, actual prompt packing, model/adapter/template/tokenizer, decoding, environment/stubs и budget. Также сохраняют judge/rubric version и правила aggregation. Изменение любого из них может изменить measured outcome.

<!-- /unit:e870738097bd28db07 -->

<!-- unit:2a2531ef4424b7bd50 -->
<a id="u-2a2531ef4424b7bd50"></a>

Стартовая sequence эксперимента: baseline → одно обоснованное изменение → paired error review → regression checks по slices. Если одновременно увеличить corpus, заменить encoder, добавить reranker и удвоить output budget, можно оценить всю новую систему, но нельзя приписать весь прирост одной выбранной части. Записать конфигурацию —не то же самое, что гарантировать bitwise deterministic GPU execution; это основа интерпретируемой воспроизводимости.

<!-- /unit:2a2531ef4424b7bd50 -->

<!-- unit:74bc5ffc18ec369992 -->
<a id="u-74bc5ffc18ec369992"></a>

## Judge и uncertainty

<!-- /unit:74bc5ffc18ec369992 -->

<!-- unit:b7d10c915d1ca6e95e -->
<a id="u-b7d10c915d1ca6e95e"></a>

LLM judge получает rubric и sources, а names outputs скрыты; менять порядок для парных сравнений, допускаются ties. Проверять human subset и систематические disagreements по длине/стилю. Для объективных fields/state использовать deterministic oracle, если он корректен и достаточно полон.

<!-- /unit:b7d10c915d1ca6e95e -->

<!-- unit:13b4be6c419804488a -->
<a id="u-13b4be6c419804488a"></a>

Paired bootstrap по query groups оценивает uncertainty различия systems. Repeated seeds/decoding runs показывают stochastic variation, но не устраняют dataset bias. Один case useful для debugging, а не достаточное основание universal ranking.

<!-- /unit:13b4be6c419804488a -->

<!-- unit:089ba6f0cd3b43cb1e -->
<a id="u-089ba6f0cd3b43cb1e"></a>

### Что сохраняем и исправляем из S3 evaluation

<!-- /unit:089ba6f0cd3b43cb1e -->

<!-- unit:2263f3669693e36037 -->
<a id="u-2263f3669693e36037"></a>

S3 правильно требует оценки после SFT/DPO/quantization/merge, но итог «perplexity мертва» неверен. Held-out NLL/PPL измеряет качество предсказания выбранных target tokens при заданных данных/tokenizer/masking. Он полезен для language modeling и regression diagnostics; просто не равен instruction/task success. Неверный factual target token получает penalty, поэтому фраза «loss вообще не увидит ошибку внутри JSON» слишком сильна. Однако много лёгких format tokens может скрыть важность редких ошибок в aggregate average.

<!-- /unit:2263f3669693e36037 -->

<!-- unit:d353194366d11fbbbc -->
<a id="u-d353194366d11fbbbc"></a>

Не доказано и «instruction following никак не коррелирует с PPL»: универсальное отсутствие корреляции не следует из различия задач. Верно, что низкая PPL на сыром corpus не гарантирует успешный agent/RAG workflow. Human preference полезна для открытых ответов, но не единственная измеримая ценность: state correctness, recovery и реальные task outcomes тоже важны. Фиксированные 500–1000 prompts —пример, а не автоматическое достаточное число для любой ошибки/разницы.

<!-- /unit:d353194366d11fbbbc -->

<!-- unit:f4693c942a9ff43633 -->
<a id="u-f4693c942a9ff43633"></a>

Alignment tax —возможный trade-off, а не неизбежное падение reasoning после любого DPO. Нужно измерять before/after на независимых benign/task slices, false refusals и нужных критериях. Падение 80%→72% означает 8 процентных пунктов или 10% относительно исходного уровня; это разные обозначения. Source порог «10–15% всегда слишком много» не является универсальным decision rule. Подробности adaptation evaluation —в [отдельной главе](<../03-adaptation/adapt-evaluation.md>).

<!-- /unit:f4693c942a9ff43633 -->

<!-- unit:a8966b4486b838fbd4 -->
<a id="u-a8966b4486b838fbd4"></a>

### Judge rubric, position и blind spots

<!-- /unit:a8966b4486b838fbd4 -->

<!-- unit:c2f824069f2dc8851f -->
<a id="u-c2f824069f2dc8851f"></a>

Judge задают вопрос, применимые sources и конкретные criteria. Names систем скрывают, порядок outputs меняют, допускают ties и uncertain labels. Но swapping не устраняет все ошибки: judge может стабильно выбирать более длинный unsupported ответ в любом position. Материалы [MT-Bench/Chatbot Arena study](https://arxiv.org/abs/2306.05685) разбирают position, verbosity и self-enhancement biases; это основание проверять их, а не обещание фиксированного human agreement в новом домене.

<!-- /unit:c2f824069f2dc8851f -->

<!-- unit:653ba15dfdc3684b3f -->
<a id="u-653ba15dfdc3684b3f"></a>

У citation task judge должен видеть passages, у environment task —достаточную информацию о final state. Судья, видящий только prose «операция выполнена», не может надёжно подтвердить настоящий effect. Для точных чисел/IDs и state invariants лучше использовать корректный deterministic oracle, когда он доступен. Judge rationale тоже может быть ошибочным и не является независимым доказательством.

<!-- /unit:653ba15dfdc3684b3f -->

<!-- unit:a2c9486aaf550a4fb9 -->
<a id="u-a2c9486aaf550a4fb9"></a>

В 30 парных сравнениях 12 wins,10 losses,8 ties. Strict win fraction12/30=0.4; half-tie score(12+4)/30≈0.533333; win fraction среди decided12/22≈0.545455. Одна надпись win-rate без tie policy неоднозначна. Missing/failed judge outputs не стоит молча считать ties: это отдельный outcome и возможный источник selection bias.

<!-- /unit:a2c9486aaf550a4fb9 -->

<!-- unit:97054a16cb7ee9febe -->
<a id="u-97054a16cb7ee9febe"></a>

### RAGAS, ARES и reference-free не означают truth-free

<!-- /unit:97054a16cb7ee9febe -->

<!-- unit:3c3f22cf6ca8c190e2 -->
<a id="u-3c3f22cf6ca8c190e2"></a>

В [Ragas paper](https://arxiv.org/abs/2309.15217) оценка разделяет качество извлечённого context, faithful использование passages и ответ. [ARES](https://arxiv.org/abs/2311.09476) исследует judges для context relevance, answer faithfulness и answer relevance с human-annotated data для корректировки оценки. Это разные frameworks/методы, а не два названия одного идеального oracle. Их library APIs и конкретные metrics versions проверяют отдельно.

<!-- /unit:3c3f22cf6ca8c190e2 -->

<!-- unit:f3df0314cd652c01ce -->
<a id="u-f3df0314cd652c01ce"></a>

Reference-free обычно означает отсутствие обязательной ручной эталонной строки ответа для выбранной metric. Всё равно остаются question/context, learned judge и допущения о критерии. Хороший proxy score проверяют на human subset и известных failure cases. Context precision относится к оценке evidence, а не к одному только стилю генерации; одинаковое название metric в разных implementations не гарантирует одинаковую формулу.

<!-- /unit:f3df0314cd652c01ce -->

<!-- unit:b873d5831d5d19de92 -->
<a id="u-b873d5831d5d19de92"></a>

Source рекомендация «достаточно 50 production edge-case prompts и json.loads» подходит как smoke/debug набор для части format failures. Она не доказывает agent reliability: корректный JSON может вызвать неверный tool или неверно обработать observation. И пользователь уточнил теоретический режим —production dataset и запуск такого benchmark ему не приписываются.

<!-- /unit:b873d5831d5d19de92 -->

<!-- unit:4e2bb11d1d51a8f24a -->
<a id="u-4e2bb11d1d51a8f24a"></a>

### Paired comparison и uncertainty

<!-- /unit:4e2bb11d1d51a8f24a -->

<!-- unit:e3b7d409e985825523 -->
<a id="u-e3b7d409e985825523"></a>

Пусть на 100 одинаковых queries новая система решила 70, baseline65. Совместная разметка:55 успехов у обеих,15 только у новой,10 только у baseline,20 неуспехов у обеих. Difference5points. Отдельные totals скрывают 25 disagreement cases; именно их просмотр объяснит, какие ошибки изменились. Пять дополнительных успехов сами по себе ещё не доказывают устойчивый общий перевес.

<!-- /unit:e3b7d409e985825523 -->

<!-- unit:958f1e85e58686ce3a -->
<a id="u-958f1e85e58686ce3a"></a>

Paired bootstrap выбирает queries с возвращением и для каждого выбранного ID берёт outcomes **обеих** систем, затем считает difference. Если независимо resample old и new results, потеряется pairing. Если несколько queries относятся к одному документу/сессии, resample можно делать на уровне таких groups, чтобы не считать зависимые paraphrases независимыми observations. Конкретный sampling unit выбирают по задаче.

<!-- /unit:958f1e85e58686ce3a -->

<!-- unit:4c6c0f5a8486b2d341 -->
<a id="u-4c6c0f5a8486b2d341"></a>

Интервал bootstrap отражает variation от resampling данного набора при выбранных assumptions. Он не обнаружит неизвестные deployment topics, систематически неверные labels и утечку test в prompt tuning. Multiple seeds измеряют stochastic run variation, а не заменяют новые независимые задачи. Повторно подбирать метод по final test и затем представлять тот же score как независимую проверку —ещё один канал переобучения.

<!-- /unit:4c6c0f5a8486b2d341 -->

<!-- unit:1b6eb61f2d9a852ec6 -->
<a id="u-1b6eb61f2d9a852ec6"></a>

Учебный итоговый experiment record должен позволять восстановить denominator, labels, traces, configs, criterion success и затраты. Пока есть только расчёты и проект протокола, это так и называется; model quality или superiority конкретного deployed pipeline не заявляются без реального измерения.

<!-- /unit:1b6eb61f2d9a852ec6 -->

<!-- unit:c021e2e07a616ade80 -->
<a id="u-c021e2e07a616ade80"></a>

### Вопросы для самопроверки

<!-- /unit:c021e2e07a616ade80 -->

<!-- unit:9f327054a89c9bcc6e -->
<a id="u-9f327054a89c9bcc6e"></a>

1. Обучает ли graph routing веса? **Ответ:** нет, без отдельного training процесса.
2. Передаёт ли ordinary RAG vector напрямую generator? **Ответ:** обычно нет, vector выбирает IDs, затем подаётся текст evidence.
3. Обязан ли каждый AutoGen agent быть LangGraph graph? **Ответ:** нет, это разные уровни/реализации loop/coordination.
4. Есть ли универсальный отдельный tool head? **Ответ:** нет; распространён autoregressive вывод протокола через vocabulary head, а отдельный router —возможное дополнительное решение.
5. Делает ли SFT format гарантированным? **Ответ:** нет. Обученная вероятность, grammar constraint и проверка смысла/прав —разные механизмы.
6. Почему checkpoint не обеспечивает exactly-once? **Ответ:** внешний effect мог произойти до сохранения результата; retry требует deduplication или recovery policy.
7. Кто решает, что поиска достаточно? **Ответ:** модель предлагает finish по контексту, runtime применяет заданные критерии и лимиты; полнота evidence не возникает автоматически.
8. Почему global top-k и поздний scope filter могут вернуть пустоту? **Ответ:** допустимый релевантный chunk мог находиться ниже global cutoff.
9. Обязан ли graph node вызывать LLM? **Ответ:** нет, это может быть обычная функция или tool executor.
10. Чем parallelism отличается от сокращения работы? **Ответ:** он может сократить критический путь, сохранив общие tokens и вычисления.
11. Что означает хороший tool-call validity при плохом task success? **Ответ:** правильный формат не гарантирует нужный инструмент, evidence и верную интерпретацию.

<!-- /unit:9f327054a89c9bcc6e -->

<!-- unit:39cec7354052eb1dfb -->
<a id="u-39cec7354052eb1dfb"></a>

**Практика:** на бумаге проследить запрос «найди правило таймаута и процитируй» через scope→retrieval→tool observation→answer→validation. Добавить no-results, timeout и forbidden-document случаи. Более подробный прикладной материал/исходные вопросы — в [разделе приложений](<../../archive/2026-10-20-before-rebuild/06-llm/04-applications/План — приложения LLM.md>).

<!-- /unit:39cec7354052eb1dfb -->

<!-- unit:a570ec1e95fb75c475 -->
<a id="u-a570ec1e95fb75c475"></a>

## Итоговая самопроверка

<!-- /unit:a570ec1e95fb75c475 -->

<!-- unit:d53c3774cce122a867 -->
<a id="u-d53c3774cce122a867"></a>

1. Чем hit@k отличается от recall@k по всем relevant documents? **Ответ:** binary hit — наличие хотя бы одного relevant item, recall — доля relevant items; достаточность полного набора evidence оценивают дополнительно.
2. Почему gold context полезен? **Ответ:** отделяет часть upstream failures от generation при доступном evidence.
3. Что доказывает schema validity? **Ответ:** только соответствующий структурный слой, не final task success.

<!-- /unit:d53c3774cce122a867 -->

<!-- unit:a6108a7c2f5853622b -->
<a id="u-a6108a7c2f5853622b"></a>

4. Равны ли MRR и обратная средняя позиция? **Ответ:** нет, усреднение и инверсия не перестановочны.
5. Почему macro и micro recall различаются? **Ответ:** macro одинаково взвешивает queries, micro —gold items по выбранной единице.
6. Достаточен ли один relevant hit для multi-hop ответа? **Ответ:** не обязательно, может требоваться набор passages.
7. Что делать с unjudged document? **Ответ:** не объявлять установленно нерелевантным без оговорки неполноты qrels.
8. Как always-abstain может выглядеть хорошо? **Ответ:** если conditional correctness скрывает отсутствие ответов; нужны coverage и full task denominator.
9. Доказывает ли oracle gain исключительно retrieval error? **Ответ:** нет, меняется context; необходим trace/controlled ablation.
10. Почему success@3 нельзя сравнить с first-attempt success как чистый model gain? **Ответ:** отличаются compute budget и selection protocol.
11. Значит ли reference-free отсутствие внешних assumptions? **Ответ:** нет, остаются judges, context и выбранный критерий.
12. Почему paired bootstrap сохраняет один query ID для обеих систем? **Ответ:** это сохраняет зависимость результатов на одной задаче.

<!-- /unit:a6108a7c2f5853622b -->

<!-- unit:99c60e1d97a942a8c0 -->
<a id="u-99c60e1d97a942a8c0"></a>

**Практика:** собрать 12 учебных questions с qrels/answerability, три tool traces с ошибками и явную rubric. Вручную посчитать Recall/MRR/nDCG examples; затем спроектировать baseline→одно изменение→error review. Большой production project не требуется для теоретической проверки, но реальные metrics можно заявлять только после запуска.

<!-- /unit:99c60e1d97a942a8c0 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как локализовать ошибку приложения и измерить стоимость успешной задачи?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
