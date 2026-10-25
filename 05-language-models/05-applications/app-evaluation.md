# Оценка retrieval, ответов и task success

**Уровень:** основа ветки.

**Предпосылки:** [RAG: BM25, dense, hybrid и multi-vector retrieval](<app-rag.md>)

**Используют:** [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>)

**Учебный вопрос:** Как локализовать ошибку приложения и измерить стоимость успешной задачи?

## Карта вычисления и границы этапов

![Оценка retrieval, ответов и task success: входы, преобразования и выходы](<../../оформление/course/app-evaluation.svg>)

Оценка приложения связывает компоненты с исходной пользовательской задачей. Отдельные проценты retrieval (поиск кандидатов) и JSON validity не складываются автоматически в вероятность успешного завершения.

## Как разделить ошибки

Retrieval failure: evidence не найдено. Context (доступный модели контекст) selection failure: найдено, но не передано/обрезано. Generation failure: evidence есть, ответ неверный. Tool (инструмент приложения) failure: wrong tool/args или ошибка исполнения. State (состояние системы) failure: потерян результат/ID, повторено действие. Такие категории позволяют выбрать intervention точнее, чем «агент тупит».

Oracle context experiment подаёт gold snippet вместо retrieval. Если ответ теперь correct, bottleneck вероятно upstream; если нет — prompt (входной запрос или инструкция)/model/evidence interpretation требуют разбора. JSON validity и tool HTTP success — локальные diagnostics, final task success оценивается отдельно.

### Разбираем одну задачу до terminal outcome

Учебная задача: «найди timeout payment-service v8 и процитируй». Trusted state разрешает только проект P. Первый поиск возвращает snippet v7 с timeout 2 s. Это observation (доступное наблюдение среды) реального поиска, но ещё не ответ на вопрос v8. Модель предлагает уточнить version, второй поиск возвращает v8 с timeout 5 s и ID `P/pay/v8#timeout`. После чтения полного условия ответ указывает 5 s и ограничение применимости. Validator проверяет наличие ID среди observations и нужную version; содержательную корректность числа сверяют с текстом.

Если второй поиск пустой, ответ «в v8 тоже 2 s» не следует из первого результата. Корректно сказать, что найдена только v7 и значение для v8 не установлено. Если tool timeout, в state хранится ошибка исполнения, а не no-results: мы не знаем, существовал ли документ. Если найденный ID недоступен, он не становится разрешённым из-за уверенности модели. Эти три случая требуют разных observations и разных дальнейших решений.

Полезная trace-запись содержит исходную задачу, revision prompt/model/tool registry, model-call index, call ID, validated args, scope decision, result/error, evidence IDs и terminal reason. Длинный лог всех строк сам по себе не заменяет связи между ними. Сводный статус success при незавершённом pending call скрывает ошибку жизненного цикла.

### Oracle experiments и локальные метрики

Чтобы проверить retrieval, вручную подставляют нужный v8 snippet. Если модель всё равно отвечает 2 s, проблема не исчерпывается поиском. Если теперь ответ верный, upstream становится основным подозреваемым, но один успешный пример не доказывает, что generator всегда надёжен. Затем отдельно проверяют context packaging: возможно, retriever (компонент поиска) уже находил v8, но truncation удалял именно его.

На 100 учебных задачах 90 calls могут быть schema-valid, 80 —к правильному tool, 70 —успешно исполненными, 60 —дающими нужный evidence и 55 —завершёнными корректным ответом. Если это вложенные множества одной трассы на задачу, validity 90% и end-to-end success 55% совместимы. Условный success после найденного evidence равен 55/60≈91.67%. При нескольких calls на задачу denominators уже нужно определять отдельно: долю валидных calls нельзя напрямую считать долей успешных tasks.

Число retries полезно вместе с причинами. Меньше retries бывает от хорошего первого выбора, а бывает от преждевременного отказа. Больше evidence бывает от полноценного поиска, а бывает от дублей. Сравнивают outcome и затраты по типам задач, включая no-results, conflicting versions, malformed call и exhausted budget. Такая проверка помогает выбрать изменение prompt/template (шаблон сериализации входа)/retriever/runtime (среда исполнения модели) по месту ошибки, а не менять всё приложение сразу.

## Независимый evaluation set

Нужны questions с user intent/constraints, допустимыми sources/versions, qrels для retrieval и rubric/answers для end-to-end. Для tool tasks — initial state, allowed actions, expected final state и oracle/checks. Train/eval разделяются по source/group/time согласно deployment task, а не только random rows.

Включить answerable/unanswerable, exact terminology/paraphrases, rare entities, misleading/stale documents, contradictory sources и permissions. Если проверить только простые FAQ с буквальным совпадением query (запрос)/doc, получится узкая уверенность, а не доказательство general quality.

Gold answers тоже проверяются: несколько formulations могут быть допустимы, а source может обновиться. Eval snapshot/revision нужно сохранять, чтобы model regression не путать с изменением test ground truth (эталонная разметка).

**Схема курса:** fixed query set + qrels/oracle → retrieval metrics → evidence coverage → answer/citation correctness → end-to-end task success + cost/error slice



### Единица оценки: query, chunk, claim, call или задача

Раздельная оценка компонентов нужна потому, что плохой final answer не сообщает, где произошёл сбой. Но и набор отдельных процентов без общих единиц не объясняет результат. Один user task может содержать два вопроса, три searches, пять passages и десять factual claims. Accuracy по calls и success по tasks имеют разные denominators. Сначала определяют единицу решения и критерий успеха, затем собирают labels (метки).

Для документационного запроса полезен record: original question, service/version/time constraints, permitted scope, answerability, достаточные evidence spans, обязательные claims и правила допустимых формулировок. Для tool task добавляют initial environment (среда взаимодействия) state, разрешённые операции и expected final state. Raw transcript сохраняют как trace; gold label (метка) не следует строить только по красивому финальному сообщению.

«Где timeout v8?» и «сравни timeout v7/v8 и объясни причину изменения» —разные задачи. Во второй нужен набор evidence и объяснение, а не один найденный scalar. Наличие хотя бы одного relevant chunk не означает покрытие обоих версий/причины. Dataset должен отражать реальный класс вопросов, включая сложность и недостающие данные, а не только повторять короткий training FAQ.

### Qrels и разметка answerability

Qrels связывают query с оценкой relevance document (документ)/chunk. Нужно определить granularity: relevant parent document может содержать ответ, но конкретный child chunk —не содержать. Для graded labels можно различать sufficient evidence, полезный background и unrelated content. Числа grades получают смысл из rubric; выбор gain function для nDCG задают отдельно.

Отсутствие qrel не равно установленной нерелевантности. Если annotator увидел только top10 старого BM25, новый dense retriever может найти хороший passage вне этого пула и получить формальный penalty. Pooling (агрегирование состояний) нескольких systems, независимый human review и аудит disagreements уменьшают этот blind spot. Не надо незаметно разметить только победы новой системы после просмотра её score (оценка): annotation policy должна быть симметричной.

Answerability относится к объявленному доступному corpus/scope/revision. Документ мог существовать в общей базе, но быть недоступным конкретному user; для grounded ответа это другая постановка. Отсутствие evidence после tool timeout нельзя разметить как «ответа в corpus нет»: результат поиска не получен. Нужны отдельные labels для unknown execution и established unanswerable.

В source предлагаются ручная разметка, LLM judge и implicit clicks. Каждый способ имеет ошибки. Click зависит от позиции выдачи, UI и привычки пользователя, а отсутствие click не доказывает irrelevance. Judge может оценивать правдоподобие вместо соответствия source. Human annotator тоже может пропустить условие; полезны инструкции, примеры пограничных случаев, повторная разметка subset и adjudication спорных labels.

### Holdout (отложенная часть данных), contamination и что значит независимость

Random split (разбиение данных) paraphrases одного вопроса легко оставляет почти ту же задачу в train и test. Split по документам помогает, если будущая задача —новые documents; split по времени —если важны будущие revisions. Разные customers/task families могут требовать group split. Для нескольких questions одного source ответы часто зависимы; это важно и при статистическом сравнении.

Загрязнение тестов обучающими данными искажает оценку. Совпадения и memorization действительно могут завысить score, но высокая оценка сама по себе не доказывает ни reasoning, ни leakage (утечка недоступной информации). Свежее задание уменьшает часть риска, а не гарантирует отсутствие повторного использования при последующем подборе prompt/adapter. Скрытый final holdout нужно отделять от development set, на котором принимаются решения.

«Инвертировать benchmark и посмотреть, сломалась ли модель» не является надёжным бинарным тестом на memorization. Обратная задача может быть недоопределена или объективно труднее: по одному ответу 3 нельзя однозначно восстановить исходное уравнение. Такие perturbations полезны как дополнительные probes с собственным oracle, но не доказывают причину изменения качества автоматически.

## Retrieval metrics

При наборе relevant documents $R_q$ и первых $k$ найденных $S_{q,k}$:

$$
\operatorname{Recall@}k=\frac{|S_{q,k}\cap R_q|}{|R_q|},\qquad
\operatorname{Precision@}k=\frac{|S_{q,k}\cap R_q|}{k}.
$$

Если qrels содержит два relevant chunks, а top3 нашёл один, recall $1/2$, precision $1/3$. Если нужна хотя бы одна достаточная passage, отдельная hit@k измеряет наличие такого passage; это не та же самая доля всех relevant chunks.

Reciprocal rank — $1/r$, где $r$ rank первого relevant item; при отсутствии — 0 по выбранному cutoff. MRR усредняет по queries. При ranks $(1,3,\infty)$ получаем:

$$
\operatorname{MRR}=\frac{1+\frac13+0}{3}=\frac49\approx0.4444.
$$

Для graded relevance используют DCG/nDCG. Объявим convention:

$$
\operatorname{DCG@}k=\sum_{i=1}^{k}\frac{2^{g_i}-1}{\log_2(i+1)},\qquad
\operatorname{nDCG@}k=\frac{\operatorname{DCG@}k}{\operatorname{IDCG@}k}.
$$

Для grades $(2,0,1)$ DCG $3+0+1/2=3.5$, ideal order $(2,1,0)$ даёт $3+1/\log_2 3\approx3.63093$, nDCG около $0.96394$. Если denominator нулевой, handling queries без relevant results определяется protocol.

Incomplete qrels не доказывает irrelevance каждого неразмеченного document. Human audit retrieval disagreements полезен, особенно при смене lexical/dense method.

### Hit, recall и достаточность evidence

В обычном binary hit@k проверяется наличие хотя бы одного relevant item, по выбранному определению relevance. Для RAG можно дополнительно определить sufficient-evidence hit, когда найдено достаточно материала для ответа. Эти определения нельзя подменять: multi-hop вопрос требует двух passages A и B, top-k содержит только A. Binary relevant-item hit=1, recall по двум passages=1/2, но sufficient-set success=0.

Если после parent mapping пять child hits относятся к одному document, оценка unique sources отличается от vector/chunk hit count. Требуется дедупликация по объявленной identity. Дублировать релевантный chunk в выдаче десять раз нельзя считать десятью найденными элементами gold set. У разных versions тоже должна быть однозначная политика identity и relevance.

Если система вернула меньше k результатов, denominator precision выбирают заранее: фиксированный k штрафует отсутствие выдачи, returned-count precision отвечает другому вопросу. Для qrels с R=0 recall по обычной формуле не определён; protocol может исключать такие queries из retrieval recall и оценивать abstention отдельно. Молча присвоить 0 или 1 меняет смысл macro average.

### Macro и micro отвечают на разные вопросы

Пусть Q1 имеет один relevant document и найден он; Q2 —девять relevant documents, найден один. Per-query recalls1 и 1/9. Macro recall —$(1+1/9)/2=5/9\approx0.555556$. Micro recall по всем gold items —$(1+1)/(1+9)=0.2$. Это не ошибка вычислений: первая величина одинаково взвешивает queries, вторая —gold items.

Для user-facing задач macro часто полезна как среднее по запросам; для покрытия массива evidence micro сообщает другое. Если query difficulty и число relevant passages различаются, один average скрывает это. В отчёте указывают aggregation и slices: exact IDs, paraphrases, rare entities, versions, multi-hop и unanswerable.

### MRR не средняя позиция и не полнота выдачи

MRR нельзя определять как среднее место первого релевантного результата. Точнее: усредняется reciprocal rank, а не rank. Для рангов 1,3 и отсутствия по cutoff MRR4/9, как посчитано выше. Инверсия среднего и среднее инверсий —разные операции; отсутствие не следует подменять произвольно огромным rank без изменения protocol.

Если обе системы ставят первый relevant passage на rank1, MRR у обеих 1, даже когда вторая теряет все остальные нужные documents. Для двух relevant items на positions1 и 3 average precision при denominator2 равна $(1+2/3)/2=5/6$. Для positions1 и 4 AP=(1+2/4)/2=0.75, хотя RR остаётся 1. MAP усредняет AP по queries; cutoff/handling incompleteness нужно объявить. [Учебник information retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) рассматривает эти различные способы оценки ranked results.

### nDCG: labels, gain и идеальный порядок

В формуле этой главы grade2 даёт gain3, grade1 —gain1. Удвоение grade не означает удвоение gain. Другой protocol может использовать linear gain g; тогда численное nDCG изменится. Поэтому одна строка «nDCG=0.9» без cutoff, gain convention и qrels недостаточна для воспроизведения.

IDCG строится по допустимому идеальному порядку gold grades на том же cutoff. Если использовать maximum только среди уже retrieved items, система, пропустившая лучший document, может выглядеть неоправданно хорошо. При incomplete qrels идеальность ограничена известными labels; это ещё одна причина аудитировать новые retrieval methods.

Сортировка релевантных items выше обычно повышает DCG, но metric не знает, что два отдельных low-grade passages вместе дают достаточное доказательство. Для multi-hop/context packing полезны дополнительные coverage/set-based checks. nDCG по candidate list не заменяет оценку того, что generator реально прочитал.

## Correctness, groundedness и citations

**Correctness:** отвечает ли result на вопрос верно. **Groundedness:** поддерживают ли переданные sources claims. **Citation correctness:** ведут ли ссылки к нужным passages и подтверждают ли конкретный claim. **Coverage/completeness:** не пропущены ли существенные части ответа.

Модель может дать правильный факт из памяти с неправильной citation; grounded answer по устаревшему документу может быть неверным для текущего вопроса; несколько citations могут не поддерживать главный вывод. Эти cases требуют разных labels, а не одного «есть ссылки».

Unknown/abstention оценивается с учётом answerability. Отказ без причины на answerable question — ошибка coverage; уверенный ответ без evidence на unanswerable — hallucination/unsupported claim. Нельзя улучшить factual score, просто отказываясь на всех сложных задачах, если denominator скрывает отказы.

### Четыре разных свойства одного ответа

Допустим, запрос —timeout payment-service v8. Текущий source D8 говорит 5s, старый D7 —2s. Ответ «5s» из памяти может быть correct, но citation D7 его не поддерживает. Ответ «2s» с точной citation D7 faithful старому passage, но не соответствует v8. Ответ «5s, это уменьшает latency (задержка отдельного запроса) на 30%» добавляет неподтверждённый claim, даже если первое число правильно.

Groundedness оценивает поддержку утверждений переданным context, factual/task correctness —правильность по задаче и применимому truth, citation correctness —правильность конкретной связи claim→source. Наличие URL и успешный HTTP response проверяют только часть provenance. Link может вести на нужный документ, но не на passage с доказательством, или быть недоступным в scope пользователя.

Completeness задаётся required aspects. Для вопроса о Transformer «что такое attention (механизм внимания)» не покрывает обучение, masks, positions, residual path и отличия CV variants, если они требуются задачей. Длинный ответ также не гарантирует полноту: можно повторить одно определение вместо объяснения механизмов. Rubric должна проверять перечисленные aspects и связь между ними.

### Claim-level citations и denominators

Пусть в ответе пять factual assertions; четыре снабжены citation, но только три пары действительно подтверждаются указанным source. Citation correctness по отмеченным pairs=3/4=0.75. Доля всех assertions с корректной cited support=3/5=0.6. Процент наличия citation4/5=0.8 —ещё третья величина. Нужно также решить, как учитывать несколько citations у одного claim и одну citation у нескольких claims.

Отдельно оценим completeness: gold requires A,B,C,D, а ответ содержит A,B,C,E,F. Покрыто 3/4 required facts; два дополнительных claims могут быть верными либо unsupported, но не заменяют отсутствующий D. Claim count не следует оптимизировать как самостоятельную цель. Если annotator разбил одну длинную фразу на десять мелких утверждений, denominator изменится —granularity rubric должна быть стабильной.

Для механических полей можно проверить actual source ID, version и string/number span. Для entailment часто требуется содержательная оценка: source «до 5 секунд при условии X» не равен безусловному «всегда 5 секунд». Перевод единиц/арифметический вывод может быть корректным при отсутствии буквального совпадения; validator должен учитывать допустимые transformations, а не требовать только exact strings.

### Abstention, coverage и цена уверенной ошибки

Учебный набор 100 задач:80 answerable и 20 unanswerable в данном corpus/scope. Система отвечает на 70:65 из answerable и 5 из unanswerable. Из ответов 60 correct; остальные 10 incorrect/unsupported. Conditional correctness среди выданных ответов 60/70≈85.71%, coverage70%. На 15 оставшихся unanswerable система корректно отказалась.

Если protocol считает успехом correct answer или правильный отказ на unanswerable, total task success=(60+15)/100=75%. На answerable корректно решено 60/80=75%; доля корректных отказов на unanswerable15/20=75%. Это не та же metric, что 85.71% среди ответов. Всегда отказывающая система имела бы 20% task success в этом наборе, а conditional answer correctness была бы не определена.

Изменение порога ответа меняет coverage и error rate среди принятых ответов. Высокая precision при маленьком coverage может быть подходящим trade-off, если такова задача, но её нельзя скрыто выдавать за универсальное улучшение. Confidence score модели требует калибровки на сопоставимых данных; красивая фраза «уверенность 0.99» не даёт статистической гарантии.

## Oracle context и разложение pipeline failures

Сравнить no retrieval, actual retrieved context и gold/oracle context при одинаковом generator/config. Если actual fail, oracle correct — evidence pipeline вероятный bottleneck. Если oracle тоже fail — причина может быть в instructions, reasoning, context interpretation или task specification.

Нужно проверить и context assembly: relevant document может быть в top20, но не попасть в final prompt из-за reranker, dedup или token (токен) truncation. Поэтому сохранять candidate IDs/ranks, selected snippets и фактический model input. Успех retriever по большому $k$ не доказывает, что generator увидел evidence.

Для tools отделять selected-tool accuracy, argument correctness, execution success и final task success. Валидный call к search может быть бесполезным; HTTP200 не гарантирует правильное изменение состояния; модель могла верно выполнить action, но неверно сообщить result.

### Воронка evidence без предположения независимости

Пусть из 100 задач нужный passage найден в candidates у 90, после filtering/rerank/packing реально передан у 80, grounded correct answer получен у 72. Conditional rates:90/100,80/90,72/80. Их произведение 0.72 —точное разложение для этих вложенных success events; независимость stages не нужна. Но если correctness допускает правильный ответ из памяти без evidence, множество outcomes уже не обязано быть вложенным, и такая воронка не описывает все correct answers.

Сохраняют source extraction, candidate IDs/scores (оценки), eligible/selected IDs, actual prompt и final labels. Если passage был найден, но исчез после truncation, менять encoder (кодировщик входа) не обязательно. Если его нет даже в exact search, нужно различить отсутствие в corpus, representation mismatch и query intent. Если passage есть в prompt, но нарушено условие версии, проблема может быть в interpretation или неоднозначном packaging.

### Oracle context —контролируемое вмешательство

Сравнивают no retrieval, actual context и oracle passages при одинаковых generator/template/decoding и сопоставимом formatting. Oracle не должен содержать скрытую готовую эталонную формулировку там, где реальный источник даёт только premises. Изменение количества distractors и length тоже влияет на generation; «дали один идеальный sentence вместо 20pages» —несколько изменений одновременно.

На 100 paired queries actual success60, oracle80. Например,55 успешны в обоих режимах,25 исправились только с oracle,5 были успешны только actual,15 неуспешны в обоих. Net gain20points скрывает 30 changed outcomes. Не все 25 улучшений можно механически назвать «только ошибка retriever»: oracle также изменил context composition. Эти cases просматривают по trace, а не присваивают одну причину из итогового процента.

Для packing ablation (сравнение с удалённой или изменённой частью метода) оставляют те же retrieved items, но гарантированно включают relevant passage. Для generator ablation фиксируют одни observations и меняют модель/инструкцию. Для retriever ablation фиксируют corpus и downstream protocol. Изолированное изменение помогает интерпретации, хотя interactions stages тоже существуют.

### Tool/environment evaluation

У call оценивают name, аргументы, scope, допустимость момента вызова и обработку observation. У task —конечное состояние и то, верно ли сообщён результат. Запись `create_report` может пройти schema (схема допустимых данных), выполниться с timeout после успешного effect и завершиться повтором, создавшим duplicate. Если oracle проверяет только «хотя бы один report есть», он пропустит эту ошибку.

Read-only stub позволяет детерминированно подставить no-results, timeout, denied и inconsistent-version results. Но happy-path stub, всегда возвращающий готовый ответ, не оценивает recovery. Для mutation задают initial state и инварианты: нужная операция совершена, лишние эффекты отсутствуют, request/intent identity не потеряна. Проверка final prose —дополнительная, а не замена environment oracle.

Loop cap и terminal status также входят в outcome. Невыполненная задача после max_steps —failure/partial по rubric, а не автоматический success из-за достижения END. Сессии, которые упали до ответа, включаются в task denominator; удаление exceptions из evaluation создаёт искусственно высокую точность.

## Стоимость, latency и reproducibility

Считать total input/output tokens (токены), retrieval/reranker/model/tool calls, retries и latency components. Для набора задач полезна стоимость успешного решения:

$$
C_{\mathrm{success}}=
\frac{\text{общая стоимость всех попыток}}{\text{число успешно решённых задач}}.
$$

Если 100 attempts стоили 20 условных units и 80 успешны, получается $0.25$ unit/success. Деление только стоимости successful attempts исключило бы wasted work и занизило реальную цену. При нуле successes ratio не определён/бесконечен по report convention, а не равен нулю.

Зафиксировать model/template/tokenizer (токенизатор), corpus/index revision, encoder/pooling, reranker, retrieval settings, prompt budget, decoding и tool stubs/environment. Не приписывать улучшение model, если одновременно расширили corpus и дали вдвое больше calls.

### First attempt и успех за бюджет

Success на первой попытке и после максимум трёх attempts —разные показатели. Даже IID model с p=0.2 даёт probability хотя бы одного успеха за 3 попытки $1-0.8^3=0.488$. Это выигрыш test-time compute, а не доказательство, что веса стали лучше. В реальности attempts могут быть зависимыми, а verifier неидеальным, поэтому formula не подставляется автоматически.

Если сгенерировано n=10 candidates и oracle считает c=2 правильными, при равномерном выборе subset из k=3 без возвращения доля subset, содержащих хотя бы один correct, равна $1-\binom{8}{3}/\binom{10}{3}=8/15\approx0.533333$. Это finite-pool подсчёт, не то же самое, что гарантированная способность реального verifier выбрать correct candidate. Для применения pass@k нужно объявить sampling (случайный выбор по распределению)/selection protocol и доступность oracle.

На 100 user tasks может быть 140 model attempts,80 task successes и суммарная цена 20units. Тогда cost per successful task0.25, attempts per task1.4. Делиться нужно на 80 успешных задач, а не на число отдельных «хороших messages». В цену входят failed attempts, tools/retrieval/reranking (повторное ранжирование кандидатов), verifier и retries по выбранному accounting. Если сравнивать системы, обе получают одинаковое budget правило или явно различаются на cost-quality curve.

### Latency: все сессии и явная quantile convention

Допустим,95 из 100 задач завершаются за 1s, пять —за 10s. Mean1.45s. При nearest-rank p95 берёт 95-й sorted result и равен 1s, p99 —10s. Другая interpolation convention может дать другое число на этой границе; она указывается в protocol. Один p95 способен скрыть ровно такой tail, поэтому смотрят несколько quantiles и timeout/failure rates.

Если пять долгих запросов завершились timeout, убрать их из latency статистики и сказать «всё всегда 1s» —selection bias. User-visible ожидание и failure остаются частью опыта. Для streaming отдельно измеряют time-to-first-token и full-answer latency; ранний token не означает раннее завершение reasoning/tool task. Для parallel stages wall time и сумма resource time также различаются.

### Замороженный experiment record

Фиксируются dataset/qrels revision и split, source snapshot/ACL policy, encoder/query/document normalization (нормировка по выбранному правилу), index/search parameters (параметры), fusion/reranker, actual prompt packing, model/adapter/template/tokenizer, decoding, environment/stubs и budget. Также сохраняют judge/rubric version и правила aggregation. Изменение любого из них может изменить measured outcome.

Стартовая sequence эксперимента: baseline (базовый вариант сравнения) → одно обоснованное изменение → paired error review → regression checks по slices. Если одновременно увеличить corpus, заменить encoder, добавить reranker и удвоить output budget, можно оценить всю новую систему, но нельзя приписать весь прирост одной выбранной части. Записать конфигурацию —не то же самое, что гарантировать bitwise deterministic GPU execution; это основа интерпретируемой воспроизводимости.

## Judge и uncertainty

LLM judge получает rubric и sources, а names outputs скрыты; менять порядок для парных сравнений, допускаются ties. Проверять human subset и систематические disagreements по длине/стилю. Для объективных fields/state использовать deterministic oracle, если он корректен и достаточно полон.

Paired bootstrap по query groups оценивает uncertainty различия systems. Repeated seeds/decoding runs показывают stochastic variation, но не устраняют dataset bias. Один case useful для debugging, а не достаточное основание universal ranking.

### Оценка после изменения модели

После SFT, обучения предпочтениям, квантования и объединения нужна повторная оценка. Из этого не следует бесполезность perplexity. Held-out NLL/PPL измеряет качество предсказания выбранных target tokens при заданных данных/tokenizer/masking. Он полезен для language modeling и regression diagnostics; просто не равен instruction/task success. Неверный factual target token получает penalty, поэтому фраза «loss (функция потерь) вообще не увидит ошибку внутри JSON» слишком сильна. Однако много лёгких format tokens может скрыть важность редких ошибок в aggregate average.

Не доказано и «instruction following никак не коррелирует с PPL»: универсальное отсутствие корреляции не следует из различия задач. Верно, что низкая PPL на сыром corpus не гарантирует успешный agent/RAG workflow. Human preference полезна для открытых ответов, но не единственная измеримая ценность: state correctness, recovery и реальные task outcomes тоже важны. Фиксированные 500–1000 prompts —пример, а не автоматическое достаточное число для любой ошибки/разницы.

Alignment tax —возможный trade-off, а не неизбежное падение reasoning после любого DPO. Нужно измерять before/after на независимых benign/task slices, false refusals и нужных критериях. Падение 80%→72% означает 8 процентных пунктов или 10% относительно исходного уровня; это разные обозначения. Source порог «10–15% всегда слишком много» не является универсальным decision rule. Подробности adaptation evaluation —в [отдельной главе](<../03-adaptation/adapt-evaluation.md>).

### Judge rubric, position и blind spots

Judge задают вопрос, применимые sources и конкретные criteria. Names систем скрывают, порядок outputs меняют, допускают ties и uncertain labels. Но swapping не устраняет все ошибки: judge может стабильно выбирать более длинный unsupported ответ в любом position. Материалы [MT-Bench/Chatbot Arena study](https://arxiv.org/abs/2306.05685) разбирают position, verbosity и self-enhancement biases; это основание проверять их, а не обещание фиксированного human agreement в новом домене.

У citation task judge должен видеть passages, у environment task —достаточную информацию о final state. Судья, видящий только prose «операция выполнена», не может надёжно подтвердить настоящий effect. Для точных чисел/IDs и state invariants лучше использовать корректный deterministic oracle, когда он доступен. Judge rationale тоже может быть ошибочным и не является независимым доказательством.

В 30 парных сравнениях 12 wins,10 losses,8 ties. Strict win fraction12/30=0.4; half-tie score(12+4)/30≈0.533333; win fraction среди decided12/22≈0.545455. Одна надпись win-rate без tie policy неоднозначна. Missing/failed judge outputs не стоит молча считать ties: это отдельный outcome и возможный источник selection bias.

### RAGAS, ARES и reference-free не означают truth-free

В [Ragas paper](https://arxiv.org/abs/2309.15217) оценка разделяет качество извлечённого context, faithful использование passages и ответ. [ARES](https://arxiv.org/abs/2311.09476) исследует judges для context relevance, answer faithfulness и answer relevance с human-annotated data для корректировки оценки. Это разные frameworks/методы, а не два названия одного идеального oracle. Их library APIs и конкретные metrics versions проверяют отдельно.

Reference-free обычно означает отсутствие обязательной ручной эталонной строки ответа для выбранной metric. Всё равно остаются question/context, learned judge и допущения о критерии. Хороший proxy score проверяют на human subset и известных failure cases. Context precision относится к оценке evidence, а не к одному только стилю генерации; одинаковое название metric в разных implementations не гарантирует одинаковую формулу.

Source рекомендация «достаточно 50 production edge-case prompts и json.loads» подходит как smoke/debug набор для части format failures. Она не доказывает agent reliability: корректный JSON может вызвать неверный tool или неверно обработать observation. И пользователь уточнил теоретический режим —production dataset и запуск такого benchmark ему не приписываются.

### Paired comparison и uncertainty

Пусть на 100 одинаковых queries новая система решила 70, baseline65. Совместная разметка:55 успехов у обеих,15 только у новой,10 только у baseline,20 неуспехов у обеих. Difference5points. Отдельные totals скрывают 25 disagreement cases; именно их просмотр объяснит, какие ошибки изменились. Пять дополнительных успехов сами по себе ещё не доказывают устойчивый общий перевес.

Paired bootstrap выбирает queries с возвращением и для каждого выбранного ID берёт outcomes **обеих** систем, затем считает difference. Если независимо resample old и new results, потеряется pairing. Если несколько queries относятся к одному документу/сессии, resample можно делать на уровне таких groups, чтобы не считать зависимые paraphrases независимыми observations. Конкретный sampling unit выбирают по задаче.

Интервал bootstrap отражает variation от resampling данного набора при выбранных assumptions. Он не обнаружит неизвестные deployment topics, систематически неверные labels и утечку test в prompt tuning. Multiple seeds измеряют stochastic run variation, а не заменяют новые независимые задачи. Повторно подбирать метод по final test и затем представлять тот же score как независимую проверку —ещё один канал переобучения.

Учебный итоговый experiment record должен позволять восстановить denominator, labels, traces, configs, criterion success и затраты. Пока есть только расчёты и проект протокола, это так и называется; model quality или superiority конкретного deployed pipeline не заявляются без реального измерения.

### Вопросы для самопроверки

1. Обучает ли graph routing веса? **Ответ:** нет, без отдельного training процесса.
2. Передаёт ли ordinary RAG vector напрямую generator? **Ответ:** обычно нет, vector выбирает IDs, затем подаётся текст evidence.
3. Обязан ли каждый AutoGen agent быть LangGraph graph? **Ответ:** нет, это разные уровни/реализации loop/coordination.
4. Есть ли универсальный отдельный tool head? **Ответ:** нет; распространён autoregressive вывод протокола через vocabulary head, а отдельный router —возможное дополнительное решение.
5. Делает ли SFT format гарантированным? **Ответ:** нет. Обученная вероятность, grammar constraint и проверка смысла/прав —разные механизмы.
6. Почему checkpoint (сохранённое состояние модели) не обеспечивает exactly-once? **Ответ:** внешний effect мог произойти до сохранения результата; retry требует deduplication или recovery policy.
7. Кто решает, что поиска достаточно? **Ответ:** модель предлагает finish по контексту, runtime применяет заданные критерии и лимиты; полнота evidence не возникает автоматически.
8. Почему global top-k и поздний scope filter могут вернуть пустоту? **Ответ:** допустимый релевантный chunk мог находиться ниже global cutoff.
9. Обязан ли graph node вызывать LLM? **Ответ:** нет, это может быть обычная функция или tool executor.
10. Чем parallelism отличается от сокращения работы? **Ответ:** он может сократить критический путь, сохранив общие tokens и вычисления.
11. Что означает хороший tool-call validity при плохом task success? **Ответ:** правильный формат не гарантирует нужный инструмент, evidence и верную интерпретацию.

**Практика:** на бумаге проследить запрос «найди правило таймаута и процитируй» через scope→retrieval→tool observation→answer→validation (проверочная часть для выбора решения). Добавить no-results, timeout и forbidden-document случаи. Более подробный прикладной материал/исходные вопросы — в [разделе приложений](<../../archive/2026-10-20-before-rebuild/06-llm/04-applications/План — приложения LLM.md>).

## Итоговая самопроверка

1. Чем hit@k отличается от recall@k по всем relevant documents? **Ответ:** binary hit — наличие хотя бы одного relevant item, recall — доля relevant items; достаточность полного набора evidence оценивают дополнительно.
2. Почему gold context полезен? **Ответ:** отделяет часть upstream failures от generation при доступном evidence.
3. Что доказывает schema validity? **Ответ:** только соответствующий структурный слой, не final task success.

4. Равны ли MRR и обратная средняя позиция? **Ответ:** нет, усреднение и инверсия не перестановочны.
5. Почему macro и micro recall различаются? **Ответ:** macro одинаково взвешивает queries, micro —gold items по выбранной единице.
6. Достаточен ли один relevant hit для multi-hop ответа? **Ответ:** не обязательно, может требоваться набор passages.
7. Что делать с unjudged document? **Ответ:** не объявлять установленно нерелевантным без оговорки неполноты qrels.
8. Как always-abstain может выглядеть хорошо? **Ответ:** если conditional correctness скрывает отсутствие ответов; нужны coverage и full task denominator.
9. Доказывает ли oracle gain исключительно retrieval error? **Ответ:** нет, меняется context; необходим trace/controlled ablation.
10. Почему success@3 нельзя сравнить с first-attempt success как чистый model gain? **Ответ:** отличаются compute budget и selection protocol.
11. Значит ли reference-free отсутствие внешних assumptions? **Ответ:** нет, остаются judges, context и выбранный критерий.
12. Почему paired bootstrap сохраняет один query ID для обеих систем? **Ответ:** это сохраняет зависимость результатов на одной задаче.

**Практика:** собрать 12 учебных questions с qrels/answerability, три tool traces с ошибками и явную rubric. Вручную посчитать Recall/MRR/nDCG examples; затем спроектировать baseline→одно изменение→error review. Большой production project не требуется для теоретической проверки, но реальные metrics можно заявлять только после запуска.

## Результат главы и переход дальше

Следующие темы: [Постановка обучения агента и baseline](<../06-agent-learning/agent-task.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
