# Classification, NER, spans и extractive QA

**Уровень:** основа ветки.

**Предпосылки:** [Encoder, tokenizer, attention и pooling](<bert-interface.md>), [Данные, task heads, fine-tuning и DAPT](<bert-adaptation.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как преобразовать текстовую задачу в targets и восстановить ответ в исходных координатах?





## Карта вычисления и границы этапов

![Classification, NER, spans и extractive QA: входы, преобразования и выходы](<../../оформление/course/bert-tasks.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:4299361cf5dab7b97d -->
<a id="u-4299361cf5dab7b97d"></a>

![Общий encoder выдаёт последовательность состояний; task head, targets и decoding определяют прикладную задачу.](../../archive/2026-10-20-before-rebuild/оформление/transformer/bert-task-heads.svg)

<!-- /unit:4299361cf5dab7b97d -->

<!-- unit:cedce13063db5dd79d -->
<a id="u-cedce13063db5dd79d"></a>

Общий encoder выдаёт последовательность состояний; task head, targets и decoding определяют прикладную задачу.

<!-- /unit:cedce13063db5dd79d -->

<!-- unit:ef93e9f77cad02b7e8 -->
<a id="u-ef93e9f77cad02b7e8"></a>

## Classification: тема, intent, sentiment и moderation

<!-- /unit:ef93e9f77cad02b7e8 -->

<!-- unit:31e227a80565a95eec -->
<a id="u-31e227a80565a95eec"></a>

Pooled vector $z$ подаётся в linear/MLP head, logits $Wz+b$ обучаются CE для mutually exclusive labels. Sentiment — оценка отношения, topic — тема, stance — позиция относительно конкретного утверждения. «В статье сказано: товар ужасный» содержит negative quotation, но позиция автора зависит от контекста. Moderation дополнительно зависит от annotation policy, адресата и цитирования.

<!-- /unit:31e227a80565a95eec -->

<!-- unit:ffc5742472838f3c11 -->
<a id="u-ffc5742472838f3c11"></a>

Tiny counterfactual pairs помогают проверить смысл: «сервис понравился»/«сервис не понравился» должны различаться для sentiment; typo без смены смысла не обязан менять label. Ошибка guideline отличается от model error. Для ordinal classes стоит оценить ordinal objective/стоимость дальнего промаха, а не считать все classes unrelated.

<!-- /unit:ffc5742472838f3c11 -->

<!-- unit:39359944dece85815b -->
<a id="u-39359944dece85815b"></a>

Baseline TF-IDF+linear, frozen vectors+head и tuning сравниваются на group/time split. Macro-F1, per-class recall, confusion matrix, calibration и truncation slices дополняют accuracy. Long-text topic head не исправит отсутствие evidence, отрезанного tokenizer.

<!-- /unit:39359944dece85815b -->

<!-- unit:66ec8a966f20200b24 -->
<a id="u-66ec8a966f20200b24"></a>

<!-- source-depth-bert-Г -->

<!-- /unit:66ec8a966f20200b24 -->

<!-- unit:10e09f0541c0557c88 -->
<a id="u-10e09f0541c0557c88"></a>

### Классификатор читает представление, но смысл классов задаёт разметка

<!-- /unit:10e09f0541c0557c88 -->

<!-- unit:2e149de008cf7dc2bf -->
<a id="u-2e149de008cf7dc2bf"></a>

Возьмём обращение «Посылку доставили, но упаковка разорвана». Topic может быть «доставка», sentiment — отрицательный, intent — «жалоба на повреждение». Три ответа относятся к одному тексту и не противоречат друг другу. Если обучающий набор смешивает эти понятия под одной меткой, encoder не сможет устранить логическое противоречие увеличением числа слоёв.

<!-- /unit:2e149de008cf7dc2bf -->

<!-- unit:be63b8fa04c6675381 -->
<a id="u-be63b8fa04c6675381"></a>

Для single-label головы pooled vector длины D преобразуется в C logits. У трёх logits (2,1,0) softmax примерно (0.665241,0.244728,0.090031). Если правильный первый класс, CE около 0.407606; если третий — 2.407606. Одна и та же уверенная модель получает разную ошибку в зависимости от правильной метки. Gradient первого случая по logits примерно (−0.334759,0.244728,0.090031), затем возвращается через head и encoder.

<!-- /unit:be63b8fa04c6675381 -->

<!-- unit:5950f0d5c2f6d3ff20 -->
<a id="u-5950f0d5c2f6d3ff20"></a>

На inference выбирается класс или применяется правило отказа. На training обычно не нужен argmax: loss работает с logits напрямую. Обучение argmax как будто это дифференцируемый continuous output обрывает нужный сигнал. Confidence тоже не равен истинности: max softmax оценивается вместе с калибровкой, источником текста и протоколом тестирования.

<!-- /unit:5950f0d5c2f6d3ff20 -->

<!-- unit:1a7657c45e952ea196 -->
<a id="u-1a7657c45e952ea196"></a>

Полезный контрфактический контроль меняет только один существенный элемент. «Не рекомендую товар» и «Рекомендую товар» должны различаться для соответствующего sentiment guideline. Но удаление «не» в цитате, которую автор опровергает, может не менять позицию автора. Такие пары проверяют конкретное правило понимания; набор искусственно простых отрицаний не заменяет оценку естественных случаев.

<!-- /unit:1a7657c45e952ea196 -->

<!-- unit:4544aac4b8ce66fa34 -->
<a id="u-4544aac4b8ce66fa34"></a>

## Multi-label, hierarchy и selective prediction

<!-- /unit:4544aac4b8ce66fa34 -->

<!-- unit:99cbf5b33c7322ac34 -->
<a id="u-99cbf5b33c7322ac34"></a>

При независимых labels sigmoid/BCE выдаёт probability каждого события. Global threshold проще, per-label thresholds позволяют учитывать prevalence/cost, но требуют достаточно validation data. Missing labels mask-ятся, а не превращаются автоматически в negatives.

<!-- /unit:99cbf5b33c7322ac34 -->

<!-- unit:206c1c43997c9f7973 -->
<a id="u-206c1c43997c9f7973"></a>

Toy truth двух examples: $(1,0,1)$ и $(0,1,0)$. Scores $(0.8,0.4,0.3)$ и $(0.2,0.6,0.1)$ при threshold 0.5 дают TP=2, FP=0, FN=1. Micro-F1=0.8, macro-F1=$2/3$: редкий третий label полностью пропущен. Threshold 0.25 только для третьего label исправляет именно этот toy set, но выбирать его по итоговому test нельзя.

<!-- /unit:206c1c43997c9f7973 -->

<!-- unit:732d3d6865cea6fc96 -->
<a id="u-732d3d6865cea6fc96"></a>

Hierarchy child→parent может требовать constrained decoding, conditional probabilities или penalty на inconsistencies. Таксономия должна действительно задавать включение; «связанные labels» не всегда parent-child. Изменение taxonomy version меняет semantics output и требует migration.

<!-- /unit:732d3d6865cea6fc96 -->

<!-- unit:a15b99731cff6fcdd4 -->
<a id="u-a15b99731cff6fcdd4"></a>

Abstention откладывает решение при недостаточной надёжности. Report содержит coverage и risk на accepted subset. Система, отвечающая только на одном лёгком примере, не становится общей 100%-accurate моделью. Confidence threshold выбирается на calibration/validation и проверяется на relevant OOD cases.

<!-- /unit:a15b99731cff6fcdd4 -->

<!-- unit:0b6be6e00d950a8b3f -->
<a id="u-0b6be6e00d950a8b3f"></a>

### Выбор порога отделён от обучения распределений

<!-- /unit:0b6be6e00d950a8b3f -->

<!-- unit:6e403d53d56195559f -->
<a id="u-6e403d53d56195559f"></a>

BCE может обучить три независимых вероятности, а threshold превращает их в набор labels. При изменении порога веса encoder не меняются. В исходном примере score третьего класса 0.3 ниже общего 0.5; отдельный порог 0.25 вернул бы нужную метку. Однако два примера не доказывают, что такой порог полезен на новых данных: у класса могут появиться ложноположительные с теми же scores.

<!-- /unit:6e403d53d56195559f -->

<!-- unit:fe4c1b79c6e04e29f0 -->
<a id="u-fe4c1b79c6e04e29f0"></a>

Micro-F1 суммирует TP/FP/FN по всем labels, macro-F1 усредняет отдельные class F1. Поэтому сильные частые классы могут скрыть полностью пропущенный редкий класс. При выборе per-label thresholds нужно достаточно validation examples именно для этих labels; иначе настройка легко переобучается. Для классов без положительных примеров отдельно объявляют правило подсчёта неопределённых precision/recall/F1.

<!-- /unit:fe4c1b79c6e04e29f0 -->

<!-- unit:0af79613959e1303af -->
<a id="u-0af79613959e1303af"></a>

Иерархия добавляет другой тип ограничения. Если «повреждённая упаковка» действительно является подкатегорией «повреждение», предсказание child без parent нарушает таксономию. Можно замкнуть предсказания вверх по дереву, но это изменит false positives родителей. Можно обучать условную вероятность child при parent, но тогда итоговая вероятность зависит от обоих решений. Нужно оценивать весь выбранный способ, а не только независимые logits.

<!-- /unit:0af79613959e1303af -->

<!-- unit:a19d862cc17c28453d -->
<a id="u-a19d862cc17c28453d"></a>

### Отказ от ответа измеряют парой coverage–risk

<!-- /unit:a19d862cc17c28453d -->

<!-- unit:c101ad0f806819a29b -->
<a id="u-c101ad0f806819a29b"></a>

Пусть из 10 примеров система принимает 8 и ошибается на одном принятом. Coverage=8/10=0.8, selective risk=1/8=0.125. Если повысить threshold и оставить только один простой правильный ответ, risk станет 0, но coverage — 0.1. Эти результаты нельзя сравнивать только по accuracy принятых ответов.

<!-- /unit:c101ad0f806819a29b -->

<!-- unit:d2ec1ec39940de8e31 -->
<a id="u-d2ec1ec39940de8e31"></a>

Порог отказа выбирают на validation/calibration, затем измеряют на новых данных. Вне распределения высокий softmax может сохраняться даже у неверных ответов. Поэтому полезны slices по источникам и типам ошибок, а не только одно число coverage. Данный пример объясняет метрики и не задаёт универсальный рабочий порог.

<!-- /unit:d2ec1ec39940de8e31 -->

<!-- unit:9bf34e4f0d753fafa6 -->
<a id="u-9bf34e4f0d753fafa6"></a>

## NER/POS/slots: word, subword и char span

<!-- /unit:9bf34e4f0d753fafa6 -->

<!-- unit:9961c33b7d679ad8d4 -->
<a id="u-9961c33b7d679ad8d4"></a>

Token projection даёт logits $B\times L\times C$. В BIO B-TYPE начинает entity, I-TYPE продолжает, O вне entities; BILOU отдельно обозначает single-token и boundaries. Word labels переносятся на pieces по выбранной policy: supervision first piece only либо coherent labels всех pieces. При втором варианте continuation исходного B-PER обычно становится I-PER, а не начинает новую entity.

<!-- /unit:9961c33b7d679ad8d4 -->

<!-- unit:f631ce06a3f3645cc9 -->
<a id="u-f631ce06a3f3645cc9"></a>

Для «Иванова-Петрова» token strings могут быть несколькими pieces. `word_ids` и offsets связывают их с исходным text; special/padding labels ignored. Нельзя восстанавливать entities простым concat `##` strings, теряя исходные пробелы/Unicode. [Alignment guide](https://huggingface.co/docs/transformers/main/tasks/token_classification).

<!-- /unit:f631ce06a3f3645cc9 -->

<!-- unit:c64e024787d20376fd -->
<a id="u-c64e024787d20376fd"></a>

Entity-level strict F1 считает correct entity только при совпадении type и boundaries. Если из четырёх gold entities предсказаны четыре, но одна имеет wrong boundary, TP=3, FP=1, FN=1, F1=0.75. Token accuracy может быть высокой из-за множества O. POS обычно оценивает word labels; slot filling дополнительно требует целостности нужных slots в task.

<!-- /unit:c64e024787d20376fd -->

<!-- unit:67fac5de5c0f1f4640 -->
<a id="u-67fac5de5c0f1f4640"></a>

CRF добавляет sequence score emissions+transitions:

<!-- /unit:67fac5de5c0f1f4640 -->

<!-- unit:1f1ca6dacaeb8ab90c -->
<a id="u-1f1ca6dacaeb8ab90c"></a>

$$
P(y\mid x)=\frac{\exp\left(\sum_t e_t(y_t)+\sum_t A_{y_{t-1},y_t}\right)}{Z(x)}.
$$

<!-- /unit:1f1ca6dacaeb8ab90c -->

<!-- unit:7d4a3ab26b7376568b -->
<a id="u-7d4a3ab26b7376568b"></a>

Forward algorithm считает normalizer, Viterbi — лучшую path, обычно cost порядка $LC^2$. Constraints могут запретить незаконные BIO transitions, но правильная syntax labels не гарантирует semantic entity correctness. CRF стоит добавлять по diagnosis, а не считать обязательной головой BERT.

<!-- /unit:7d4a3ab26b7376568b -->

<!-- unit:3136422c695a4d02e0 -->
<a id="u-3136422c695a4d02e0"></a>

### Как subwords меняют число supervised решений

<!-- /unit:3136422c695a4d02e0 -->

<!-- unit:67e78691b31b1a458f -->
<a id="u-67e78691b31b1a458f"></a>

Пусть слово с gold B-PER разбилось на «Иван», «##ова». При first-piece supervision первый subword получает B-PER, второй — ignore. При supervision всех частей можно дать B-PER, I-PER. Это сохраняет одну сущность, в отличие от двух B-PER. Политика должна согласовываться с тем, как предсказания возвращаются к словам и spans.

<!-- /unit:67e78691b31b1a458f -->

<!-- unit:35c048f5190bb38dba -->
<a id="u-35c048f5190bb38dba"></a>

Если длинная фамилия разбита на пять частей и loss усредняется по всем subwords, она даёт до пяти вкладов, а короткое имя — один. При first-piece supervision оба слова дают по одному. Можно использовать и веса частей, но это уже явно выбранная нормировка. Различие особенно заметно при сравнении языков с разной fragmentation. Alignment, loss mask и метрика должны описывать одну и ту же единицу разметки.

<!-- /unit:35c048f5190bb38dba -->

<!-- unit:0a2f42d12291a6e277 -->
<a id="u-0a2f42d12291a6e277"></a>

Верная последовательность «B-PER I-PER O» ещё не доказывает правильные границы: encoder может выделить лишний титул или пропустить часть фамилии. Strict entity evaluation сравнивает исходные char spans и types. Если модель предсказала «Иванова» вместо «Иванова-Петрова», совпавшие subwords не превращают это в полностью правильную сущность при exact-span правиле.

<!-- /unit:0a2f42d12291a6e277 -->

<!-- unit:fcff7378a6531bc68b -->
<a id="u-fcff7378a6531bc68b"></a>

### CRF: маленький пример, где локальный выбор проигрывает

<!-- /unit:fcff7378a6531bc68b -->

<!-- unit:2577927c632c929db0 -->
<a id="u-2577927c632c929db0"></a>

Для прозрачной арифметики возьмём только два labels O и B-PER и две позиции; I-labels в этом примере не используются. Emissions первой позиции (2,1), второй (0,3). Transition O→B равен −2, остальные переходы — 0; start/end scores тоже 0. Независимые argmax выбирают O,B: emission sum=5, но после transition penalty path score=3.

<!-- /unit:2577927c632c929db0 -->

<!-- unit:9b4a1cdcd6fe11c00a -->
<a id="u-9b4a1cdcd6fe11c00a"></a>

Все четыре пути имеют scores: O,O — 2; O,B — 3; B,O — 1; B,B — 4. Viterbi выберет B,B с общим score 4. Два B-PER подряд здесь означают две соседние однотокенные сущности, что допустимо в этом сокращённом пространстве labels. Пример показывает влияние transitions, а не универсальную предпочтительность такого предсказания.

<!-- /unit:9b4a1cdcd6fe11c00a -->

<!-- unit:339bba2e11037f1d14 -->
<a id="u-339bba2e11037f1d14"></a>

Нормировка training loss включает **все** пути: $\log Z=\log(e^2+e^3+e^1+e^4)$. Для gold O,B negative log-likelihood равна logZ−3. Это отличается от суммы двух независимых CE: путь оценивается совместно. Производная по emission выражается как ожидаемое число соответствующего label в позиции минус gold indicator; по transition — как ожидаемое число переходов минус gold count.

<!-- /unit:339bba2e11037f1d14 -->

<!-- unit:8692b5ced1470b0a36 -->
<a id="u-8692b5ced1470b0a36"></a>

Вместо перебора C^L путей forward recursion хранит по одному значению для каждого последнего label:

<!-- /unit:8692b5ced1470b0a36 -->

<!-- unit:9aa2fbeba5cafd05c7 -->
<a id="u-9aa2fbeba5cafd05c7"></a>

$$
\alpha_t(j)=e_t(j)+\operatorname{logsumexp}_i\bigl[\alpha_{t-1}(i)+A_{ij}\bigr].
$$

<!-- /unit:9aa2fbeba5cafd05c7 -->

<!-- unit:1eb2c7c25b419a2f9b -->
<a id="u-1eb2c7c25b419a2f9b"></a>

Viterbi использует max вместо log-sum-exp и сохраняет backpointers, чтобы восстановить путь. Обе процедуры разобраны также в [учебном примере CRF PyTorch](https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html). Forward нужен для суммы вероятностей, Viterbi — для лучшего пути; эти цели нельзя поменять местами только потому, что обе процедуры динамические. BIO constraints, например запрет начинать с I-PER или продолжать I-PER после B-ORG, задаются отдельно. Они гарантируют допустимость labels, но не правильность распознанного имени.

<!-- /unit:1eb2c7c25b419a2f9b -->

<!-- unit:6eb15546d4847483ec -->
<a id="u-6eb15546d4847483ec"></a>

## Nested entities, relations, events и coreference

<!-- /unit:6eb15546d4847483ec -->

<!-- unit:8aa92e2cf65e4d9d73 -->
<a id="u-8aa92e2cf65e4d9d73"></a>

Flat BIO даёт один label на token и не выражает произвольные nested/overlapping spans. Span classifier рассматривает candidates $(i,j)$ и representation endpoints/pooling, затем type/NA. Всего contiguous spans $L(L+1)/2$; при maximum width $M\le L$ их $ML-M(M-1)/2$. Для $L=5,M=2$ — девять вместо пятнадцати.

<!-- /unit:8aa92e2cf65e4d9d73 -->

<!-- unit:a88615f77e05f1d273 -->
<a id="u-a88615f77e05f1d273"></a>

Пример «Университет города Москвы» может содержать organization целиком и location внутри по guideline. Один flat sequence не кодирует оба произвольных overlapping entities одновременно. Candidate pruning уменьшает compute, но вводит recall ceiling: удалённый gold span head уже не вернёт.

<!-- /unit:a88615f77e05f1d273 -->

<!-- unit:f17603559ae374606f -->
<a id="u-f17603559ae374606f"></a>

Relation extraction получает pair entity representations и context; negatives включают NA и трудные похожие pairs. Gold-entity evaluation показывает head quality условно на идеальном NER; predicted-entity pipeline включает upstream losses. Фраза «Анна не работает в компании Б» не подтверждает employment relation только по совместному упоминанию.

<!-- /unit:f17603559ae374606f -->

<!-- unit:fe22ced4a90d27f616 -->
<a id="u-fe22ced4a90d27f616"></a>

Event extraction разделяет trigger, arguments и roles: «Компания купила завод» требует buyer/asset и event type. Coreference связывает mentions одного referent через antecedent scoring и document context; одинаковая surface string не гарантирует один entity. Staged против joint models сравниваются по total outcome и supervision budget, с split по documents/entities.

<!-- /unit:fe22ced4a90d27f616 -->

<!-- unit:cbc4b16f13dc202769 -->
<a id="u-cbc4b16f13dc202769"></a>

### Что теряется при выборе candidates до головы

<!-- /unit:cbc4b16f13dc202769 -->

<!-- unit:2e6921eade5e834657 -->
<a id="u-2e6921eade5e834657"></a>

Для L=5 возможны 15 непрерывных spans. Ограничение width≤2 оставляет 5 однотокенных и 4 двухтокенных, всего 9. Если gold entity занимает три токена, в этом candidate set её нет. Никакие веса span classifier уже не выберут отсутствующий candidate. Поэтому pruning recall измеряют отдельно от качества классификации оставшихся spans.

<!-- /unit:2e6921eade5e834657 -->

<!-- unit:b159af9929d1a8df99 -->
<a id="u-b159af9929d1a8df99"></a>

Span representation может объединять состояния начала и конца, pooling внутренности и признак длины. Его score не обязан равняться произведению token-label probabilities. Для nested entities допускаются несколько spans, перекрывающих одни позиции, если guideline этого требует. Для полного pipeline дополнительно нужно решить, какие комбинации spans разрешены и как подавлять дубликаты.

<!-- /unit:b159af9929d1a8df99 -->

<!-- unit:0f66e86991a3951bcf -->
<a id="u-0f66e86991a3951bcf"></a>

Relation head получает не только названия двух сущностей, но и контекст их связи. «Анна работала в Б» и «Анна не работала в Б» имеют те же entity strings и разные отношения. Если candidate entity pairs формируются из предсказанного NER, пропуск одной сущности делает правильную relation недостижимой. Метрики с gold entities и с predicted entities отвечают на разные вопросы и должны называться явно.

<!-- /unit:0f66e86991a3951bcf -->

<!-- unit:81d08b2d6b53b4ca74 -->
<a id="u-81d08b2d6b53b4ca74"></a>

Event arguments тоже зависят от ролей: в «Компания А продала завод компании Б» А — seller, Б — buyer, завод — asset. Замена порядка упоминаний не должна автоматически менять роли без изменения смысла. Coreference требует связать mentions одного referent, включая местоимения; одинаковые имена могут относиться к разным людям, а разные выражения — к одному. Эти задачи требуют своих labels и decoding, даже если underlying encoder общий.

<!-- /unit:81d08b2d6b53b4ca74 -->

<!-- unit:c7db6082b923730e87 -->
<a id="u-c7db6082b923730e87"></a>

## NLI, paraphrase, STS и relevance

<!-- /unit:c7db6082b923730e87 -->

<!-- unit:3e99419e435f135afd -->
<a id="u-3e99419e435f135afd"></a>

Pair encoding позволяет tokens двух texts взаимодействовать в attention. Premise «Анна живёт в Москве», hypothesis «Анна живёт в России» может быть entailment при заданных знаниях; reverse direction не обязательно entailment. «Анна не живёт в Москве» lexical близко, но contradiction. Общая тема без ответа query не означает relevance.

<!-- /unit:3e99419e435f135afd -->

<!-- unit:0fb63b63bf9ad99687 -->
<a id="u-0fb63b63bf9ad99687"></a>

NLI three-class labels, paraphrase equivalence, STS continuous similarity и retrieval relevance различаются. Для STS выбирают scale, MSE/Huber и correlations; для duplicate detection — identity policy и errors ложного merge. Нельзя брать любой NLI classifier и без проверки называть score universal similarity.

<!-- /unit:0fb63b63bf9ad99687 -->

<!-- unit:29b7c5b985441cbbff -->
<a id="u-29b7c5b985441cbbff"></a>

Cross-encoder sees pair jointly, bi-encoder encodes independently и сравнивает vectors. Truncation second segment может удалить нужный passage. Сравнение проводится на same held-out queries/pairs и budget; candidate order/asymmetry учитываются.

<!-- /unit:29b7c5b985441cbbff -->

<!-- unit:a7260504d407e11f2a -->
<a id="u-a7260504d407e11f2a"></a>

### Похожесть, логическое следование и полезность запроса

<!-- /unit:a7260504d407e11f2a -->

<!-- unit:63d7d6b744a19142df -->
<a id="u-63d7d6b744a19142df"></a>

«Кошка спит на диване» и «Кошка не спит на диване» почти совпадают лексически. Для поиска по теме они могут быть близкими, для entailment — противоречить, для exact duplicate policy — быть разными. Нельзя требовать одной универсальной шкалы, не объявив задачу. Представление и labels должны сохранять именно те различия, которые важны для нужного решения.

<!-- /unit:63d7d6b744a19142df -->

<!-- unit:451f2c50a357ce3c61 -->
<a id="u-451f2c50a357ce3c61"></a>

NLI направлен: premise может поддерживать hypothesis, но обратное следование не обязано выполняться. Paraphrase обычно требует взаимной смысловой эквивалентности по выбранному guideline. STS задаёт градуированную похожесть, relevance относится к конкретному query и может быть асимметричной. Поэтому label «похожая тема» не заменяет judgement «документ содержит ответ».

<!-- /unit:451f2c50a357ce3c61 -->

<!-- unit:dc1b847b2e5812a932 -->
<a id="u-dc1b847b2e5812a932"></a>

При joint pair encoding attention читает обе строки вместе, и score зависит от их взаимодействия. При independent encoding можно кешировать документы, но сложное взаимодействие приходится выражать геометрией и функцией сравнения. Это архитектурный компромисс между переиспользованием вычислений и точностью попарного сопоставления, который проверяют на нужной задаче.

<!-- /unit:dc1b847b2e5812a932 -->

<!-- unit:0a5b5a10cce126e7a2 -->
<a id="u-0a5b5a10cce126e7a2"></a>

## Extractive QA: start/end и ограничения

<!-- /unit:0a5b5a10cce126e7a2 -->

<!-- unit:d1e2d6c037eafaad01 -->
<a id="u-d1e2d6c037eafaad01"></a>

Question+context encoder получает two scalar projections:

<!-- /unit:d1e2d6c037eafaad01 -->

<!-- unit:c337c3230ec83c37de -->
<a id="u-c337c3230ec83c37de"></a>

$$
s_i=w_s^\top h_i+b_s,\qquad e_j=w_e^\top h_j+b_e,
\qquad L=\operatorname{CE}(s,i^*)+\operatorname{CE}(e,j^*).
$$

<!-- /unit:c337c3230ec83c37de -->

<!-- unit:7aefdcac5f00d6b9fa -->
<a id="u-7aefdcac5f00d6b9fa"></a>

Некоторые implementations усредняют two terms; scale convention проверяют. Decoding максимизирует $s_i+e_j$ при valid context positions, $i\le j$ и maximum answer length. Mask question/SEP/padding задаётся явно. [QA offsets/windowing](https://huggingface.co/docs/transformers/main/tasks/question_answering).

<!-- /unit:7aefdcac5f00d6b9fa -->

<!-- unit:6ca519a7ddc3e3fce4 -->
<a id="u-6ca519a7ddc3e3fce4"></a>

Для three context tokens start $(0,5,2)$, end $(4,0,2)$ independent maxima дают invalid start=2,end=1. При max length 2 лучший valid span (2,3), score 7; span (1,1) score 4. Argmax двух heads независимо не решает constrained span task.

<!-- /unit:6ca519a7ddc3e3fce4 -->

<!-- unit:7cb6efe386db698c86 -->
<a id="u-7cb6efe386db698c86"></a>

Char offsets half-open позволяют вернуть исходный substring. В «Анна в Москве» «Москве» занимает [7,13). Нормализация/tokenization не должна потерять mapping original characters. При overlapping windows gold answer может быть целиком только в одном; labels других windows должны соответствовать no-answer/window protocol, а не указывать обрезанный fragment как full truth.

<!-- /unit:7cb6efe386db698c86 -->

<!-- unit:6f2ab21324ffb05ea3 -->
<a id="u-6f2ab21324ffb05ea3"></a>

Unanswerable handling сравнивает null score с лучшим valid span по calibrated rule. Например, null=8 против best=7 даёт difference 1; выберется null при threshold 0 в такой convention. Смена sign/threshold меняет decision. Несколько gold spans, boundary windows и cross-window score aggregation требуют explicit rules; они не решаются автоматически одним head.

<!-- /unit:6f2ab21324ffb05ea3 -->

<!-- unit:24f3a1d28907ef2e91 -->
<a id="u-24f3a1d28907ef2e91"></a>

Оцениваются EM/token-F1, no-answer precision/recall и source correctness. Retrieval+QA отдельно проверяется на gold и retrieved context: плохой retrieval не следует смешивать с head failure на достаточном evidence. Extractive model выбирает fragment, а не доказывает, что в corpus существовал ответ.

<!-- /unit:24f3a1d28907ef2e91 -->

<!-- unit:b9fc88a00bd2d62321 -->
<a id="u-b9fc88a00bd2d62321"></a>

### Loss по двум осям ещё не задаёт допустимый ответ

<!-- /unit:b9fc88a00bd2d62321 -->

<!-- unit:19883562d3bbc1c625 -->
<a id="u-19883562d3bbc1c625"></a>

Start и end heads возвращают по одному score каждой позиции. Cross-entropy по start учит правильное начало, по end — правильный конец. Но обучение двух распределений не запрещает на inference комбинацию end раньше start. Поэтому decoding явно перебирает или эффективно ищет допустимые пары: обе позиции в context, start≤end, длина не больше M. В задачах с no-answer дополнительно предусматривается специальный null вариант.

<!-- /unit:19883562d3bbc1c625 -->

<!-- unit:871c0645260f1856cd -->
<a id="u-871c0645260f1856cd"></a>

В примере выше индексы 1–3 **однобазовые**. Пять spans длины не больше 2 имеют scores: (1,1)=4, (1,2)=0, (2,2)=5, (2,3)=7, (3,3)=4. Так видно, что лучший допустимый ответ — (2,3), хотя независимые головы предпочли начало 2 и конец 1. При нулевой нумерации соответствующая пара была бы (1,2); смешение этих соглашений с offsets даёт ошибку на один токен.

<!-- /unit:871c0645260f1856cd -->

<!-- unit:8ecd1f8fb983341095 -->
<a id="u-8ecd1f8fb983341095"></a>

Произведение независимых start/end probabilities максимизируется тем же span, что сумма logits **внутри одного окна**, поскольку оба softmax-знаменателя общие для всех candidate pairs данного окна. Но добавление одинаковой константы к start logits и другой константы к end logits не меняет их распределения, а все raw span scores сдвигает. Поэтому непосредственное сравнение raw scores разных windows требует объявленного протокола, а не предположения об общей абсолютной шкале.

<!-- /unit:8ecd1f8fb983341095 -->

<!-- unit:70ffb282c692e6a29a -->
<a id="u-70ffb282c692e6a29a"></a>

### Окна меняют обучающие примеры, а не исходный вопрос

<!-- /unit:70ffb282c692e6a29a -->

<!-- unit:e06a4f61f1cf894360 -->
<a id="u-e06a4f61f1cf894360"></a>

Длинный context может быть разбит на overlapping windows с повторённым вопросом. У каждого окна свои token offsets, но ответ возвращается в исходный документ. Если gold answer [a,b) целиком в одном окне, там можно построить правильные start/end labels. Если он пересекает границу, нельзя выдавать только видимую половину за полный ответ. Применяемое правило обработки таких окон должно совпадать с training recipe.

<!-- /unit:e06a4f61f1cf894360 -->

<!-- unit:d54f08b60f52f5a9e4 -->
<a id="u-d54f08b60f52f5a9e4"></a>

Окно без ответа не обязательно означает, что исходный вопрос не имеет ответа в документе: evidence могло остаться в другом окне. Это различие важно для null supervision и объединения результатов. Восстановление исходной подстроки выполняется по offsets выбранного span; специальные tokens и вопрос не должны попадать в ответ.

<!-- /unit:d54f08b60f52f5a9e4 -->

<!-- unit:335a620fa538337bd5 -->
<a id="u-335a620fa538337bd5"></a>

Для сравнения с null удобно явно объявить знак, например $\Delta=s_{\rm null}-s_{\rm best}$. Тогда в приведённом примере Δ=8−7=1, и при правиле «null, если Δ>0» выбирается отсутствие ответа. Противоположное определение margin требует противоположного знака порога. Разность span/null внутри одного окна устраняет общий аддитивный сдвиг scores, но сама по себе не гарантирует калибровку между разными окнами или документами.

<!-- /unit:335a620fa538337bd5 -->

<!-- unit:7a1f8bb375fdeb777c -->
<a id="u-7a1f8bb375fdeb777c"></a>

### Метрики должны различать извлечение и наличие evidence

<!-- /unit:7a1f8bb375fdeb777c -->

<!-- unit:f280ff19285fa97dd0 -->
<a id="u-f280ff19285fa97dd0"></a>

EM требует полного совпадения ответа после оговорённой нормализации. Token-F1 допускает частичное пересечение, поэтому лишнее слово может снижать EM до нуля, сохраняя ненулевой F1. Для no-answer дополнительно важны ложные извлечения из нерелевантного context. Если retrieval не доставил нужный passage, QA head оценивают отдельно на gold context, чтобы понять, где потерян ответ.

<!-- /unit:f280ff19285fa97dd0 -->

<!-- unit:19d97c18b882f1220a -->
<a id="u-19d97c18b882f1220a"></a>

Извлечённая фраза может буквально присутствовать в источнике и при этом не отвечать вопросу: например, относится к другому человеку или к отвергнутому утверждению. Extractive architecture ограничивает форму ответа подстрокой, но не гарантирует правильную связь question–evidence. Примеры с отрицанием, одинаковыми именами и несколькими датами проверяют именно этот механизм.

<!-- /unit:19d97c18b882f1220a -->

<!-- unit:34ca802dc63feabbbc -->
<a id="u-34ca802dc63feabbbc"></a>

## Few-shot и zero-shot

<!-- /unit:34ca802dc63feabbbc -->

<!-- unit:ff7112901b78282903 -->
<a id="u-ff7112901b78282903"></a>

NLI zero-shot переводит candidate label в hypothesis template. Его phrasing влияет на entailment score; mutually-exclusive и multi-label decisions имеют разные normalization/threshold rules. Zero-shot относится к новой task label set, а не отсутствию pretraining/NLI supervision вообще.

<!-- /unit:ff7112901b78282903 -->

<!-- unit:c9b2b164d2e6633d2d -->
<a id="u-c9b2b164d2e6633d2d"></a>

MLM verbalizer сопоставляет class словам в MASK positions. Multi-token label не равна одному vocabulary logit; нужно определить aggregation и prompt calibration. Это специализированный classifier recipe, а не general chat interface.

<!-- /unit:c9b2b164d2e6633d2d -->

<!-- unit:ecc57ecc74788dcfc7 -->
<a id="u-ecc57ecc74788dcfc7"></a>

[SetFit](https://arxiv.org/abs/2209.11055) адаптирует sentence representations по pairs и затем обучает classifier при малом количестве labels. Тысячи pairs из 16 texts не создают тысячи independent исходных examples. Нужны несколько few-shot samples/seeds, frozen baseline и ordinary tuning с comparable budget.

<!-- /unit:ecc57ecc74788dcfc7 -->

<!-- unit:a3a5f23ae5c5e75499 -->
<a id="u-a3a5f23ae5c5e75499"></a>

### Откуда берётся supervision у zero-shot и few-shot

<!-- /unit:a3a5f23ae5c5e75499 -->

<!-- unit:a759f9289cae5597cc -->
<a id="u-a759f9289cae5597cc"></a>

У NLI zero-shot уже есть модель, обученная различать entailment/neutral/contradiction. Новая задача задаётся шаблонами hypotheses, например «Это обращение о возврате». Формулировка label и шаблона входит в pipeline. Scores нескольких hypotheses могут быть независимыми или нормироваться для mutually-exclusive выбора; это отдельное решение, которое нельзя скрыть за словом zero-shot.

<!-- /unit:a759f9289cae5597cc -->

<!-- unit:b02b42a04256a62bdc -->
<a id="u-b02b42a04256a62bdc"></a>

MLM verbalizer сопоставляет class словам в MASK-позиции. Если один label кодируется одним token, а другой — тремя, прямое сравнение одного vocabulary logit уже недостаточно. Нужны правила агрегации и сопоставимости scores. Слово с высокой частотой может иметь высокий prior вне зависимости от текста, поэтому шаблон и калибровка оцениваются на разрешённых данных.

<!-- /unit:b02b42a04256a62bdc -->

<!-- unit:d57832825cff56c2a6 -->
<a id="u-d57832825cff56c2a6"></a>

Few-shot pairs тоже не создают новые независимые источники. Из 16 текстов можно составить $16\cdot15/2=120$ неупорядоченных пар, но все они зависят от тех же 16 текстов. Если сначала сформировать пары, а затем случайно разделить их, один текст легко попадёт на обе стороны evaluation. Сначала разделяют исходные группы, затем создают pairs внутри допустимой части. Несколько выборок few-shot примеров показывают чувствительность к тому, какие именно тексты были размечены.

<!-- /unit:d57832825cff56c2a6 -->

<!-- unit:1aeeb6bab746bbbc1b -->
<a id="u-1aeeb6bab746bbbc1b"></a>

## Контроль и практика

<!-- /unit:1aeeb6bab746bbbc1b -->

<!-- unit:2c986988fe63ca5fd9 -->
<a id="u-2c986988fe63ca5fd9"></a>

1. Почему high micro-F1 скрывает rare label? **Ответ:** суммарные counts доминирующих labels могут перевесить отсутствие редкого.
2. Почему strict NER F1 ниже token accuracy? **Ответ:** entity требует одновременно правильных boundaries/type, а O tokens многочисленны.
3. Что не выражается flat BIO? **Ответ:** произвольные nested/overlapping entities при одном label на token.
4. Может ли NLI entailment score заменить relevance без оценки? **Ответ:** нет; это different supervision.
5. Почему separate start/end argmax ошибочен? **Ответ:** нарушает constraints порядка/длины/context positions.

<!-- /unit:2c986988fe63ca5fd9 -->

<!-- unit:60942c8f9e5a3fc399 -->
<a id="u-60942c8f9e5a3fc399"></a>

6. Равны ли forward и Viterbi у CRF? **Ответ:** нет. Первый суммирует вероятности путей для normalizer, второй ищет один максимальный путь с backpointers.
7. Вернёт ли span classifier gold entity, удалённую width pruning? **Ответ:** нет. Достигнут потолок recall candidate set ещё до классификации.
8. Сопоставимы ли raw start/end scores разных QA windows автоматически? **Ответ:** нет. Их общие аддитивные сдвиги не меняют внутривоконные probabilities, но меняют raw суммы. Нужен объявленный cross-window протокол.

<!-- /unit:60942c8f9e5a3fc399 -->

<!-- unit:28a690349b1e0dd2fe -->
<a id="u-28a690349b1e0dd2fe"></a>

**Практика:** выполните multi-label confusion counts, разметьте nested entities и relation с negation, восстановите char spans. Выберите valid QA span и null decision в примере. Для проектов classification/NER/QA определите baseline, annotation guideline, metric и один controlled intervention; различайте specification и фактический run.

<!-- /unit:28a690349b1e0dd2fe -->

## Проектная практика ветки

<!-- unit:99bcd89231dba8f250 -->
<a id="u-99bcd89231dba8f250"></a>

### P1. Классификация обращений — минимальный практический проект

<!-- /unit:99bcd89231dba8f250 -->

<!-- unit:5d8123f14e3ae2b26b -->
<a id="u-5d8123f14e3ae2b26b"></a>

Russian intent/topic data, group/time split, TF-IDF+linear baseline, frozen task-appropriate vectors и tuned encoder. Один training intervention. Сохранить per-class errors, calibration и abstention policy. Итог: notebook/script, config, data manifest и отчёт о качестве/cost.

<!-- /unit:5d8123f14e3ae2b26b -->

<!-- unit:9b1d2c67a46c142f66 -->
<a id="u-9b1d2c67a46c142f66"></a>

### P2. NER и structured extraction — следующая ступень

<!-- /unit:9b1d2c67a46c142f66 -->

<!-- unit:2c638768430f06beb1 -->
<a id="u-2c638768430f06beb1"></a>

Char-span annotation → tokenizer alignment → token/span head → восстановленные spans. Сравнить rare types и boundaries; добавить relations только после оценки NER. Итог: annotation guideline, alignment examples, strict entity metrics и ошибки.

<!-- /unit:2c638768430f06beb1 -->

<!-- unit:7c7173126907038e85 -->
<a id="u-7c7173126907038e85"></a>

### P4. Вопросы по русским документам — композиция задач

<!-- /unit:7c7173126907038e85 -->

<!-- unit:6f620ef437090c3d36 -->
<a id="u-6f620ef437090c3d36"></a>

P3 → extractive QA → unanswerable handling с source offsets. Сравнить gold-context/retrieved-context regimes и несколько evidence positions. Генеративный RAG добавить как отдельную сравниваемую pipeline. Итог: answer correctness, retrieval failures и обоснованный reject policy.

<!-- /unit:6f620ef437090c3d36 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как преобразовать текстовую задачу в targets и восстановить ответ в исходных координатах?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
