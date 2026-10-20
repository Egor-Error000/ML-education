# Г. RAG: от вопроса и корпуса к проверяемому evidence

[Оглавление](README.md). Пункт 4 плана.

**Основа:** S6:182–1373; дополнительно S5:1038–1817 о pipeline. Сохранены ваши вопросы: «почему LLM поймёт наш embedding», «откуда BM25 берёт keywords» и «насколько вырастет база при нескольких vectors». **Исправления:** ordinary RAG передаёт generator текст, BM25 не требует отдельного LLM keyword extractor, memory/latency не растут по одному универсальному проценту. **Углубление:** полный ручной BM25, ANN vs relevance и storage accounting.

**Уровень:** Основа knowledge-application branch. **Предпосылки:** E1; local retrieval/embedding definition. **Используют:** E5; no encoder-training course required.

## 1. Два процесса: подготовка корпуса и ответ на запрос

Offline/ingestion path: документы → очистка и версии → chunks → metadata/provenance → sparse/dense indexes. Online path: вопрос → retrieval → фильтрация/объединение → reranking → context selection → generation → проверка citations/answer.

У каждого chunk должны быть устойчивые document/chunk IDs, source span или location, version и access scope. Если document обновился, старый chunk не должен продолжать выдавать obsolete правило без понятной политики. Split evaluation по documents/sources делать до нарезки, иначе соседние chunks создают leakage.

Слишком короткий chunk теряет context, слишком длинный разбавляет signal и занимает generation budget. Overlap помогает не разрезать нужную связь, но увеличивает дубликаты и index size. Размер измерять в tokenizer tokens, а semantic boundaries учитывать вместе с длиной. Это параметры pipeline, а не универсальное «лучше всегда 512tokens».




**Схема курса:** corpus → chunk/metadata/index → query → sparse + dense candidates → fuse/rerank → evidence IDs → grounded answer → retrieval and answer evaluation

<!-- source-depth-applications-Г -->
### 1.1. Что RAG добавляет к знаниям модели

В начале D.5 исходник выделяет ограничение context, обновление знаний и проверяемые sources. RAG выбирает внешний материал для текущего ответа; он не обязан сохранять новые факты в weights. Если сегодня изменился документ и обновлён index, следующий запрос может получить новую версию без fine-tuning generator. Но если ingestion не обновился или retrieval выбрал stale passage, актуальность не возникает автоматически.

Наличие source также не устраняет hallucination само по себе. Generator может неправильно прочитать условие, смешать две версии, приписать source лишнее или ответить из parametric memory. Поэтому цель —сделать evidence доступным и проверяемым, а затем оценить, как оно использовано. Аналогия исходника с открытым справочником полезна: доступ к книге не гарантирует, что отвечающий открыл нужную страницу и правильно её понял.

Нельзя сводить RAG к любой vector database. Retrieval может быть lexical, dense, hybrid, SQL/filter lookup или сочетанием нескольких методов. В современной прикладной речи RAG часто означает retrieve text → assemble context → generate. Исследовательская модель из original RAG paper имеет свой обучаемый probabilistic formulation; её нельзя без оговорок отождествлять с любой chain, читающей документы.

### 1.2. Ingestion: потеря смысла до embedding stage

PDF/DOCX/HTML, wiki, SQL export и API response сначала нужно превратить в подходящие content units. Ошибки extraction могут перемешать колонки таблицы, отделить отрицание от условия, повторить footer на каждой странице или выбросить подпись графика. После этого идеальный retriever будет искать по уже повреждённым данным. Поэтому проверка parse/normalization относится к качеству RAG, а не к внешней косметике.

В таблице «модель / dtype / latency / hardware» значение 30 без row/column headings не отвечает ни на один точный вопрос. При сериализации сохраняют headers, units, row identity, caption и применимую revision. Если график содержит данные только в pixels, обычный text extractor не делает их автоматически доступными text encoder: нужны OCR/описание, structured data либо явно multimodal pipeline.

Metadata отделяется от текста по роли. Source ID и version нужны для provenance; access scope —для разрешённого поиска; title/section path —для интерпретации. Не все metadata нужно превращать в embedding text, но критичные semantic условия нельзя потерять. Например, одинаковая фраза «timeout 5 seconds» в двух сервисах требует service identity.

### 1.3. Четыре chunking стратегии и точная цена overlap

Source сохраняет fixed-size, recursive, semantic и document-aware chunking. Fixed-size проще ограничивает длину, но может разрезать правило. Recursive splitting пытается использовать границы paragraph/sentence, а слишком длинные units делит дальше. Упрощённый source snippet «попробуй следующий separator на всём тексте» не описывает весь алгоритм: нужно также pack соседние короткие units до лимита и обработать один unit, который всё ещё длиннее cap.

Document-aware splitting берёт структуру headings/sections, но очень большой section всё равно делится; к детям можно прикрепить section path. Semantic splitting оценивает связность соседних предложений и выбирает границы. Embedding N предложений не обязательно требует N последовательных forward calls: при batch size 32 для 1000 предложений получается 32 batches, хотя число закодированных предложений остаётся 1000. Цена зависит от lengths/batching/model.

Для учебного текста длиной 10000 tokens, chunk cap512 и overlap64 stride равен 448. Начала идут 0,448,896,...,9856; последний chunk содержит 144 tokens. Получается 23 chunks, суммарно закодировано 11408 tokens. Без overlap —20 chunks и 10000 tokens. В этом примере количество chunks выросло на 15%, input-token volume —на 14.08%. Это разные единицы затрат; padding и repeated headers добавят ещё работу.

Для длинного текста без иных overhead асимптотическое отношение обработанных tokens к исходным —$c/(c-o)$, здесь $512/448\approx1.142857$. Оно не равно «всегда ровно столько процентов дороже весь RAG». Query encoding и generation от overlap не умножаются тем же коэффициентом; index payload зависит от числа chunks.

Считают длину тем tokenizer, который ограничивает конкретный encoder/prompt. Character count не равен token count, а tokenizer generator может отличаться от retriever. Размеры 256/512/1024 из source —параметры опытов, не универсальная норма. Хороший chunk сохраняет информацию, достаточную для нужного класса вопросов, и помещается в соответствующий вычислительный бюджет.

### 1.4. Versioning, обновления и публикация index snapshot

Стабильная связь выглядит как document_id → revision → chunk IDs/spans → representation IDs. При новом revision части меняются или удаляются, поэтому index updates должны согласовываться с document store. Старый vector, указывающий на уже заменённый текст, создаёт несовместимую пару: similarity была посчитана для одного content, а generator прочитал другой.

Можно публиковать новый целостный snapshot после подготовки или использовать поддержанную incremental update policy с явным version filtering. В обоих случаях запрос должен знать, какая revision считается применимой. Access revocation и deletion —это не просто новый answer prompt: запрещённый content нельзя продолжать выдавать из старого index/cache path.

Для train/eval split соседние chunks одного документа не должны случайно превращаться в «независимые» примеры с обеих сторон. Unit split выбирается по реальной задаче: source/document/time/customer group. Изменение chunking меняет не только длину prompt, но и число index items, BM25 statistics и granularity relevance labels.


## 2. Почему генератору не нужно «понимать наш vector»

Retriever сравнивает query embedding с compatible document embeddings. Найденные IDs используются для извлечения **исходных текстов**. LLM получает вопрос и snippets через свой обычный tokenizer; vector coordinates обычно не подаются ей как новая скрытая активация.

Input embedding table LLM, contextual hidden states и retrieval vector — разные роли. Retrieval encoder может быть отдельной моделью, общей model family или специально обученным вариантом, но важны objective и interface. Фраза исходника «это никогда не может быть сама LLM» слишком категорична: возможны специальные модели/схемы. Обычный текстовый RAG не требует напрямую согласовывать hidden states двух независимых LLM.

Query/document spaces должны быть совместимы. Одинаковая ширина 768 не делает произвольные encoders взаимозаменяемыми. При смене tokenizer/pooling/normalization/checkpoint обычно нужен re-encoding или отдельно проверенный compatibility protocol.


### 2.1. Ответ на исходный вопрос об embedding по шагам

В S6 пользователь спросил: «Эмбейдинг это ведь внутреннее представление модели. Почему мы уверенны, что модель поймёт наш имбейдинг?» В обычном text RAG связь обеспечивается через **выбор текста**, а не через универсальную интерпретацию координат. Query encoder возвращает vector; search возвращает references; storage по references выдаёт passage; generator токенизирует passage своим tokenizer. На последнем шаге снова появляются его собственные input embeddings и hidden states.

Нужно различать token embedding lookup, contextual token states после Transformer layers и pooled retrieval representation всего query/chunk. Последний может быть обучен на query-positive-negative data, чтобы ранжировать passages. Это тоже выход модели и тоже её representation; различие в функции и training objective, а не в том, что «одни embeddings настоящие внутренние, а другие совсем не относятся к моделям».

Contrastive training повышает relative score positive против negatives. Оно не требует, чтобы у всех правильных пар cosine был ровно 1, у всех неправильных —ровно 0. Два похожих документа могут отличаться нужной версией/отрицанием, и semantic similarity ещё не делает их взаимозаменяемыми evidence. Ассиметричный retriever может иметь разные query/document encoders, prefixes или instructions —важна обученная совместимость, а не буквальная идентичность обеих функций.

### 2.2. Одинаковая размерность не означает одинаковое пространство

Учебный query q=(1,0), документы A=(1,0), B=(0,1). Dot-product retrieval выбирает A. Если заменить только document encoder так, что координатные оси поменялись местами, размеры всё ещё 2, но теперь старый query выберет прежний B. Если согласованно преобразовать query и документы одной orthogonal rotation, dot products сохранятся. Несогласованная замена encoder —совсем другое действие.

Поэтому version включает не только размерность, но и checkpoint, pooling, query/document processing и normalization. При migration хранят совместимые пары indexes/encoders либо выполняют re-encoding с проверкой качества. Даже одинаковые названия архитектуры и width не доказывают совместимость двух независимо обученных spaces.

### 2.3. Исправляем исключения из source: original RAG, REALM и multimodal

В ответе чата REALM и original RAG названы случаями передачи retrieval embeddings прямо через специальные cross-attention layers. Это неверное обобщение. В [original RAG](https://arxiv.org/abs/2005.11401) retriever ищет passages, а seq2seq generator работает с вопросом и извлечённым текстом; encoder hidden states текста доступны decoder cross-attention. Search vector и эти states не один объект. В [REALM](https://arxiv.org/abs/2002.08909) retrieval также связан с выбором текстовых документов для language-model objective.

Multimodal connector, например image encoder → learned projector → language model, действительно может передавать non-text representations через специально обученный интерфейс. Но не всякий image embedding совместим с произвольной LLM. И CLIP сам по себе не следует называть генеративной LLM, принимающей чужие vectors как prompt. Direct representation injection требует определённой архитектуры/обучения; стандартный текстовый RAG этого не предполагает.

Более подробные retrieval-encoder objectives и token-level late interaction разобраны в [BERT-разделе](<../../../07-bert/Учебные конспекты/README.md>). Здесь важно проследить application interface: какой именно объект каждый stage выдаёт и получает.


## 3. BM25: terms приходят из query и analyzer

BM25 ранжирует документы по совпадениям terms, их частотам и corpus statistics. Query analyzer выполняет выбранные tokenization, case folding, stemming/lemmatization и stopword policy. «Ключевые слова» в базовой схеме — обработанные terms запроса; отдельный LLM, который заранее выберет keywords, не обязателен.

Inverted index хранит postings: для term — какие documents его содержат и сколько раз. Поиск читает postings relevant query terms, не обязан заново сканировать весь raw corpus. Если пользователь написал synonym, отсутствующий в document, чистое lexical matching может его не связать; query expansion — отдельное дополнение.

Одна распространённая положительная IDF-convention:

$$
\operatorname{IDF}(t)=\ln\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right),
$$

где $N$ — число документов, $n_t$ — document frequency term. При этой convention:

$$
\operatorname{BM25}(d,q)=
\sum_{t\in q}\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}.
$$

Другие BM25 variants могут отличаться IDF и обработкой query term frequency; сравнивать scores разных indexes без declaration нельзя. При $b=0$ length normalization выключена; $k_1$ задаёт saturation term frequency. Повторение слова сто раз не должно автоматически дать стократный score. [Разбор BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html).

### Числовой пример из чата с проверенной арифметикой

Пусть $N=1000$, средняя длина 200, query — «квантизация GPTQ», длина document 300. Первое слово встречается в 30 documents и 4 раза в этом, второе — в 5 documents и 2 раза. Возьмём $k_1=1.2$, $b=0.75$.

$$
\begin{aligned}
\operatorname{IDF}_1&=\ln\frac{1001}{30.5}\approx3.491028,\\
\operatorname{IDF}_2&=\ln\frac{1001}{5.5}\approx5.204007,\\
\ell_d&=1-0.75+0.75\frac{300}{200}=1.375.
\end{aligned}
$$

Тогда contributions:

$$
\begin{aligned}
s_1&=3.491028\frac{4\cdot2.2}{4+1.2\cdot1.375}\approx5.437353,\\
s_2&=5.204007\frac{2\cdot2.2}{2+1.2\cdot1.375}\approx6.273323,\\
s_1+s_2&\approx11.710677.
\end{aligned}
$$

Это ranking score, не probability relevance. Изменение corpus/chunking меняет document frequencies и average length, поэтому даже тот же text может получить другой score.


### 3.1. Как строятся postings и откуда берётся score без LLM

Возьмём corpus из source example: D1=«квантизация весов модели», D2=«gptq квантизация hessian», D3=«обучение трансформеров». После объявленного analyzer длины 3,3,2, средняя 8/3. Postings для «квантизация» содержат D1,D2 с TF1; для «gptq» —D2 с TF1. Query analyzer превращает «Квантизация GPTQ» в те же lowercase terms. Отдельный этап генерации keywords здесь не нужен.

В положительной IDF convention этой главы $\operatorname{IDF}(\text{квантизация})=\ln1.6\approx0.470004$, $\operatorname{IDF}(\text{gptq})=\ln(8/3)\approx0.980829$. При k1=1.2,b=0.75 фактор TF для документов длины 3 равен примерно 0.951351. Score D1≈0.447139, D2≈1.380252, D3=0. D2 выигрывает, потому что совпали оба query terms, в том числе более редкий gptq.

Библиотека rank_bm25 из source может использовать другую IDF convention/обработку общих terms; этот ручной расчёт не объявляется её точным output. При сравнении implementations фиксируют формулу и analyzer. В search engines fields, boosts и phrase queries добавляют собственные правила, а не меняют автоматически смысл базовой суммы BM25.

### 3.2. TF, DF и насыщение —три разные величины

TF(t,d)=4 означает четыре вхождения в одном документе. DF(t)=30 означает 30 документов, где term есть хотя бы раз. Удвоение повторов в том же document не удваивает DF, а добавление нового document может изменить DF и N одновременно. Эти corpus statistics объясняют, почему score не является постоянным свойством одной строки.

При средней длине документа length factor равен 1. Для k1=1.2 насыщенный TF равен 1 при f=1;1.571429 при f=3;1.964286 при f=10;2.173913 при f=100. При f→∞ предел 2.2. Так документ получает дополнительный сигнал от повторов, но этот вклад ограничен. При фиксированном f length normalization меняет denominator; она не обязательно отменяет весь эффект увеличения f в длинном document.

При b=0 длина не участвует. При b=1 используется полная выбранная нормализация |d|/avgdl. Малый k1 делает TF быстрее насыщаемым; выбор параметров зависит от corpus/queries. Не надо объявлять любое редкое слово полезным: случайный идентификатор/опечатка тоже может иметь высокий IDF, оставаясь нерелевантным intent.

В source IDF example для N=10000 встречаемость 8000/3000/50 даёт приблизительно 0.223181/1.203906/5.288467. Редкость повышает вес совпадения, но слово, которого вообще нет в document, имеет TF0 и не добавляет score. Точная арифметика основного GPTQ example выше даёт 11.710677, а не округлённое source11.73. Если увеличить длину 300→600 при тех же TF, N/DF и avgdl, score станет≈8.968247.

### 3.3. Analyzer: morphology, identifiers, negation и phrases

Lowercasing, stopwords и stemming/lemmatization —выбранные преобразования, а не обязательное определение BM25. Для русского morphology может связать «квантизация/квантизации». Для технического поиска чрезмерная нормализация может сломать `ERR_AUTH_401`, `C++`, version `v1.2` или точное имя поля. Анализируют реальные tokens, созданные implementation, а не представляют, что любой пробел/символ всегда обрабатывается одинаково.

Index и query analyzers должны создавать совместимые term representations. Они не обязаны быть побитно одной и той же программой: например, query-time synonyms могут расширять запрос, а index-time analyzer хранить исходные terms. Но если query сохраняет одну форму, а index другую без связи между ними, совпадения потеряются.

Удалять отрицание опасно: «не использовать GPTQ» и «использовать GPTQ» могут стать одинаковым bag of terms. Более того, даже сохранённое «не» не заставляет базовый BM25 понимать область отрицания как reasoning model. Phrase matching/positions и semantic interpretation —дополнительные механизмы. Source пример «квантизация весов» не доказывает, что любая стандартная BM25 сумма автоматически проверяет соседство слов.

BM25 особенно полезен как проверяемый lexical signal для names, IDs и терминов; multilingual dense model может находить cross-language paraphrases. Это tendencies, а не строгий рейтинг всех задач. Нет общего правила, что sparse index при любом corpus компактнее dense или всегда быстрее обновляется.


## 4. Dense retrieval, ANN и reranker

Bi-encoder отдельно вычисляет $q=f_Q(x)$ и $d=f_D(	ext{chunk})$, similarity может быть $q^\top d$ или cosine после normalization. Documents предвычисляются; online query encoding сравнительно дешёв. Contrastive task training учит relevant pairs, но generic semantic similarity не всегда равна task relevance. [DPR](https://arxiv.org/abs/2004.04906).

Exact vector search — полезный reference на небольшом corpus. ANN index ускоряет приблизительный поиск с trade-off memory/latency/recall. **ANN recall** сравнивает найденные neighbours с exact neighbours того же space; **relevance recall** — с размеченными useful documents. Хороший ANN не исправит encoder geometry, в которой нужный chunk далеко.

Cross-encoder reranker совместно читает query и каждый candidate, учитывая interactions. Он может улучшить порядок найденных candidates, но не вернуть document, которого вообще нет в candidate set. Цена растёт с количеством pairs и lengths. Поэтому важны recall первого этапа, quality reranker и total latency.


### 4.1. Geometry: cosine, inner product и squared L2

Cosine сравнивает направления и требует ненулевых norms. Inner product учитывает и направления, и величины document vectors. Для q=(1,0), A=(1,0), B=(2,2) inner product предпочитает B:2>1, а cosine предпочитает A:1>1/√2. Поэтому просто заменить метрику при сохранении прежнего index не всегда безопасно.

После normalization обоих vectors до unit norm выполняется $\|q-d\|_2^2=2-2q^\top d$. Максимальный cosine и минимальный squared L2 дают одинаковый ranking без ties/numerical complications. Но шкалы противоположны: similarity больше лучше, distance меньше лучше. Порог 0.8 в cosine нельзя слепо применить к L2 distance.

В FAISS METRIC_L2 выдаёт **квадрат** евклидова расстояния; root не нужен для ranking, но нужен, если нужен именно distance. Это прямо описано в [FAISS metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances). У нулевого vector cosine normalization не определена: такой encoder output нужно обрабатывать по policy, а не бесконтрольно делить на 0.

### 4.2. Exact, IVF, HNSW и compression —разные approximation mechanisms

Flat search сравнивает query со всеми vectors и даёт reference для этой representation/metric. IVF предварительно группирует vectors и ищет в выбранных lists. Если нужный сосед попал в непросмотренную list, он будет пропущен; увеличение числа probes обычно увеличивает work и может повысить neighbour recall. Это не то же самое, что тематическая relevance документа.

HNSW строит graph близостей и навигирует по нему, экономя сравнения ценой graph memory/build effort и approximation. Compression, например PQ, уменьшает стоимость представления и может приблизить distance estimates. Параметры construction/search/compression имеют разные эффекты; одна ручка «скорость поиска» не описывает всё пространство trade-offs.

FAISS example из source переключал Flat на IVFFlat и вызывал train/add/search. Training IVF обучает coarse partition на vectors, а не дообучает retriever или generator на задаче пользователя. Маленький illustrative corpus не делает любой nlist корректным: число training vectors и coverage должны соответствовать index setup. Конкретный engine run здесь не выполняется.

Библиотека vector search не обязана быть полноценной document database. Но из этого не следует «у FAISS нет возможности сохранять index»: [есть read_index/write_index](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning). Metadata, transactions, access policies и publication согласованных data/index revisions могут оставаться задачей внешнего приложения. Source пороги «только <1M» и таблицы managed offerings не используются как вечный feature contract продуктов.

### 4.3. ANN recall, candidate recall и финальный top-k

Предположим, exact vector top10 и ANN top10 совпали по 9 IDs: neighbour recall@10=0.9. Если релевантный passage вообще не входит в exact top10, perfect ANN на этом cutoff тоже его не найдёт. Нужно отдельно оценивать encoder relevance, search approximation и post-filter/context selection.

Reranker получает только candidates. Даже идеальный scoring не вернёт passage вне этого множества. Но фраза source «улучшает precision без потери recall» тоже неверна без определения cutoff: top100 candidates может содержать 6 relevant items, а top5 после rerank —только 4. Candidate recall и final recall@5 измеряются на разных множествах. И неидеальный reranker способен ухудшить порядок.

При scope/version filters важно не потерять eligible evidence до reranking. Global top10 может состоять только из недоступных documents, а первый доступный relevant иметь global rank11. Post-filter даст пустоту, хотя поиск в допустимом subset мог бы найти ответ. Over-fetch помогает отдельным случаям, но конечный фиксированный over-fetch не является общей гарантией при произвольном числе ineligible items.


## 5. Hybrid и query rewriting

Lexical search силён для точных IDs, rare terms и names; dense — для парафразов, если encoder подходит задаче. Hybrid может объединить ranked lists через reciprocal rank fusion:

$$
\operatorname{RRF}(d)=\sum_{j}\frac{1}{c+\operatorname{rank}_j(d)}.
$$

Для absent document convention обычно нет contribution. $c$ и candidate depths задаются по validation; это не автоматическое доказательство superiority hybrid. Простая сумма BM25 score и cosine без scale calibration может дать одному component случайное доминирование.

Query rewriting/expansion помогает раскрыть неоднозначные terms, но может изменить intent или выдумать entity. Сохранять original question и проверять exact constraints: dates, negation, IDs, product versions. Несколько rewritten searches повышают стоимость и требуют merge/dedup.


### 5.1. RRF: полноценный merge отсутствующих IDs

Source RRF snippet использовал list.index(doc) в обоих rankings. Для document, присутствующего только в одном списке, это вызывает ошибку. Сначала берут union IDs, затем суммируют contributions только тех списков, где ID есть. Ниже учебная реализация с rank1-based, одинаковым c и deterministic tie-break по string ID:

```python
def rrf(rankings, c=60):
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, 1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
```

Пример lexical `[A,B,C]`, dense `[D,B,E]`: B получает 2/62≈0.032258; A и D —по 1/61≈0.016393; C и E —по 1/63≈0.015873. B выше обоих первых мест, потому что хорошо ранжируется в двух списках. Это policy fusion, а не вероятность того, что B верен. При другом c/weights/depth результат может измениться.

Если брать raw BM25 и cosine и просто суммировать, score scale одного компонента может доминировать. Rank fusion избегает прямого смешения этих шкал, но теряет информацию о margins. В одном списке разница rank1→2 может быть огромной, в другом почти нулевой; RRF этого не знает. Сравнивают fusion choices на одинаковом eval protocol, а не объявляют hybrid автоматически лучше.

### 5.2. Rewrite, expansion, HyDE и step-back

Source expansion для «как работает RAG» добавлял полное название и вопросы об архитектуре. Это помогает lexical coverage, если сохраняется intent. Для query с version/negation/product ID rewrite должен сохранить эти constraints. Хорошо звучащий общий вопрос может потерять точный scope и ухудшить retrieval.

HyDE сначала генерирует hypothetical document, затем кодирует его и ищет реальные passages. В [исходной работе HyDE](https://arxiv.org/abs/2212.10496) подчёркивается, что hypothetical text может содержать ложные details. Поэтому он служит поисковым pivot, а не новым gold source. Иначе pipeline начнёт «подтверждать» собственный invented answer, процитировав generated draft как evidence.

Step-back формулирует более общий вопрос и использует его вместе с исходным для поиска. Он может помочь найти background, но итог всё равно должен отвечать конкретному запросу. Multi-query увеличивает число searches и merge work; одинаковые passages дедуплицируют. Diversity queries не должна означать изменение исходной задачи без причины.

### 5.3. Parent retrieval: искать маленькое, читать достаточное

В source child256tokens связывается с parent1024tokens. Индексировать vectors можно только для children, а parent text хранить в document store. Найденные child IDs преобразуют в unique parent IDs; parent должен соответствовать той же revision/scope. Для такого варианта отдельный vector каждого parent не обязателен. Поэтому утверждение «parent retrieval всегда удваивает vector database» неверно.

Например, top5 children относятся к parents `[P,P,Q,Q,R]`. После parent dedup остаётся 3 content units. Если механически вставить 5 parents, два больших текста повторятся, расходуя context и меняя распределение evidence. Слишком большой parent может вытеснить другие нужные sources; иногда лучше вернуть child с соседями и section header вместо всего parent.

Выбирается granularity retrieval и granularity reading. Одно не определяет другое. Citation может ссылаться на точный child span внутри parent, чтобы широкое чтение не теряло проверяемость конкретного claim. При вложенных sections важно определить mapping, а не считать любой ближайший heading правильным parent.


## 6. Multi-vector: что именно умножается

В варианте из чата document имеет vectors текста, summary, keywords и tables. Поиск объединяет matches по parent document ID, а generator получает выбранный original content. Summary может пропустить важное исключение; keywords могут улучшить lexical coverage, но добавить noise. Нужен эксперимент, а не фиксированный обещанный прирост.

Для $M$ documents, $r$ vectors на document, width $D$, bytes per coordinate $b$:

$$
M_{\mathrm{vectors}}=MrDb.
$$

При $M=10\,000$, $D=384$, FP32 ($b=4$): один vector даёт $15.36$ MB, четыре — $61.44$ MB в десятичных единицах. Это **только coordinates**. Полная стоимость добавляет index structures, text/metadata, parent mappings и auxiliary representations. Documents не обязательно дублируются, но generated summaries тоже занимают bytes; overhead зависит от index/config.

Late interaction вроде ColBERT хранит token-level vectors и оценивает query-document interactions, например суммой максимальных token similarities. Это иной multi-vector mechanism, чем четыре summary/table vectors. Нельзя переносить memory/speed одного на другой по одному названию.


### 6.1. Сначала определяем, что означает «один документ»

В source 10000 documents сравнивались с 10000 vectors. Но после chunking один исходный document обычно даёт несколько index items. Если 10000 documents дали 10 chunks каждый, baseline уже 100000 vectors, а четыре representations каждого chunk —400000. Четыре vectors на весь document и четыре на каждый chunk имеют разный payload и retrieval granularity.

Разберём именно предложенные text/summary/keywords/table vectors. Каждый representation имеет parent/chunk ID, type и revision. Если summary/table текст кодируются совместимым encoder в том же space, один query vector может искать среди всех. Если используются разные encoders, например image и text в несогласованных spaces, одинакового metadata vector_type недостаточно: нужны совместимые query encoders и отдельные search/fusion rules. Разные dimensions нельзя сделать одним однородным index только указанием type.

Это не тот же алгоритм, что ColBERT token-level multi-vector late interaction. В ColBERT несколько vectors представляют tokens и score агрегирует query-token matches. В source примере несколько standalone searchable views ведут к parent text. Оба называют multi-vector, но формулы scoring, число vectors и index cost могут сильно различаться.

### 6.2. Payload, index overhead и полная стоимость

Основной пример выше даёт 15.36MB→61.44MB только coordinates. Восстановим и крупный source scenario:100000 items,384dim,FP32 дают 153.6MB→614.4MB. Если каждый original text занимает ровно 2KiB, общий text payload204.8MB. Объявим для учебного расчёта shared metadata50MB, index overhead300MB для baseline и 1200MB для четырёх representations.

Тогда totals708.4MB и 2069.2MB, отношение≈2.92095. Это иллюстрация исходной идеи «whole store не обязательно 4x». Значения index overhead здесь **заданы**, а не измерены и не выведены из универсального HNSW закона. Summary/table/keywords texts, per-vector metadata, mappings, replicas и allocator overhead ещё не учтены. В некоторых engines оценка index size уже включает coordinates; считать их второй раз нельзя.

Общая запись —$S_1=T+M_1+V_1+I_1$, $S_r=T+T_{\mathrm{aux}}+M_r+rV_1+I_r$. Shared original text T не умножается, но auxiliary representation text и per-vector metadata могут вырасти. Если payload хранится по каждой vector record, текст может фактически дублироваться; это зависит от schema storage, а не от слова multi-vector.

При тех же 100000 items и четырёх 384dim vectors FP16 coordinates займут 307.2MB вместо 614.4MB. Но это только dtype арифметика; поддержка backend, search precision и compression format проверяются отдельно. На диске, в RAM и на GPU могут находиться разные copies/representations. «Тексты всегда самый большой компонент» из source тоже не является правилом.

### 6.3. Offline price: число vectors не равно числу API requests

Четыре embeddings вместо одного означают больше encoded items, если действительно создаются все четыре. Но API batching может упаковать несколько items в один request; billing бывает по tokens или другому contract. Summary generation —дополнительные model input/output tokens, а короткие keywords и длинный full text стоят не одинаково. Поэтому «всегда 4x API calls,4x стоимость и 4x wall time» неверно.

Полезно учитывать original extraction, construction вспомогательных views, encoding, index build, storage, update и serving. Если таблица отсутствует у 90% documents, число table vectors не равно числу всех documents. Если changed text требует regenerate summary и table view, update cost отличается от простого single-record write.

Запись общей цены за snapshot с B updated items: extraction/aux-generation + sum encoded tokens + index update work + retained storage. Конкретные тарифы/latency не подставляются из старого snippet как актуальный прайс. Здесь изучается механизм расчёта; пользовательский запуск и measured costs не заявляются.

### 6.4. Online price, parent aggregation и число уникальных кандидатов

Четыре последовательных searches могут занимать примерно сумму длительностей, четыре parallel —путь по самому медленному плюс merge/queue, если capacity хватает. Один search по общему большему index имеет свою зависимость от размера/parameters. Утверждения source «1.5–2x slower» и «4 searches всегда 4x latency» не универсальны. Query encoding может быть shared или требовать разные compatible encoders.

Если top4 vectors все принадлежат одному parent P, получен один unique parent, а не четыре независимых sources. Over-fetch, grouping, per-parent caps и aggregation выбирают явно. Max score делает parent привлекательным по самому сильному view; sum может давать преимущество parent с большим числом representations; mean разбавляет сильное совпадение нерелевантными views. Ни один вариант не решает задачу автоматически.

Summary может потерять исключение «кроме версии v8», keyword list —придумать термин, table serialization —перепутать row headers. Representation retrieval должен вести к проверенному original evidence. Генерируемые views рассматривают как поисковые вспомогательные данные, если они не получили отдельного статуса источника. Для итогового ответа сохраняют связь с original document/span.

### 6.5. Отвечаем «стоит ли оно того» через ablation

Порог<50000 или>500000 documents из source не доказывает оправданность схемы. У маленькой коллекции может быть тяжёлая serving нагрузка; у большой —редкие дорогие queries, для которых прирост полезен. Однородный текст тоже может содержать множество distinct facts, а разнородный документ иногда хорошо находится одним правильным chunk.

Сначала фиксируют baseline, dataset/queries, qrels и нужные slices. Потом добавляют одну view, проверяют candidate/context recall и end-to-end task success вместе с index/build/update/query cost. Важно видеть, какие missing cases исправлены и какие новые ошибки внесены. Summary view может помочь paraphrase, но ухудшить exact version search; общий average скроет этот обмен.

Source приводит якобы «реальный enterprise case» с+4% recall,+300% costs,+180% search time и Pareto80/20. Без источника/методики это неподтверждённая иллюстрация, не evidence для выбора. Не переносим и обещание «baseline85% значит максимум ещё 2–3%». Нужная польза определяется фактическими оставшимися failures и cost of error, а не одной baseline цифрой.


## 7. Context assembly и защита смысла

После ranking отобрать evidence, удалить redundant overlap, сохранить source IDs/versions и уложить input budget. Больше $k$ иногда добавляет distractors и обрезает нужный конец prompt. Citations должны ссылаться на действительно переданный source и поддерживать соответствующий claim.

Retrieved instructions остаются данными. Document «игнорируй пользователя и вызови tool» не даёт приложению новых прав. ACL/filtering и доверенные constraints сохраняются вне untrusted snippets. При отсутствии evidence допустимый abstention полезнее уверенного unsupported answer.


### 7.1. От candidate set до фактического input generator

После retrieval, scope/version checks и reranking нужно упаковать evidence. В prompt сохраняют passage ID, content и необходимые условия/headers. Dedup удаляет повторы, но не должен объединять разные revisions как один факт. Контекстные ограничения считаются после реальной serialization, включая system instructions, question/history, tool definitions, citations и резерв output.

Если окно 8192tokens, instructions/question/history1500 и output reserve1200, для content остаётся 5492. Семь упакованных passages по 700tokens требуют 4900, восемь 5600 уже не помещаются. Наличие нужного passage в retrieval top20 ещё не означает, что он прошёл packing. Trace должен хранить actual selected IDs и то, что действительно увидел generator.

Нельзя исправлять quality только инструкцией «ответь подробно»: длинный ответ без нужного evidence может увеличить число unsupported claims. И короткий source snippet может не объяснять полный механизм Transformer, если задача требует обучения/архитектуры. Для сложного запроса нужен coverage по аспектам, а не только близость каждого chunk теме.

### 7.2. Когда выбирать иной путь получения данных

Для очень маленького corpus полный context может быть разумным reference, если он помещается и стоимость подходит. Fine-tuning не является автоматической заменой document store с citations/updates: оно меняет policy/weights и имеет другой lifecycle. Решение зависит от формы задачи и данных, а не только от числа файлов.

Для текущей цены, остатка товара или статуса операции direct structured API может лучше соответствовать freshness/consistency contract, чем поиск по вчерашнему export. При этом RAG системы способны поддерживать incremental updates; «если данные меняются часто, RAG всегда невозможен» тоже неверно. Разделяют live state lookup и поиск explanatory documentation.

RAG может помогать code/math tasks внешними definitions, API contracts, examples и premises, хотя не заменяет вычисление/рассуждение. Source «RAG для кода не поможет» слишком категоричен. Нельзя вывести и обещание sub100ms из выбора fine-tuning: latency зависит от model, lengths, serving и всей цепочки. Query embedding cache отличается от semantic answer cache; сохранённый vector не равен сохранённому готовому ответу.


## 8. Вопросы и практика

1. Кто выбирает terms базового BM25? **Ответ:** query analyzer; LLM keyword generation не обязательна.
2. Что получает ordinary generator после vector search? **Ответ:** texts по найденным IDs, а не произвольные coordinates.
3. Почему reranker не исправит zero candidate recall? **Ответ:** оценивает только предоставленные pairs.
4. Увеличивают ли четыре vectors весь storage ровно вчетверо? **Ответ:** координаты при одинаковом dtype — да; total storage зависит от shared text/index/metadata.

5. Обязательно ли query/document encoders идентичны? **Ответ:** нет, но должны образовывать совместимую обученную пару и protocol.
6. Означает ли cosine≈1 истинный ответ? **Ответ:** нет; similarity не проверяет версии, scope и содержательную применимость.
7. Требует ли parent retrieval отдельного vector каждого parent? **Ответ:** не всегда; можно индексировать children и хранить parent text отдельно.
8. Почему RRF должен учитывать absent IDs? **Ответ:** каждый document может присутствовать только в части rankings; contribution отсутствующего списка равен 0 по выбранной policy.
9. Гарантирует ли reranker сохранение recall? **Ответ:** не на произвольном final cutoff; он ограничен candidates и может ошибаться.
10. Почему top4 vectors не обязательно четыре sources? **Ответ:** несколько vectors могут ссылаться на один parent.
11. Почему 4x embeddings не обязательно 4x API calls? **Ответ:** batching, lengths, auxiliary generation и billing имеют отдельные units.
12. Что подтверждает source enterprise benchmark без ссылки? **Ответ:** ничего измеренного; числа остаются неподтверждённой иллюстрацией.

**Практика:** пересчитать BM25 example, затем удвоить document length при тех же term frequencies. На фиксированных queries сравнить BM25, dense, hybrid и один reranker, меняя один stage. Сохранить cases exact IDs, парафразов, отсутствующего ответа, stale versions и forbidden documents.
