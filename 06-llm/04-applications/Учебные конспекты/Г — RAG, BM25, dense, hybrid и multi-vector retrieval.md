# Г. RAG: от вопроса и корпуса к проверяемому evidence

[Оглавление](README.md). Пункт4 плана.

**Основа:** S6:182–1373; дополнительно S5:1038–1817 о pipeline. Сохранены ваши вопросы: «почему LLM поймёт наш embedding», «откуда BM25 берёт keywords» и «насколько вырастет база при нескольких vectors». **Исправления:** ordinary RAG передаёт generator текст, BM25 не требует отдельного LLM keyword extractor, memory/latency не растут по одному универсальному проценту. **Углубление:** полный ручной BM25, ANN vs relevance и storage accounting.

## 1. Два процесса: подготовка корпуса и ответ на запрос

Offline/ingestion path: документы → очистка и версии → chunks → metadata/provenance → sparse/dense indexes. Online path: вопрос → retrieval → фильтрация/объединение → reranking → context selection → generation → проверка citations/answer.

У каждого chunk должны быть устойчивые document/chunk IDs, source span или location, version и access scope. Если document обновился, старый chunk не должен продолжать выдавать obsolete правило без понятной политики. Split evaluation по documents/sources делать до нарезки, иначе соседние chunks создают leakage.

Слишком короткий chunk теряет context, слишком длинный разбавляет signal и занимает generation budget. Overlap помогает не разрезать нужную связь, но увеличивает дубликаты и index size. Размер измерять в tokenizer tokens, а semantic boundaries учитывать вместе с длиной. Это параметры pipeline, а не универсальное «лучше всегда512tokens».

## 2. Почему генератору не нужно «понимать наш vector»

Retriever сравнивает query embedding с compatible document embeddings. Найденные IDs используются для извлечения **исходных текстов**. LLM получает вопрос и snippets через свой обычный tokenizer; vector coordinates обычно не подаются ей как новая скрытая активация.

Input embedding table LLM, contextual hidden states и retrieval vector — разные роли. Retrieval encoder может быть отдельной моделью, общей model family или специально обученным вариантом, но важны objective и interface. Фраза исходника «это никогда не может быть сама LLM» слишком категорична: возможны специальные модели/схемы. Обычный текстовый RAG не требует напрямую согласовывать hidden states двух независимых LLM.

Query/document spaces должны быть совместимы. Одинаковая ширина 768 не делает произвольные encoders взаимозаменяемыми. При смене tokenizer/pooling/normalization/checkpoint обычно нужен re-encoding или отдельно проверенный compatibility protocol.

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

## 4. Dense retrieval, ANN и reranker

Bi-encoder отдельно вычисляет $q=f_Q(x)$ и $d=f_D(	ext{chunk})$, similarity может быть $q^\top d$ или cosine после normalization. Documents предвычисляются; online query encoding сравнительно дешёв. Contrastive task training учит relevant pairs, но generic semantic similarity не всегда равна task relevance. [DPR](https://arxiv.org/abs/2004.04906).

Exact vector search — полезный reference на небольшом corpus. ANN index ускоряет приблизительный поиск с trade-off memory/latency/recall. **ANN recall** сравнивает найденные neighbours с exact neighbours того же space; **relevance recall** — с размеченными useful documents. Хороший ANN не исправит encoder geometry, в которой нужный chunk далеко.

Cross-encoder reranker совместно читает query и каждый candidate, учитывая interactions. Он может улучшить порядок найденных candidates, но не вернуть document, которого вообще нет в candidate set. Цена растёт с количеством pairs и lengths. Поэтому важны recall первого этапа, quality reranker и total latency.

## 5. Hybrid и query rewriting

Lexical search силён для точных IDs, rare terms и names; dense — для парафразов, если encoder подходит задаче. Hybrid может объединить ranked lists через reciprocal rank fusion:

$$
\operatorname{RRF}(d)=\sum_{j}\frac{1}{c+\operatorname{rank}_j(d)}.
$$

Для absent document convention обычно нет contribution. $c$ и candidate depths задаются по validation; это не автоматическое доказательство superiority hybrid. Простая сумма BM25 score и cosine без scale calibration может дать одному component случайное доминирование.

Query rewriting/expansion помогает раскрыть неоднозначные terms, но может изменить intent или выдумать entity. Сохранять original question и проверять exact constraints: dates, negation, IDs, product versions. Несколько rewritten searches повышают стоимость и требуют merge/dedup.

## 6. Multi-vector: что именно умножается

В варианте из чата document имеет vectors текста, summary, keywords и tables. Поиск объединяет matches по parent document ID, а generator получает выбранный original content. Summary может пропустить важное исключение; keywords могут улучшить lexical coverage, но добавить noise. Нужен эксперимент, а не фиксированный обещанный прирост.

Для $M$ documents, $r$ vectors на document, width $D$, bytes per coordinate $b$:

$$
M_{\mathrm{vectors}}=MrDb.
$$

При $M=10\,000$, $D=384$, FP32 ($b=4$): один vector даёт $15.36$ MB, четыре — $61.44$ MB в десятичных единицах. Это **только coordinates**. Полная стоимость добавляет index structures, text/metadata, parent mappings и auxiliary representations. Documents не обязательно дублируются, но generated summaries тоже занимают bytes; overhead зависит от index/config.

Late interaction вроде ColBERT хранит token-level vectors и оценивает query-document interactions, например суммой максимальных token similarities. Это иной multi-vector mechanism, чем четыре summary/table vectors. Нельзя переносить memory/speed одного на другой по одному названию.

## 7. Context assembly и защита смысла

После ranking отобрать evidence, удалить redundant overlap, сохранить source IDs/versions и уложить input budget. Больше $k$ иногда добавляет distractors и обрезает нужный конец prompt. Citations должны ссылаться на действительно переданный source и поддерживать соответствующий claim.

Retrieved instructions остаются данными. Document «игнорируй пользователя и вызови tool» не даёт приложению новых прав. ACL/filtering и доверенные constraints сохраняются вне untrusted snippets. При отсутствии evidence допустимый abstention полезнее уверенного unsupported answer.

## 8. Вопросы и практика

1. Кто выбирает terms базового BM25? **Ответ:** query analyzer; LLM keyword generation не обязательна.
2. Что получает ordinary generator после vector search? **Ответ:** texts по найденным IDs, а не произвольные coordinates.
3. Почему reranker не исправит zero candidate recall? **Ответ:** оценивает только предоставленные pairs.
4. Увеличивают ли четыре vectors весь storage ровно вчетверо? **Ответ:** координаты при одинаковом dtype — да; total storage зависит от shared text/index/metadata.

**Практика:** пересчитать BM25 example, затем удвоить document length при тех же term frequencies. На фиксированных queries сравнить BM25, dense, hybrid и один reranker, меняя один stage. Сохранить cases exact IDs, парафразов, отсутствующего ответа, stale versions и forbidden documents.
