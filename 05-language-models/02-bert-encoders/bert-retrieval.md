# Dense retrieval, cross-encoder, SPLADE и ColBERT

**Уровень:** основа ветки.

**Предпосылки:** [Encoder, tokenizer, attention и pooling](<bert-interface.md>), [Sentence embeddings и contrastive learning](<bert-embeddings.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как выбрать место взаимодействия query и document и оценить полный budget?





## Карта вычисления и границы этапов

![Dense retrieval, cross-encoder, SPLADE и ColBERT: входы, преобразования и выходы](<../../оформление/course/bert-retrieval.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:ca7c24d15f5950861d -->
<a id="u-ca7c24d15f5950861d"></a>

## Dense retrieval: reusable document vectors

<!-- /unit:ca7c24d15f5950861d -->

<!-- unit:c9fd69fa4ac55bca5e -->
<a id="u-c9fd69fa4ac55bca5e"></a>

Query encoder и document encoder обучаются так, чтобы relevant pair имела высокий score. Shared weights возможны, но tasks asymmetric: короткий вопрос и длинный passage имеют разные distributions. Prefixes `query:`/`passage:` или инструкции берутся из конкретной model card/revision; нельзя считать их universal syntax всех encoders. [Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-base) — пример task-trained retrieval checkpoint.

<!-- /unit:c9fd69fa4ac55bca5e -->

<!-- unit:f5ecb263a7ad8d5516 -->
<a id="u-f5ecb263a7ad8d5516"></a>

Offline pipeline: source documents → permitted chunks/metadata → encoder → vectors → index. Online: query preprocessing → vector → filtered search → document aggregation. IDs, namespaces, versions и access filters должны сопровождать vectors, а не восстанавливаться по порядку строк случайно.

<!-- /unit:f5ecb263a7ad8d5516 -->

<!-- unit:1847fbecd988802e52 -->
<a id="u-1847fbecd988802e52"></a>

Exact vector search сравнивает со всеми candidates выбранного corpus; ANN приближает этот поиск. ANN recall@k — overlap с exact vector top-k, semantic relevance recall@k — найденные judged relevant docs относительно qrels. ANN может идеально воспроизвести semantic плохой vector ranking. Поэтому обе проверки нужны отдельно.

<!-- /unit:1847fbecd988802e52 -->

<!-- unit:9fb7b9ca316ed7dc6d -->
<a id="u-9fb7b9ca316ed7dc6d"></a>

Chunk→document aggregation тоже меняет ranking: max score предпочитает один сильный passage, mean — average evidence и penalizes long docs с множеством irrelevant chunks. Count multiple hits без контроля duplicates может завышать длинный документ. Cutoffs и aggregation fit на validation.

<!-- /unit:9fb7b9ca316ed7dc6d -->

<!-- unit:2d925f16b84eb10b37 -->
<a id="u-2d925f16b84eb10b37"></a>

<!-- source-depth-bert-Е -->

<!-- /unit:2d925f16b84eb10b37 -->

<!-- unit:4d068dd3941b4aa5fd -->
<a id="u-4d068dd3941b4aa5fd"></a>

### Какая работа переносится offline

<!-- /unit:4d068dd3941b4aa5fd -->

<!-- unit:c14d4cad6d59ae13eb -->
<a id="u-c14d4cad6d59ae13eb"></a>

Для N документов bi-encoder заранее вычисляет document vectors и сохраняет связь «vector row → document/chunk ID → исходный текст». Новый query кодируется отдельно, затем сравнивается с индексом. Если encoder document не зависит от query, один вектор можно использовать для многих запросов. Это переиспользование отличает retrieval от полного совместного кодирования каждой пары.

<!-- /unit:c14d4cad6d59ae13eb -->

<!-- unit:7af128a07947d0ba8e -->
<a id="u-7af128a07947d0ba8e"></a>

Prefix или instruction также часть функции encoding. Если checkpoint обучен различать query и passage через специальные инструкции, их удаление меняет распределение входов. Но это не универсальное правило всех моделей: произвольное добавление `query:` к модели, которая так не обучалась, не создаёт нужной асимметрии автоматически. Сохраняют точный pipeline вместе с revision tokenizer и encoder.

<!-- /unit:7af128a07947d0ba8e -->

<!-- unit:527409bb34f61825cf -->
<a id="u-527409bb34f61825cf"></a>

Документ может иметь несколько chunks. Пусть scores документа A равны (0.9,0.1,0.1), B — (0.7,0.7). Max предпочитает A, mean — B. Оба решения имеют понятный смысл: один сильный passage против более равномерного покрытия. При выборе parent document по child hits нужно также удалить дубликаты и учитывать, сколько исходного evidence попало в итоговый context. Сумма scores без контроля может вознаграждать число chunks, а не качество ответа.

<!-- /unit:527409bb34f61825cf -->

<!-- unit:75080cef6d3137d323 -->
<a id="u-75080cef6d3137d323"></a>

### Две проверки recall отвечают на разные вопросы

<!-- /unit:75080cef6d3137d323 -->

<!-- unit:86a9523ca8fea88cad -->
<a id="u-86a9523ca8fea88cad"></a>

Пусть exact vector top-3 — {A,B,C}, approximate top-3 — {A,B,D}. ANN overlap recall относительно exact top-3 равна 2/3. Пусть judged relevant documents для query — {C,E}. Тогда semantic recall@3 exact поиска — 1/2, approximate — 0. Даже полный overlap ANN с exact не исправит ошибочную геометрию самого encoder.

<!-- /unit:86a9523ca8fea88cad -->

<!-- unit:6752b9d874035e912c -->
<a id="u-6752b9d874035e912c"></a>

Нужны отдельные сравнения: exact search по тем же vectors для качества approximation; qrels для качества смысла; проверка corpus/filters для доступности нужного документа. Если relevant passage отсутствует в корпусе или закрыт выбранным фильтром, изменение ANN-параметров не вернёт его в candidate set. Неполные qrels требуют аккуратной трактовки unjudged documents, а не автоматического объявления их negatives.

<!-- /unit:6752b9d874035e912c -->

<!-- unit:d706aaad018ed5c681 -->
<a id="u-d706aaad018ed5c681"></a>

### Вопрос S6: «Откуда берутся keywords в BM25?»

<!-- /unit:d706aaad018ed5c681 -->

<!-- unit:73a95ba796cad51cdb -->
<a id="u-73a95ba796cad51cdb"></a>

В разбираемом примере вы просили подробно объяснить происхождение keywords. Для базового BM25 отдельный нейронный извлекатель keywords не требуется: query проходит analyzer, и его terms сопоставляются с postings документов. Analyzer может включать normalization, разбиение, stemming или другие операции. Важно согласовать document/query analysis; простой BERT WordPiece tokenizer и lexical analyzer индекса не обязаны быть одним инструментом.

<!-- /unit:73a95ba796cad51cdb -->

<!-- unit:3b580cfd00aa7dc9ff -->
<a id="u-3b580cfd00aa7dc9ff"></a>

У term t есть частота в данном документе f(t,d) и document frequency n(t), число документов, где term встречается хотя бы раз. Это разные величины: сто повторений в одном документе увеличивают f, но добавляют только один документ к n. Один распространённый вариант формулы:

<!-- /unit:3b580cfd00aa7dc9ff -->

<!-- unit:bb1c8d16a602970500 -->
<a id="u-bb1c8d16a602970500"></a>

$$
\operatorname{BM25}(q,d)=\sum_{t\in q}\log\!\left(1+\frac{N-n(t)+0.5}{n(t)+0.5}\right)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}.
$$

<!-- /unit:bb1c8d16a602970500 -->

<!-- unit:14b29c4ff000a6a5ff -->
<a id="u-14b29c4ff000a6a5ff"></a>

Повторения terms самого query, stopwords и детали IDF зависят от variant/analyzer. Это объявленный учебный вариант, а не точное поведение любой поисковой библиотеки. При положительном нормирующем коэффициенте TF-вклад насыщается: при f→∞ дробь стремится к k1+1, а не растёт неограниченно. При b=0 отключена указанная коррекция длины, при b=1 она полностью зависит от отношения |d|/avgdl.

<!-- /unit:14b29c4ff000a6a5ff -->

<!-- unit:929893cb150517a735 -->
<a id="u-929893cb150517a735"></a>

Восстановим числа S6: N=1000, avgdl=200, |d|=300, query «квантизация GPTQ». Для первого term f=4,n=30; для второго f=2,n=5; k1=1.2,b=0.75. Length factor равен 1.375, поэтому k1×factor=1.65. IDF равны примерно 3.491028 и 5.204007. TF-дроби — 8.8/5.65≈1.557522 и 4.4/3.65≈1.205479. Итоговый score≈11.710677.

<!-- /unit:929893cb150517a735 -->

<!-- unit:3b525dde894cf4117b -->
<a id="u-3b525dde894cf4117b"></a>

В разбираемом примере получилось около 11.73 из-за раннего округления; здесь сохранены промежуточные значения до финального шага. Это score выбранной формулы, не вероятность релевантности. Редкое слово может быть ошибкой или случайным кодом, поэтому высокий IDF сам по себе не доказывает, что документ отвечает запросу.

<!-- /unit:3b525dde894cf4117b -->

<!-- unit:540e06580be455b59c -->
<a id="u-540e06580be455b59c"></a>

## Cross-encoder reranking

<!-- /unit:540e06580be455b59c -->

<!-- unit:90f0fd83fe54907e6d -->
<a id="u-90f0fd83fe54907e6d"></a>

Cross-encoder кодирует query+candidate jointly и возвращает relevance score. Внутреннее attention даёт token interactions до head, а не только сравнение готовых vectors. Такой model обычно применяется к ограниченному candidate set: он не вернёт missing document из-за первого этапа.

<!-- /unit:90f0fd83fe54907e6d -->

<!-- unit:e1dda27d42ffeadf6f -->
<a id="u-e1dda27d42ffeadf6f"></a>

Pointwise BCE обучает relevance каждого pair. Pairwise logistic loss усиливает positive относительно negative:

<!-- /unit:e1dda27d42ffeadf6f -->

<!-- unit:3cf4b7f203c46e0874 -->
<a id="u-3cf4b7f203c46e0874"></a>

$$
L_{\mathrm{pair}}=\log\left(1+e^{-(s(q,d^+)-s(q,d^-))}\right).
$$

<!-- /unit:3cf4b7f203c46e0874 -->

<!-- unit:eef5172eca26b0df9f -->
<a id="u-eef5172eca26b0df9f"></a>

При scores 2 и 1 margin 1, loss около 0.313262. Если negative на деле тоже relevant, loss создаёт ложное ordering pressure. Listwise objective сравнивает весь list одного query, например CE по normalized target grades/teacher distribution; grading convention и incomplete judgements важны. [CrossEncoder training](https://sbert.net/docs/cross_encoder/training_overview.html).

<!-- /unit:eef5172eca26b0df9f -->

<!-- unit:00469a299d2dc4914d -->
<a id="u-00469a299d2dc4914d"></a>

Score может быть ranking logit, а не calibrated relevance probability across queries. Если threshold должен означать «ответ отсутствует», нужна отдельная calibration на representative queries, включая no-answer. Кандидаты k=100 дают примерно в пять раз больше pair encodings, чем k=20, при одинаковой длине; actual latency зависит от batching/padding/queue.

<!-- /unit:00469a299d2dc4914d -->

<!-- unit:1908adf64aec56191c -->
<a id="u-1908adf64aec56191c"></a>

Evaluation фиксирует first-stage candidates и проверяет incremental ranking gain, candidate recall ceiling, total latency и truncation. Split по queries/documents/groups, а не случайным pair rows одного query.

<!-- /unit:1908adf64aec56191c -->

<!-- unit:c91be36adb38905501 -->
<a id="u-c91be36adb38905501"></a>

### Где появляется детальное взаимодействие query и документа

<!-- /unit:c91be36adb38905501 -->

<!-- unit:d7d9c2323d9260f316 -->
<a id="u-d7d9c2323d9260f316"></a>

Cross-encoder получает общую последовательность query+document и читает её bidirectional attention. Состояние слова документа может зависеть от вопроса ещё до final head. У bi-encoder такое взаимодействие должно быть выражено независимо вычисленными vectors и выбранным score. У cross-encoder doc states нельзя заранее закешировать как один query-independent ответ для любой пары.

<!-- /unit:d7d9c2323d9260f316 -->

<!-- unit:36cc22be9416c30d30 -->
<a id="u-36cc22be9416c30d30"></a>

В упрощённом сравнении без special tokens query length=20, document length=100. Совместный dense attention имеет 120²=14400 пар, два независимых массива — 20²+100²=10400. Разница 4000 — cross-segment взаимодействия. Полная цена также включает ширину, слои, MLP и переиспользование document encoding. Этот подсчёт показывает механизм, но не является измерением latency.

<!-- /unit:36cc22be9416c30d30 -->

<!-- unit:227089d3a0e8625447 -->
<a id="u-227089d3a0e8625447"></a>

Обычно reranker применяется к небольшому pool после retrieval. Если из пяти relevant документов first stage вернул только три, максимальный recall по этим пяти остаётся 3/5 даже у идеального reranker. Он может поднять все три на первые места и улучшить precision/ranking, но два отсутствующих документа ему недоступны. Поэтому сначала измеряют candidate recall, затем качество перестановки внутри pool.

<!-- /unit:227089d3a0e8625447 -->

<!-- unit:d1602fc95c1c89d582 -->
<a id="u-d1602fc95c1c89d582"></a>

### Pairwise loss учит относительный margin

<!-- /unit:d1602fc95c1c89d582 -->

<!-- unit:3a842ff93e26faa2d0 -->
<a id="u-3a842ff93e26faa2d0"></a>

При Δ=s+−s− pairwise loss равна softplus(−Δ), derivative по Δ — $-\sigma(-\Delta)$. Для Δ=1 это примерно −0.268941. Derivative по positive score отрицательная, по negative — положительная; gradient descent увеличивает margin. Если оба документа релевантны и их порядок не задан supervision, такой gradient может навязывать ненужное различие.

<!-- /unit:3a842ff93e26faa2d0 -->

<!-- unit:aaef41de34ad94b4ce -->
<a id="u-aaef41de34ad94b4ce"></a>

Добавление одной константы к обоим scores не меняет Δ. Поэтому pairwise ranking loss сам по себе не задаёт абсолютной шкалы вероятности «документ релевантен». Pointwise BCE и listwise objectives дают другие constraints. Если downstream нужно выбрать no-answer по порогу, калибровку проводят отдельно на representative queries, включая случаи без ответа.

<!-- /unit:aaef41de34ad94b4ce -->

<!-- unit:af4faf898cca8f9c1b -->
<a id="u-af4faf898cca8f9c1b"></a>

Кандидаты training должны отражать нужную трудность: только случайные документы могут сделать задачу слишком простой, а ошибочно размеченные hard negatives — противоречивой. Для teacher-student distillation candidate distributions и нормировки сохраняются явно. При evaluation фиксируют одинаковый first-stage pool, чтобы gain нельзя было объяснить незаметной заменой candidates.

<!-- /unit:af4faf898cca8f9c1b -->

<!-- unit:b2b6aa3c15c810ba31 -->
<a id="u-b2b6aa3c15c810ba31"></a>

## Learned sparse: vocabulary weights

<!-- /unit:b2b6aa3c15c810ba31 -->

<!-- unit:07114f26953e45a82f -->
<a id="u-07114f26953e45a82f"></a>

SPLADE-like representations используют vocabulary-space signals encoder и нелинейное pooling. Один распространённый вариант:

<!-- /unit:07114f26953e45a82f -->

<!-- unit:db5607dcb5d815f195 -->
<a id="u-db5607dcb5d815f195"></a>

$$
v_j=\max_t\log(1+\operatorname{ReLU}(\ell_{t,j})),
\qquad
s(q,d)=\sum_j v_j^{(q)}v_j^{(d)}.
$$

<!-- /unit:db5607dcb5d815f195 -->

<!-- unit:b7fd62898f8051a299 -->
<a id="u-b7fd62898f8051a299"></a>

Token logits превращаются в nonnegative term weights, включая learned expansion terms. Это не merely dense vector, в котором случайно много zeros: vocabulary coordinates имеют lexical interpretation и используются inverted index. Variant pooling/query architecture/regularization различаются. [SPLADE](https://arxiv.org/abs/2107.05720).

<!-- /unit:b7fd62898f8051a299 -->

<!-- unit:049bbef8fc195b1854 -->
<a id="u-049bbef8fc195b1854"></a>

Toy two-position logits rows $(1,-1,0)$ и $(0,2,-2)$ дают $v=(\log2,\log3,0)$. Query weights $(1,0,2)$ дают dot score $\log2$. Extra expansion coordinate может добавить полезный match или noise; unseen synonym semantics зависит от training, а не от algebra pooling alone.

<!-- /unit:049bbef8fc195b1854 -->

<!-- unit:2efe9a5a41bc138ee1 -->
<a id="u-2efe9a5a41bc138ee1"></a>

L1 и FLOPS-style penalties регулируют sparsity/частоту postings. Слишком много common expansion terms делает inverted lists длинными и ухудшает latency. Низкое число nonzeros не гарантирует лучший index, если все documents используют одни и те же frequent coordinates.

<!-- /unit:2efe9a5a41bc138ee1 -->

<!-- unit:97fdbab0d8d8d57a1e -->
<a id="u-97fdbab0d8d8d57a1e"></a>

Hybrid combine BM25/dense/sparse scores либо rankings. Raw scales могут различаться; weighted sums требуют normalization/tuning. Reciprocal Rank Fusion использует ranks:

<!-- /unit:97fdbab0d8d8d57a1e -->

<!-- unit:13c32e69a713991227 -->
<a id="u-13c32e69a713991227"></a>

$$
\operatorname{RRF}(d)=\sum_{m}\frac1{k_0+r_m(d)}
$$

<!-- /unit:13c32e69a713991227 -->

<!-- unit:ad511f342bdb81ec34 -->
<a id="u-ad511f342bdb81ec34"></a>

по объявленным retrieved lists и missing-document convention. Это ranking heuristic; $k_0$ и cutoffs выбираются без test tuning. Нельзя объединить разные corpora/filters и приписать gain только fusion formula.

<!-- /unit:ad511f342bdb81ec34 -->

<!-- unit:7ef824c63b7187682a -->
<a id="u-7ef824c63b7187682a"></a>

### От MLM logits к разреженному словарному представлению

<!-- /unit:7ef824c63b7187682a -->

<!-- unit:5f61ae834e6ca60318 -->
<a id="u-5f61ae834e6ca60318"></a>

Для каждой входной позиции MLM head может выдать V logits, по одному на vocabulary coordinate. Нелинейность log(1+ReLU(ℓ)) зануляет отрицательные значения и сдерживает рост положительных. После pooling по позициям получается один vocabulary-sized vector. Его координаты не обязаны соответствовать буквально присутствовавшим terms: ranking training может обучить expansion на другие словарные tokens.

<!-- /unit:5f61ae834e6ca60318 -->

<!-- unit:81588a20efffb146e7 -->
<a id="u-81588a20efffb146e7"></a>

Это не vocabulary softmax: веса не обязаны суммироваться в 1 и не интерпретируются как распределение одного следующего слова. При запросе важны совпадения активных координат query/document, а inverted index хранит postings для них. Learned expansion может связать «автомобиль» и «машина», но неверное expansion тоже возможно; формула pooling не создаёт правильные synonyms без обучения.

<!-- /unit:81588a20efffb146e7 -->

<!-- unit:bd9dae5ce0bdc74674 -->
<a id="u-bd9dae5ce0bdc74674"></a>

Уточнение variant: исходный SPLADE использовал sum aggregation; max pooling относится к SPLADE v2 и последующим вариантам. В текущем основном примере главы выбран max вариант. Различия описаны в [SPLADE v2](https://arxiv.org/abs/2109.10086) и [коде pooling](https://github.com/naver/splade/blob/main/splade/models/transformer_rep.py). Нельзя считать sum и max взаимозаменяемыми при том же checkpoint.

<!-- /unit:bd9dae5ce0bdc74674 -->

<!-- unit:367096775fc9d5e993 -->
<a id="u-367096775fc9d5e993"></a>

Для logits двух позиций (1,−1,0) и (3,2,−2) после log/ReLU rows равны (log2,0,0) и (log4,log3,0). Max pooling даёт (log4,log3,0); sum pooling — (log8,log3,0). Первую координату усилило повторное evidence только при sum. В прежнем примере главы каждая координата положительна лишь в одной позиции, поэтому оба pooling случайно совпадали.

<!-- /unit:367096775fc9d5e993 -->

<!-- unit:ce7e6a343803071d52 -->
<a id="u-ce7e6a343803071d52"></a>

### Backward и цена postings

<!-- /unit:ce7e6a343803071d52 -->

<!-- unit:a85eefb9bf68895360 -->
<a id="u-a85eefb9bf68895360"></a>

При ℓ>0 derivative log(1+ℓ) равна 1/(1+ℓ); при ℓ<0 ReLU закрывает этот путь. Для max по позициям gradient координаты идёт через выбранную максимальную позицию, а для sum — через все её contributing positions. В ties и в точке ReLU=0 используются соглашения конкретной реализации. Эти свойства меняют способ обучения, а не только представление output.

<!-- /unit:a85eefb9bf68895360 -->

<!-- unit:69b7844d4e53a2d825 -->
<a id="u-69b7844d4e53a2d825"></a>

Ranking loss дополняют sparsity penalties. Один FLOPS-style surrogate можно записать как сумму квадратов средних activation каждой координаты по batch: $\sum_j(\frac1B\sum_bv_{b,j})^2$. Он не равен измеренному числу процессорных инструкций, а ограничивает частые активные terms, влияющие на postings. Его коэффициент меняет quality/efficiency trade-off.

<!-- /unit:69b7844d4e53a2d825 -->

<!-- unit:18ff39233574e3780c -->
<a id="u-18ff39233574e3780c"></a>

Например, два documents с weights (1,0),(1,0) имеют всего два nonzeros; penalty равна 1. Для (1,0),(0,1) nonzeros тоже два, penalty равна 0.5. Различие связано с концентрацией активности в одной общей координате. Один только средний nnz на документ не описывает длины postings и распределение query terms.

<!-- /unit:18ff39233574e3780c -->

<!-- unit:0fb59729b64382b872 -->
<a id="u-0fb59729b64382b872"></a>

### Hybrid fusion: почему rank и raw score — разные данные

<!-- /unit:0fb59729b64382b872 -->

<!-- unit:90555f1fa397911336 -->
<a id="u-90555f1fa397911336"></a>

BM25 score около 11.7 и cosine около 0.8 нельзя складывать как две одинаковые вероятности. Их шкалы сформированы разными механизмами. Для weighted score sum нужна выбранная normalization и validation; для RRF используются ranks, что устраняет прямую зависимость от численного масштаба scores, но теряет часть информации об их расстояниях.

<!-- /unit:90555f1fa397911336 -->

<!-- unit:7f1c6ace9e15ba9ade -->
<a id="u-7f1c6ace9e15ba9ade"></a>

При k0=60 документ A с ranks 1 и 100 получает $1/61+1/160\approx0.022643$. Документ B с ranks 10 и 10 получает $2/70\approx0.028571$ и выигрывает за согласованность двух списков. Это поведение самой формулы, а не доказательство большей семантической релевантности B. Cutoffs, missing-document convention и tie handling сохраняются вместе с протоколом.

<!-- /unit:7f1c6ace9e15ba9ade -->

<!-- unit:2f8f4ae1d3bb8b9555 -->
<a id="u-2f8f4ae1d3bb8b9555"></a>

## Late interaction и MaxSim

<!-- /unit:2f8f4ae1d3bb8b9555 -->

<!-- unit:e67076bf07a7fa00fc -->
<a id="u-e67076bf07a7fa00fc"></a>

ColBERT хранит document token vectors $D\in\mathbb R^{L_d\times d_e}$; query vectors $Q\in\mathbb R^{L_q\times d_e}$ взаимодействуют позже:

<!-- /unit:e67076bf07a7fa00fc -->

<!-- unit:03fd0f1727b4faafdb -->
<a id="u-03fd0f1727b4faafdb"></a>

$$
s(q,d)=\sum_{i=1}^{L_q}\max_{j\in\mathcal V_d}Q_i^\top D_j.
$$

<!-- /unit:03fd0f1727b4faafdb -->

<!-- unit:32dfc90b3c7fd16e14 -->
<a id="u-32dfc90b3c7fd16e14"></a>

Valid tokens, projection, normalization, query augmentation/padding и punctuation policy задаёт implementation. Document encoding reusable, в отличие от full cross-encoder; matching детальнее single-vector pooling. [ColBERTv2](https://arxiv.org/abs/2112.01488) также исследует compression и denoised supervision.

<!-- /unit:32dfc90b3c7fd16e14 -->

<!-- unit:94c2ab24f599a90848 -->
<a id="u-94c2ab24f599a90848"></a>

Пусть query vectors $(1,0),(0,1)$. Document A содержит $(1,0),(0,1),(-1,0)$; B — $(0,1),(0,1),(0,-1)$. Все vectors unit norm. Mean обоих documents одинаков $(0,1/3)$, так что single mean representation их не различит. MaxSim A=$1+1=2$, B=$0+1=1$: отдельный match первой query component сохранился только у A.

<!-- /unit:94c2ab24f599a90848 -->

<!-- unit:44b07f4febe8e93de4 -->
<a id="u-44b07f4febe8e93de4"></a>

Это иллюстрирует information trade-off, а не доказывает universal superiority MaxSim. Max independently для каждого query token может сопоставить разные, семантически несовместимые части document. Training/negative sampling и aggregation остаются существенными.

<!-- /unit:44b07f4febe8e93de4 -->

<!-- unit:0bc9b5c8a41046ae1d -->
<a id="u-0bc9b5c8a41046ae1d"></a>

![Каждая query-строка выбирает лучший document token; выбранные значения суммируются.](../../archive/2026-10-20-before-rebuild/оформление/transformer/colbert-maxsim.svg)

<!-- /unit:0bc9b5c8a41046ae1d -->

<!-- unit:d425415a601b98f1c0 -->
<a id="u-d425415a601b98f1c0"></a>

Каждая query-строка выбирает лучший document token; выбранные значения суммируются.

<!-- /unit:d425415a601b98f1c0 -->

<!-- unit:f38e17fc3f37f20c18 -->
<a id="u-f38e17fc3f37f20c18"></a>

### Что сохраняется после независимого encoding

<!-- /unit:f38e17fc3f37f20c18 -->

<!-- unit:973937901130154e42 -->
<a id="u-973937901130154e42"></a>

Single-vector encoder сворачивает document в одну точку. ColBERT сохраняет несколько contextual token vectors и сравнивает каждый query vector с набором document vectors позже. Это позволяет оставить более детальное evidence до matching, сохраняя возможность заранее кодировать документы. Общий механизм — [contextualized late interaction](https://arxiv.org/abs/2004.12832).

<!-- /unit:973937901130154e42 -->

<!-- unit:246fb605145dc81bca -->
<a id="u-246fb605145dc81bca"></a>

На исходном примере A/B в главе матрица dot products для A имеет строки (1,0,−1) и (0,1,0). Row maxima равны 1 и 1, сумма 2. Для B строки (0,0,0) и (1,1,−1), maxima 0 и 1, сумма 1. Mean vectors A и B совпадают: положительный и отрицательный first-axis content у A взаимно сократились, но late matching сохранил отдельный полезный token.

<!-- /unit:246fb605145dc81bca -->

<!-- unit:026470dff2aac2be57 -->
<a id="u-026470dff2aac2be57"></a>

MaxSim не требует взаимно однозначного matching. Два query tokens могут выбрать один и тот же document token. Для двух одинаковых queries (1,0) и единственного document vector (1,0) score равна 2. Это не Hungarian assignment и не global alignment последовательностей. Контекстность vectors и ranking training важны, потому что независимые локальные matches могут не гарантировать правильную общую связь слов в тексте.

<!-- /unit:026470dff2aac2be57 -->

<!-- unit:5fc3bc4a86a488071f -->
<a id="u-5fc3bc4a86a488071f"></a>

### Как gradient проходит через выбранные matches

<!-- /unit:5fc3bc4a86a488071f -->

<!-- unit:e4703c3b7852d455eb -->
<a id="u-e4703c3b7852d455eb"></a>

Если у query i есть единственный максимальный source $j_i$, то derivative score по $Q_i$ равна $D_{j_i}$. По document vector $D_j$ суммируются Q_i тех queries, которые выбрали j. Затем derivative ranking loss умножает эти вклады, а projection/normalization и encoder получают свой chain rule. При ties max недифференцируем в обычном смысле, и implementation выбирает допустимый subgradient.

<!-- /unit:e4703c3b7852d455eb -->

<!-- unit:32430a18ae058ffff8 -->
<a id="u-32430a18ae058ffff8"></a>

В примере A первый query выбирает первый document token, второй — второй. Производная score по Q равна [[1,0],[0,1]], по D — [[1,0],[0,1],[0,0]] до внешнего ranking gradient. Третий token не выиграл ни один max и не получает прямого вклада этой MaxSim-ветви. Но его раннее состояние могло участвовать в self-attention других document tokens, поэтому отсутствие такого прямого вклада не отсоединяет весь токен от обучения encoder.

<!-- /unit:32430a18ae058ffff8 -->

<!-- unit:0bf4ae70cf33fa41c2 -->
<a id="u-0bf4ae70cf33fa41c2"></a>

Добавление document token не может уменьшить exact MaxSim при неизменных остальных Q/D и допустимых masks: максимум по расширенному множеству не уменьшается. Это математическое свойство matching, не гарантия, что более длинные документы всегда лучше. После нового encoding другие contextual vectors могут измениться; training, masks и длины также задают реальные условия поиска.

<!-- /unit:0bf4ae70cf33fa41c2 -->

<!-- unit:3485c1b7a4ec2fc5e6 -->
<a id="u-3485c1b7a4ec2fc5e6"></a>

## Storage и полный budget

<!-- /unit:3485c1b7a4ec2fc5e6 -->

<!-- unit:a97a3be88362744b57 -->
<a id="u-a97a3be88362744b57"></a>

Raw single-vector storage $N d_e b$ bytes. Multi-vector — примерно $N\overline L_d d_e b$ до compression/metadata/index. При $N=100\,000,d_e=128,b=2$ single vectors занимают 25.6 MB decimal; при $\overline L_d=128$ — 3.2768 GB, в 128 раз больше raw vectors. Реальные системы compress, prune и используют сложные candidate indexes; только этот multiplication не описывает весь service.

<!-- /unit:a97a3be88362744b57 -->

<!-- unit:9c8fd68cd3937d5b3c -->
<a id="u-9c8fd68cd3937d5b3c"></a>

[BGE-M3](https://arxiv.org/abs/2402.03216) — пример dense/sparse/multi-vector modes в одном опубликованном семействе. Их coexistence не доказывает равные quality/cost на вашем corpus. Compare one corpus, qrels, filters, query groups и total offline/online budget.

<!-- /unit:9c8fd68cd3937d5b3c -->

<!-- unit:47ec071aaf9a77b0ed -->
<a id="u-47ec071aaf9a77b0ed"></a>

First-stage retrieval, reranker и generator оцениваются раздельно. Стоимость включает offline encoding/reindex, index footprint, query latency и tail latency pairs. New embedding revision обычно требует coordinated re-encoding; vectors разных spaces нельзя считать совместимыми лишь из-за одинаковой dimension.

<!-- /unit:47ec071aaf9a77b0ed -->

<!-- unit:060e4d3ff32872b6bb -->
<a id="u-060e4d3ff32872b6bb"></a>

### Два разных смысла multi-vector из разбираемого примера

<!-- /unit:060e4d3ff32872b6bb -->

<!-- unit:542a9c1b0186f97766 -->
<a id="u-542a9c1b0186f97766"></a>

В S6 вы спрашивали, насколько разрастается БД, если хранить text/summary/keywords/table vectors. Это несколько **представлений документа**. У ColBERT обычно сохраняется множество **token representations**. Четыре views на документ и 128 token vectors имеют разные свойства matching и storage; одно название multi-vector не делает их одной архитектурой.

<!-- /unit:542a9c1b0186f97766 -->

<!-- unit:77d66f0962cbd460fe -->
<a id="u-77d66f0962cbd460fe"></a>

Числа S6 для 10 000 документов ширины 384 в float32: один vector на документ — 15 360 000 bytes, то есть 15.36 MB decimal; четыре views — 61.44 MB. Для 100 000 документов — 153.6 и 614.4 MB. Эти отношения описывают только raw vector payload. Если текст/metadata хранятся один раз и views ссылаются на общий parent ID, суммарная БД растёт меньше чем в четыре раза по этой части; конкретный индекс добавляет свои затраты.

<!-- /unit:77d66f0962cbd460fe -->

<!-- unit:abbec63d1fd548fddd -->
<a id="u-abbec63d1fd548fddd"></a>

Приведённые в разбираемом примере HNSW размеры, latency×1.5–2, пороги 50k/500k документов и приросты recall не обоснованы конфигурацией или измерениями. Их нельзя переносить как универсальные benchmarks. Четыре encoding views могут обрабатываться batch; число API requests не обязано совпадать с числом текстов, а цена определяется фактическими единицами сервиса. Это отдельно от числа stored vectors.

<!-- /unit:abbec63d1fd548fddd -->

<!-- unit:74cc700359607e7665 -->
<a id="u-74cc700359607e7665"></a>

Один поиск query vector по объединённым views корректен лишь при совместимом пространстве и понятной aggregation. Если text/table vectors созданы несогласованными encoders, одинаковая ширина и metadata filter не исправят matching. Для нескольких query views нужно отдельно определить поиск и объединение. «Вместо четырёх запросов один» может изменить математическую задачу, а не только ускорить её.

<!-- /unit:74cc700359607e7665 -->

<!-- unit:cf034d3b6c9e2d2419 -->
<a id="u-cf034d3b6c9e2d2419"></a>

### Parent retrieval и честный полный budget

<!-- /unit:cf034d3b6c9e2d2419 -->

<!-- unit:96fd01fa8e29eea453 -->
<a id="u-96fd01fa8e29eea453"></a>

Parent document retrieval может индексировать только маленькие child chunks, а большие parent texts хранить для выдачи context. Индексировать оба размера vectors обязательно не требуется; поэтому удвоение storage не является свойством самой идеи. Выбор parent увеличивает переданный context и может добавить нерелевантные участки, что тоже оценивается.

<!-- /unit:96fd01fa8e29eea453 -->

<!-- unit:dd288def875a38019d -->
<a id="u-dd288def875a38019d"></a>

Для сравнения фиксируют корпус, qrels, фильтры, query groups и требуемый output. Затем измеряют first-stage recall, final ranking, offline encoding/reindex, raw vectors и index size, query/reranking latency и tail latency. Сначала single/dense или lexical baseline, затем контролируемое добавление views, hybrid и reranking. Само число документов не отвечает на вопрос «стоит ли оно того»: нужны измеренный gain и цена нужного режима на ваших данных.

<!-- /unit:dd288def875a38019d -->

<!-- unit:a63c6e6160689afbbf -->
<a id="u-a63c6e6160689afbbf"></a>

## Контроль и практика

<!-- /unit:a63c6e6160689afbbf -->

<!-- unit:2a39834b9115bf7a0b -->
<a id="u-2a39834b9115bf7a0b"></a>

1. Чем ANN recall отличается от relevance recall? **Ответ:** первое — approximation exact vector ranking, второе — найденное semantic evidence по qrels.
2. Может ли reranker исправить отсутствующего candidate? **Ответ:** нет; его choices ограничены first-stage pool.
3. Почему SPLADE не просто arbitrary sparse dense embedding? **Ответ:** coordinates связаны с vocabulary и learned lexical expansion/indexing.
4. Что потеряно при mean в toy MaxSim example? **Ответ:** отдельный exact match query direction, компенсированный другими token vectors.

<!-- /unit:2a39834b9115bf7a0b -->

<!-- unit:0de023d50a207e6056 -->
<a id="u-0de023d50a207e6056"></a>

5. Откуда берутся keywords базового BM25? **Ответ:** из terms обработанного query, сопоставляемых с индексом; отдельный обученный keyword extractor не требуется.
6. Равны ли четыре document views и token vectors ColBERT? **Ответ:** нет. Это разные способы представления и matching, с разным числом vectors и storage budget.
7. Нужно ли parent retrieval индексировать и малые, и большие chunks? **Ответ:** нет. Можно индексировать child vectors, а parent texts хранить для выдачи контекста.
8. Почему max SPLADE pooling нельзя молча заменить sum? **Ответ:** меняются values, длина/повторяемость evidence и путь gradient; checkpoint обучался с определённой агрегацией.

<!-- /unit:0de023d50a207e6056 -->

<!-- unit:a36a2928378000ac60 -->
<a id="u-a36a2928378000ac60"></a>

**Практика:** вычислите pairwise loss, sparse weights и MaxSim вручную. Для одного fixed query set спроектируйте BM25→dense/hybrid→reranker ablation и optional late interaction. Запишите quality, candidate ceiling, storage и latency отдельно; числа своих models добавляйте после запуска.

<!-- /unit:a36a2928378000ac60 -->

## Проектная практика ветки

<!-- unit:a5d70d09aa7c4a04ca -->
<a id="u-a5d70d09aa7c4a04ca"></a>

### P3. Поиск по документам — основная retrieval-практика

<!-- /unit:a5d70d09aa7c4a04ca -->

<!-- unit:0b5da4de57b0b3fbd4 -->
<a id="u-0b5da4de57b0b3fbd4"></a>

Corpus + held-out queries/qrels → BM25 → dense retrieval → reranker. Изменять один этап за раз; выбирать pooling/prefixes по checkpoint. Итог: versioned index, retrieval metrics, latency/memory и список queries, где методы расходятся.

<!-- /unit:0b5da4de57b0b3fbd4 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как выбрать место взаимодействия query и document и оценить полный budget?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
