# Текстовые представления, LDA, NMF и Top2Vec

**Уровень:** основа ветки.

**Предпосылки:** [Кластеризация замороженных представлений](<cl-frozen.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как различать тему документа, embedding и кластер?





## Карта вычисления и границы этапов

![Текстовые представления, LDA, NMF и Top2Vec: входы, преобразования и выходы](<../оформление/course/cl-text.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:0fcd7808b6581d4d3b -->
<a id="u-0fcd7808b6581d4d3b"></a>

## Нет гарантированной лестницы TF-IDF → BERT

<!-- /unit:0fcd7808b6581d4d3b -->

<!-- unit:4f88ff5fb4b431efdd -->
<a id="u-4f88ff5fb4b431efdd"></a>

TF-IDF представляет документ по весам слов/термов. Одна smoothed convention:

<!-- /unit:4f88ff5fb4b431efdd -->

<!-- unit:c6bcc49fd37a269e8f -->
<a id="u-c6bcc49fd37a269e8f"></a>

$$
\operatorname{idf}(w)=1+\log\frac{N+1}{\operatorname{df}(w)+1},
\qquad
x_{dw}=\operatorname{tf}(w,d)\operatorname{idf}(w).
$$

<!-- /unit:c6bcc49fd37a269e8f -->

<!-- unit:e41459e8cc72068b44 -->
<a id="u-e41459e8cc72068b44"></a>

Затем vectors часто L2-normalize и сравнивают cosine. Tokenization, stopwords, n-grams и frequency cutoffs определяют, что значит lexical similarity. Для technical topics точные термины могут быть сильнее общей semantic близости. Словарь/idf fit по допустимому corpus; prospective evaluation сохраняет его frozen.

<!-- /unit:e41459e8cc72068b44 -->

<!-- unit:3742ce0e6c1efe7534 -->
<a id="u-3742ce0e6c1efe7534"></a>

Word2Vec даёт word vectors, которые можно агрегировать, но average теряет порядок и часто размывает rare relevant words. Doc2Vec обучает document representations своим objective. Обычный BERT hidden state не обязан быть хорошим sentence-level vector; SBERT-like training целенаправленно формирует geometry сходства. Название более новой модели не доказывает лучший clustering на вашей задаче.

<!-- /unit:3742ce0e6c1efe7534 -->

<!-- unit:dc4c98f088754ec08a -->
<a id="u-dc4c98f088754ec08a"></a>

Два коротких сообщения «счёт не оплачен» и «счёт оплачен» lexical близки, но decision semantics различается. Два разных technical термина могут обозначать одно понятие, которое bag-of-words не свяжет. Representation выбирают по тому, какие distinctions важны target grouping.

<!-- /unit:dc4c98f088754ec08a -->

<!-- unit:29fee8dd6086b53c16 -->
<a id="u-29fee8dd6086b53c16"></a>

## LDA: документ — смесь, не один cluster

<!-- /unit:29fee8dd6086b53c16 -->

<!-- unit:77f14aaf59882f1ed8 -->
<a id="u-77f14aaf59882f1ed8"></a>

[Latent Dirichlet Allocation](https://www.jmlr.org/papers/v3/blei03a.html) — generative topic model. Для каждого topic задаётся distribution по words $\phi_k$, для document — mixture $\theta_d$. Word position выбирает topic и затем word:

<!-- /unit:77f14aaf59882f1ed8 -->

<!-- unit:a4fa5fc7bb62f173ea -->
<a id="u-a4fa5fc7bb62f173ea"></a>

$$
\theta_d\sim\operatorname{Dirichlet}(\alpha),
\quad z_{dn}\sim\operatorname{Categorical}(\theta_d),
\quad w_{dn}\sim\operatorname{Categorical}(\phi_{z_{dn}}).
$$

<!-- /unit:a4fa5fc7bb62f173ea -->

<!-- unit:e39074b4ce140bc7f2 -->
<a id="u-e39074b4ce140bc7f2"></a>

У topics тоже может быть Dirichlet prior. После marginalization topic word probability равна $P(w\mid d)=\sum_k\theta_{dk}\phi_{kw}$. Document одновременно содержит несколько topics; превращение его в hard cluster через argmax — дополнительное упрощение, а не исходная модель.

<!-- /unit:e39074b4ce140bc7f2 -->

<!-- unit:124288efa683184849 -->
<a id="u-124288efa683184849"></a>

Пример: document mixture $(0.7,0.3)$ для topics ML/финансы, слово «модель» имеет probabilities 0.1/0.01. Итоговая вероятность $0.7\cdot0.1+0.3\cdot0.01=0.073$. Это не вероятность, что весь document относится к одному классу. Bag-of-words assumptions не сохраняют word order; короткий текст даёт мало co-occurrence statistics для оценки mixture.

<!-- /unit:124288efa683184849 -->

<!-- unit:d09bdc72ea0fb501d3 -->
<a id="u-d09bdc72ea0fb501d3"></a>

## NMF как additive representation

<!-- /unit:d09bdc72ea0fb501d3 -->

<!-- unit:2724e9910cf3acc45e -->
<a id="u-2724e9910cf3acc45e"></a>

Для nonnegative document-term matrix $X\in\mathbb R_+^{n\times V}$:

<!-- /unit:2724e9910cf3acc45e -->

<!-- unit:4c1ea182d5e7baed64 -->
<a id="u-4c1ea182d5e7baed64"></a>

$$
X\approx WH,\qquad
W\in\mathbb R_+^{n\times K},\quad H\in\mathbb R_+^{K\times V},
\qquad
\min_{W,H\ge0}\frac12\|X-WH\|_F^2.
$$

<!-- /unit:4c1ea182d5e7baed64 -->

<!-- unit:634624b710fac1ccab -->
<a id="u-634624b710fac1ccab"></a>

Row H описывает topic words, row W — topic weights document. Возможны другие divergences/penalties. Nonnegativity делает decomposition additive, но сама по себе не даёт probability normalization. Если одну column W умножить на $a>0$, соответствующую row H разделить на $a$, product не изменится.

<!-- /unit:634624b710fac1ccab -->

<!-- unit:4fb75420caf121dfe9 -->
<a id="u-4fb75420caf121dfe9"></a>

Для $W=(2,1)$ и topic-word rows $(1,0),(0,3)$ реконструкция document vector равна $(2,3)$. Чтобы читать W как mixture, требуется дополнительная normalization и понимание scales H. NMF иногда даёт удобные topics, но её interpretation/скорость не universally лучше LDA: всё зависит от matrix, loss, solver и requested quality.

<!-- /unit:4fb75420caf121dfe9 -->

<!-- unit:7c78f81a8804535cb1 -->
<a id="u-7c78f81a8804535cb1"></a>

## Top2Vec и гибриды

<!-- /unit:7c78f81a8804535cb1 -->

<!-- unit:dc0dfa5886502a6065 -->
<a id="u-dc0dfa5886502a6065"></a>

[Top2Vec](https://arxiv.org/abs/2008.09470) использует semantic document/word space; reduced representation помогает density clustering, затем topic vectors и близкие words интерпретируются в semantic space. UMAP/HDBSCAN — stages pipeline, а не доказательство истинного числа тем. Версия и выбранный embedding backend определяют конкретное поведение.

<!-- /unit:dc0dfa5886502a6065 -->

<!-- unit:9bc77fc4e45068bec7 -->
<a id="u-9bc77fc4e45068bec7"></a>

Гибрид lexical topic features и BERT embeddings может соединять точные термины с semantics. Но concatenation без scaling даёт dominance одной части; autoencoder может сохранить nuisance factors. Нужны ablations каждого source representation и independent human review. Для коротких texts aggregation по author/time может добавить co-occurrences, но меняет единицу анализа и group split.

<!-- /unit:9bc77fc4e45068bec7 -->

<!-- unit:529c34d1abe0563b27 -->
<a id="u-529c34d1abe0563b27"></a>

## Оценка темы

<!-- /unit:529c34d1abe0563b27 -->

<!-- unit:30b113b8ca56f1e88f -->
<a id="u-30b113b8ca56f1e88f"></a>

Top words помогают назвать topic, но могут выглядеть coherent при плохой группировке documents. Смотрите representative и borderline texts, coherence/diversity при явно заданной definition, stability и downstream utility. Topic coherence не тождественна held-out likelihood, а ни одна из них не гарантирует usefulness человеку.

<!-- /unit:30b113b8ca56f1e88f -->

<!-- unit:35d53a3332f20f5b79 -->
<a id="u-35d53a3332f20f5b79"></a>

Количество topics/K выбирают на development criterion. Если hidden human classes доступны, не используйте final test labels для tuning topic count. Отдельно учитывайте language/source/style: clustering может разделить языки вместо тем. Multilingual encoder не гарантирует, что language factor исчез полностью.

<!-- /unit:35d53a3332f20f5b79 -->

<!-- unit:dbce464640d02c42c5 -->
<a id="u-dbce464640d02c42c5"></a>

### Ограничения и failure modes текстовых тем

<!-- /unit:dbce464640d02c42c5 -->

<!-- unit:5f62be18ad06e81ec4 -->
<a id="u-5f62be18ad06e81ec4"></a>

Topic labels зависят от tokenizer, stop-word policy, number of topics, corpus time range и частотных cutoffs. LDA/NMF могут отнести документ сразу к нескольким темам; принудительная argmax меняет задачу на hard assignment. Embedding density methods чувствительны к encoder, reduction и min_cluster_size; нерелевантные chunks, boilerplate и language/source clusters могут доминировать над предметной темой. Проверяйте устойчивость на новых временных/языковых slices и читайте representative и borderline texts. Если назначение темы меняет маршрут пользователя, измеряйте downstream utility и стоимость review, а не только красивую двумерную проекцию.

<!-- /unit:5f62be18ad06e81ec4 -->

<!-- unit:cebec8c47540cbd283 -->
<a id="u-cebec8c47540cbd283"></a>

## Контроль и практика

<!-- /unit:cebec8c47540cbd283 -->

<!-- unit:bff601714d1e630c83 -->
<a id="u-bff601714d1e630c83"></a>

1. Является ли LDA hard document clustering? **Ответ:** исходно нет; document имеет mixture topics, hard assignment добавляется отдельно.
2. Почему weights NMF не автоматически probabilities? **Ответ:** nonnegativity не задаёт sum-to-one и decomposition имеет scale ambiguity.
3. Всегда ли SBERT лучше TF-IDF для topics? **Ответ:** нет; lexical distinctions и domain могут делать TF-IDF сильным baseline.

<!-- /unit:bff601714d1e630c83 -->

<!-- unit:08f16aa299327f662f -->
<a id="u-08f16aa299327f662f"></a>

**Практика:** соберите 20 коротких документов двух языков и трёх тем. Сравните TF-IDF и frozen sentence embeddings при одном algorithm. Для LDA/NMF вручную интерпретируйте mixture/product примеры, затем составьте rubric document coherence без выбора по красивым top words alone.

<!-- /unit:08f16aa299327f662f -->

<!-- unit:9fc19d5371f3130fcf -->
<a id="u-9fc19d5371f3130fcf"></a>

**Схема курса:** documents → tokenizer/vectorizer or embedding → LDA/NMF/topic embedding → topic/cluster representation → inspect terms and representative documents

<!-- /unit:9fc19d5371f3130fcf -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как различать тему документа, embedding и кластер?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
