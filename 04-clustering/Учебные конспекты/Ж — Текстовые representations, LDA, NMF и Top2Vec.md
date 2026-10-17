# Ж. Темы, смысл и lexical similarity документов

[Оглавление](README.md). Пункт 7 плана. **Основа:** S6:182–1373 различает retrieval embeddings и внутренние LLM states; topic models здесь — дополнение.

## 1. Нет гарантированной лестницы TF-IDF → BERT

TF-IDF представляет документ по весам слов/термов. Одна smoothed convention:

$$
\operatorname{idf}(w)=1+\log\frac{N+1}{\operatorname{df}(w)+1},
\qquad
x_{dw}=\operatorname{tf}(w,d)\operatorname{idf}(w).
$$

Затем vectors часто L2-normalize и сравнивают cosine. Tokenization, stopwords, n-grams и frequency cutoffs определяют, что значит lexical similarity. Для technical topics точные термины могут быть сильнее общей semantic близости. Словарь/idf fit по допустимому corpus; prospective evaluation сохраняет его frozen.

Word2Vec даёт word vectors, которые можно агрегировать, но average теряет порядок и часто размывает rare relevant words. Doc2Vec обучает document representations своим objective. Обычный BERT hidden state не обязан быть хорошим sentence-level vector; SBERT-like training целенаправленно формирует geometry сходства. Название более новой модели не доказывает лучший clustering на вашей задаче.

Два коротких сообщения «счёт не оплачен» и «счёт оплачен» lexical близки, но decision semantics различается. Два разных technical термина могут обозначать одно понятие, которое bag-of-words не свяжет. Representation выбирают по тому, какие distinctions важны target grouping.

## 2. LDA: документ — смесь, не один cluster

[Latent Dirichlet Allocation](https://www.jmlr.org/papers/v3/blei03a.html) — generative topic model. Для каждого topic задаётся distribution по words $\phi_k$, для document — mixture $\theta_d$. Word position выбирает topic и затем word:

$$
\theta_d\sim\operatorname{Dirichlet}(\alpha),
\quad z_{dn}\sim\operatorname{Categorical}(\theta_d),
\quad w_{dn}\sim\operatorname{Categorical}(\phi_{z_{dn}}).
$$

У topics тоже может быть Dirichlet prior. После marginalization topic word probability равна $P(w\mid d)=\sum_k\theta_{dk}\phi_{kw}$. Document одновременно содержит несколько topics; превращение его в hard cluster через argmax — дополнительное упрощение, а не исходная модель.

Пример: document mixture $(0.7,0.3)$ для topics ML/финансы, слово «модель» имеет probabilities 0.1/0.01. Итоговая вероятность $0.7\cdot0.1+0.3\cdot0.01=0.073$. Это не вероятность, что весь document относится к одному классу. Bag-of-words assumptions не сохраняют word order; короткий текст даёт мало co-occurrence statistics для оценки mixture.

## 3. NMF как additive representation

Для nonnegative document-term matrix $X\in\mathbb R_+^{n\times V}$:

$$
X\approx WH,\qquad
W\in\mathbb R_+^{n\times K},\quad H\in\mathbb R_+^{K\times V},
\qquad
\min_{W,H\ge0}\frac12\|X-WH\|_F^2.
$$

Row H описывает topic words, row W — topic weights document. Возможны другие divergences/penalties. Nonnegativity делает decomposition additive, но сама по себе не даёт probability normalization. Если одну column W умножить на $a>0$, соответствующую row H разделить на $a$, product не изменится.

Для $W=(2,1)$ и topic-word rows $(1,0),(0,3)$ реконструкция document vector равна $(2,3)$. Чтобы читать W как mixture, требуется дополнительная normalization и понимание scales H. NMF иногда даёт удобные topics, но её interpretation/скорость не universally лучше LDA: всё зависит от matrix, loss, solver и requested quality.

## 4. Top2Vec и гибриды

[Top2Vec](https://arxiv.org/abs/2008.09470) использует semantic document/word space; reduced representation помогает density clustering, затем topic vectors и близкие words интерпретируются в semantic space. UMAP/HDBSCAN — stages pipeline, а не доказательство истинного числа тем. Версия и выбранный embedding backend определяют конкретное поведение.

Гибрид lexical topic features и BERT embeddings может соединять точные термины с semantics. Но concatenation без scaling даёт dominance одной части; autoencoder может сохранить nuisance factors. Нужны ablations каждого source representation и independent human review. Для коротких texts aggregation по author/time может добавить co-occurrences, но меняет единицу анализа и group split.

## 5. Оценка темы

Top words помогают назвать topic, но могут выглядеть coherent при плохой группировке documents. Смотрите representative и borderline texts, coherence/diversity при явно заданной definition, stability и downstream utility. Topic coherence не тождественна held-out likelihood, а ни одна из них не гарантирует usefulness человеку.

Количество topics/K выбирают на development criterion. Если hidden human classes доступны, не используйте final test labels для tuning topic count. Отдельно учитывайте language/source/style: clustering может разделить языки вместо тем. Multilingual encoder не гарантирует, что language factor исчез полностью.

## 6. Контроль и практика

1. Является ли LDA hard document clustering? **Ответ:** исходно нет; document имеет mixture topics, hard assignment добавляется отдельно.
2. Почему weights NMF не автоматически probabilities? **Ответ:** nonnegativity не задаёт sum-to-one и decomposition имеет scale ambiguity.
3. Всегда ли SBERT лучше TF-IDF для topics? **Ответ:** нет; lexical distinctions и domain могут делать TF-IDF сильным baseline.

**Практика:** соберите 20 коротких документов двух языков и трёх тем. Сравните TF-IDF и frozen sentence embeddings при одном algorithm. Для LDA/NMF вручную интерпретируйте mixture/product примеры, затем составьте rubric document coherence без выбора по красивым top words alone.
