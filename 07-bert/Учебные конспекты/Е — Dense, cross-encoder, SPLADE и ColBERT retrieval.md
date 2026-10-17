# Е. Четыре способа оценить релевантность query–document

[Оглавление](README.md). Группа F, модули **26–29**. **Основа:** S6:182–1373 подробно обсуждает RAG/BM25/multi-vector storage; здесь добавлены encoder training, sparse expansion и точный MaxSim. Общий RAG pipeline находится в [приложениях LLM](../../06-llm/04-applications/Учебные%20конспекты/README.md).

## 1. Dense retrieval: reusable document vectors

Query encoder и document encoder обучаются так, чтобы relevant pair имела высокий score. Shared weights возможны, но tasks asymmetric: короткий вопрос и длинный passage имеют разные distributions. Prefixes `query:`/`passage:` или инструкции берутся из конкретной model card/revision; нельзя считать их universal syntax всех encoders. [Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-base) — пример task-trained retrieval checkpoint.

Offline pipeline: source documents → permitted chunks/metadata → encoder → vectors → index. Online: query preprocessing → vector → filtered search → document aggregation. IDs, namespaces, versions и access filters должны сопровождать vectors, а не восстанавливаться по порядку строк случайно.

Exact vector search сравнивает со всеми candidates выбранного corpus; ANN приближает этот поиск. ANN recall@k — overlap с exact vector top-k, semantic relevance recall@k — найденные judged relevant docs относительно qrels. ANN может идеально воспроизвести semantic плохой vector ranking. Поэтому обе проверки нужны отдельно.

Chunk→document aggregation тоже меняет ranking: max score предпочитает один сильный passage, mean — average evidence и penalizes long docs с множеством irrelevant chunks. Count multiple hits без контроля duplicates может завышать длинный документ. Cutoffs и aggregation fit на validation.

## 2. Cross-encoder reranking

Cross-encoder кодирует query+candidate jointly и возвращает relevance score. Внутреннее attention даёт token interactions до head, а не только сравнение готовых vectors. Такой model обычно применяется к ограниченному candidate set: он не вернёт missing document из-за первого этапа.

Pointwise BCE обучает relevance каждого pair. Pairwise logistic loss усиливает positive относительно negative:

$$
L_{\mathrm{pair}}=\log\left(1+e^{-(s(q,d^+)-s(q,d^-))}\right).
$$

При scores 2 и 1 margin 1, loss около 0.313262. Если negative на деле тоже relevant, loss создаёт ложное ordering pressure. Listwise objective сравнивает весь list одного query, например CE по normalized target grades/teacher distribution; grading convention и incomplete judgements важны. [CrossEncoder training](https://sbert.net/docs/cross_encoder/training_overview.html).

Score может быть ranking logit, а не calibrated relevance probability across queries. Если threshold должен означать «ответ отсутствует», нужна отдельная calibration на representative queries, включая no-answer. Кандидаты k=100 дают примерно в пять раз больше pair encodings, чем k=20, при одинаковой длине; actual latency зависит от batching/padding/queue.

Evaluation фиксирует first-stage candidates и проверяет incremental ranking gain, candidate recall ceiling, total latency и truncation. Split по queries/documents/groups, а не случайным pair rows одного query.

## 3. Learned sparse: vocabulary weights

SPLADE-like representations используют vocabulary-space signals encoder и нелинейное pooling. Один распространённый вариант:

$$
v_j=\max_t\log(1+\operatorname{ReLU}(\ell_{t,j})),
\qquad
s(q,d)=\sum_j v_j^{(q)}v_j^{(d)}.
$$

Token logits превращаются в nonnegative term weights, включая learned expansion terms. Это не merely dense vector, в котором случайно много zeros: vocabulary coordinates имеют lexical interpretation и используются inverted index. Variant pooling/query architecture/regularization различаются. [SPLADE](https://arxiv.org/abs/2107.05720).

Toy two-position logits rows $(1,-1,0)$ и $(0,2,-2)$ дают $v=(\log2,\log3,0)$. Query weights $(1,0,2)$ дают dot score $\log2$. Extra expansion coordinate может добавить полезный match или noise; unseen synonym semantics зависит от training, а не от algebra pooling alone.

L1 и FLOPS-style penalties регулируют sparsity/частоту postings. Слишком много common expansion terms делает inverted lists длинными и ухудшает latency. Низкое число nonzeros не гарантирует лучший index, если все documents используют одни и те же frequent coordinates.

Hybrid combine BM25/dense/sparse scores либо rankings. Raw scales могут различаться; weighted sums требуют normalization/tuning. Reciprocal Rank Fusion использует ranks:

$$
\operatorname{RRF}(d)=\sum_{m}\frac1{k_0+r_m(d)}
$$

по объявленным retrieved lists и missing-document convention. Это ranking heuristic; $k_0$ и cutoffs выбираются без test tuning. Нельзя объединить разные corpora/filters и приписать gain только fusion formula.

## 4. Late interaction и MaxSim

ColBERT хранит document token vectors $D\in\mathbb R^{L_d\times d_e}$; query vectors $Q\in\mathbb R^{L_q\times d_e}$ взаимодействуют позже:

$$
s(q,d)=\sum_{i=1}^{L_q}\max_{j\in\mathcal V_d}Q_i^\top D_j.
$$

Valid tokens, projection, normalization, query augmentation/padding и punctuation policy задаёт implementation. Document encoding reusable, в отличие от full cross-encoder; matching детальнее single-vector pooling. [ColBERTv2](https://arxiv.org/abs/2112.01488) также исследует compression и denoised supervision.

Пусть query vectors $(1,0),(0,1)$. Document A содержит $(1,0),(0,1),(-1,0)$; B — $(0,1),(0,1),(0,-1)$. Все vectors unit norm. Mean обоих documents одинаков $(0,1/3)$, так что single mean representation их не различит. MaxSim A=$1+1=2$, B=$0+1=1$: отдельный match первой query component сохранился только у A.

Это иллюстрирует information trade-off, а не доказывает universal superiority MaxSim. Max independently для каждого query token может сопоставить разные, семантически несовместимые части document. Training/negative sampling и aggregation остаются существенными.

## 5. Storage и полный budget

Raw single-vector storage $N d_e b$ bytes. Multi-vector — примерно $N\overline L_d d_e b$ до compression/metadata/index. При $N=100\,000,d_e=128,b=2$ single vectors занимают 25.6 MB decimal; при $\overline L_d=128$ — 3.2768 GB, в 128 раз больше raw vectors. Реальные системы compress, prune и используют сложные candidate indexes; только этот multiplication не описывает весь service.

[BGE-M3](https://arxiv.org/abs/2402.03216) — пример dense/sparse/multi-vector modes в одном опубликованном семействе. Их coexistence не доказывает равные quality/cost на вашем corpus. Compare one corpus, qrels, filters, query groups и total offline/online budget.

First-stage retrieval, reranker и generator оцениваются раздельно. Стоимость включает offline encoding/reindex, index footprint, query latency и tail latency pairs. New embedding revision обычно требует coordinated re-encoding; vectors разных spaces нельзя считать совместимыми лишь из-за одинаковой dimension.

## 6. Контроль и практика модулей 26–29

1. Чем ANN recall отличается от relevance recall? **Ответ:** первое — approximation exact vector ranking, второе — найденное semantic evidence по qrels.
2. Может ли reranker исправить отсутствующего candidate? **Ответ:** нет; его choices ограничены first-stage pool.
3. Почему SPLADE не просто arbitrary sparse dense embedding? **Ответ:** coordinates связаны с vocabulary и learned lexical expansion/indexing.
4. Что потеряно при mean в toy MaxSim example? **Ответ:** отдельный exact match query direction, компенсированный другими token vectors.

**Практика:** вычислите pairwise loss, sparse weights и MaxSim вручную. Для одного fixed query set спроектируйте BM25→dense/hybrid→reranker ablation и optional late interaction. Запишите quality, candidate ceiling, storage и latency отдельно; числа своих models добавляйте после запуска.
