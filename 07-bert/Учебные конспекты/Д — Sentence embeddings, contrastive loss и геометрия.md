# Д. Как научить vectors выражать нужную похожесть

[Оглавление](README.md). Группа E, модули **22–25**. **Основа:** S6:182–1373 задаёт вопросы internal/retrieval embeddings; sentence representation learning здесь расширено по BERT-плану.

## 1. Независимое encoding и learned geometry

Bi-encoder вычисляет vectors независимо: $z_x=\operatorname{pool}(f_\theta(x))$, $z_y=\operatorname{pool}(f_\theta(y))$, затем score. Shared weights образуют siamese architecture; separate query/document weights задают asymmetric model. Document vectors можно precompute, если representation не зависит от query.

[Sentence-BERT](https://aclanthology.org/D19-1410/) адаптирует encoder под sentence comparisons. Это не просто mean pooling любого MLM backbone: objective влияет на geometry. Pooling/projection/prefix/normalization сохраняются вместе с checkpoint. Cross-encoder обрабатывает pair jointly и может выучить более детальные interactions, но обычно не даёт reusable одного document vector, заменяющего всю pair computation.

Dot product учитывает norm и direction, cosine — только direction. Query $(1,0)$, documents A=(2,0), B=(0.9,0.1): dot A=2,B=0.9, cosine A=1,B около 0.9939. Если у другого document huge norm при менее точном direction, dot может предпочесть его. После L2 normalization cosine/dot совпадают, а squared Euclidean равен $2-2z_x^\top z_y$. До normalization rankings могут различаться.

## 2. Objectives и supervision

Cosine regression подгоняет similarity к заданному continuous label; scale label должен соответствовать score. Triplet loss требует anchor, positive и negative:

$$
L_{\mathrm{triplet}}=\max\{0,m+d(a,p)-d(a,n)\}.
$$

При margin 0.2, positive distance 0.3, negative 0.4 loss равна 0.1. После достаточного separation gradient этой triplet становится нулевым; mining меняет долю информативных cases. Это другой objective, чем CE по in-batch candidates.

Для $N$ query-positive pairs InfoNCE/multiple-negatives form:

$$
S_{ij}=\frac{\operatorname{sim}(q_i,d_j)}\tau,
\qquad
L_i=-\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}},
\qquad L=\frac1N\sum_iL_i.
$$

First row logits $(2,1,0)$ даёт probabilities примерно $(0.665241,0.244728,0.090031)$ и loss $0.407606$. Gradient строки по logits — $(-0.334759,0.244728,0.090031)$; при общем mean по N он делится на N. По unscaled similarity добавляется factor $1/\tau$. Маленькая temperature делает softmax sharper и меняет gradient scale; это не inference sampling temperature causal LM.

Symmetric loss добавляет обратное direction document→query по выбранному protocol. Оно уместно не для любой asymmetric relevance task. Несколько positives требуют mask/multi-positive objective: второй релевантный document не должен автоматически считаться negative.

## 3. Hard negatives, batch и SimCSE

Random negatives часто просты; mined lexical/semantic negatives учат различать близкие irrelevant candidates. Но near-duplicate/alternative relevant passages становятся false negatives, и off-diagonal gradient тогда отталкивает полезный document. Teacher ranking не безошибочен; judgement provenance и review важны.

В [SimCSE](https://arxiv.org/abs/2104.08821) unsupervised positive views получают разные dropout masks одного text; supervised variant использует NLI information. Два deterministic identical passes задают другой training signal. Это не утверждение, что любая dropout perturbation сохраняет всё нужное meaning.

Contrastive negative pool отличается от optimization batch. Accumulation нескольких independent losses не создаёт cross-microbatch terms. Distributed all-gather/cached losses могут расширять pool, но имеют requirements gradient correctness, stale representations и memory. [Loss implementations и data formats](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) сверяются с installed version.

Mining проводится без final held-out labels; cross-validation включает encoder training. Иначе supervision test попадёт в representation до оценки downstream classifier/clustering.

## 4. Distillation и размерность

Для teacher column $t\in\mathbb R^{768}$ и student $s\in\mathbb R^{256}$ coordinate matching требует общего space, например fixed/обоснованного projection $P\in\mathbb R^{256\times768}$:

$$
L_{\mathrm{coord}}=\|s-Pt\|_2^2.
$$

Если обе стороны/проекция свободно обучаются без других constraints, collapse может сделать loss малым. Поэтому target, detach и normalization задаются явно. Alternative — pairwise score MSE либо KL между teacher/student candidate distributions. Score matching сохраняет выбранные relationships без требования одинаковых coordinates, но ограниченный набор pairs не определяет всю geometry.

Teacher language coverage, data и capacity student влияют независимо от loss. Прямое copy coordinates разных independently trained spaces не имеет смысла без alignment: rotations могут сохранить similarities и изменить coordinates. Teacher encoding cost включается в training budget.

## 5. Matryoshka и geometry diagnostics

Matryoshka Representation Learning обучает несколько prefixes одного vector:

$$
L=\sum_{d\in\mathcal D}\alpha_d\,
L_d\!\left(\operatorname{normalize}(z_{1:d})\right).
$$

Конкретные task losses/weights зависят от recipe. Prefix ordinary embedding без такого обучения не имеет гарантированной полезности. После truncation нужна согласованная renormalization/similarity и новый index contract. [Matryoshka](https://arxiv.org/abs/2205.13147).

Anisotropy — concentration vectors в ограниченных направлениях; hubness — некоторые points часто становятся neighbours многим queries. Norm drift влияет на dot-product ranking. Collapse делает vectors почти одинаковыми и уничтожает discriminative information. Проверять нужно variance, norms, neighbour quality и target task, а не только projection plot. Whitening/centering fit на разрешённой training части и сами меняют geometry.

## 6. STS, duplicates, clustering, anomalies и рекомендации

STS оценивает graded similarity, duplicate detection — identity по policy, retrieval — полезность ответа query. A≈B и B≈C не доказывают, что A duplicate C: transitive closure threshold graph может слить разные records цепочкой. Оценивать следует pair precision и ошибки resulting groups.

Clustering требует stability/semantics из отдельного курса. Необычный text не автоматически harmful/anomalous outcome. Item-text vectors могут помочь cold-start recommendations, но textual closeness не равна user preference; нужны behavioural labels с учётом exposure/popularity и temporal split.

Одна STS correlation не доказывает полезность всех приложений. Сравнивайте lexical baseline, frozen sentence-trained checkpoint и task adaptation на независимых cases с числами/отрицаниями/rare terms. Reindexing cost и compatibility версий учитываются при смене representation.

## 7. Контроль и практика модулей 22–25

1. Что делает false negative в InfoNCE? **Ответ:** off-diagonal pressure уменьшает similarity релевантного candidate, противореча задаче.
2. Гарантирует ли coordinate MSE одинаковый ranking всюду? **Ответ:** нет; зависит от coverage/normalization/projection и fit error.
3. Можно ли обрезать любой vector как Matryoshka? **Ответ:** технически да, но полезность prefixes без обучения/validation не гарантируется.
4. Равна ли document similarity предпочтению пользователя? **Ответ:** нет; recommendation supervision — отдельный target.

**Практика:** вычислите строку InfoNCE и gradient, затем объявите второй document тоже positive и объясните конфликт. Сравните dot/cosine разных norms. Спроектируйте teacher/student dimension mapping и две задачи с разным понятием similarity; укажите отдельные metrics и source splits.
