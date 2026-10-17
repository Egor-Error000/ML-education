# З. Как оценить encoder без переоценки его возможностей

[Оглавление](README.md). Группа H, модули **35–37**. **Основа:** evaluation из S3 и retrieval из S6; task-specific statistics и interpretation — расширение по плану. Полные общие разделы находятся в [данных](../../01-data/Учебные%20конспекты/README.md).

## 1. Метрика следует единице задачи

Classification: macro/micro-F1, PR-AUC и cost threshold. NER: strict entity F1 с type/boundaries. QA: exact match и token-overlap F1 с declared normalization плюс unanswerable quality. STS: Pearson linear association и Spearman rank correlation — разные properties. High correlation не означает calibrated absolute scores.

Retrieval для queries $q$:

$$
\operatorname{MRR}=\frac1{|Q|}\sum_q\frac1{r_q},
$$

где вклад 0, если relevant не найден в cutoff. Для graded labels:

$$
\operatorname{DCG}@k=\sum_{i=1}^k\frac{2^{g_i}-1}{\log_2(i+1)},
\qquad
\operatorname{nDCG}@k=\frac{\operatorname{DCG}@k}{\operatorname{IDCG}@k}.
$$

Если IDCG=0, нужна declared convention. Для grades (2,0,1) DCG@3=3.5, ideal (2,1,0) даёт $3+1/\log_2 3$, nDCG около 0.96394. MRR этого ranking равен 1, потому что первый result relevant; MRR не видит ухудшение порядка остальных релевантных docs.

Recall@k относительно qrels зависит от полноты judgements. Unjudged document не обязательно irrelevant. Aggregate queries без judged relevant отдельно определяются protocol, иначе denominators несопоставимы. Classification, retrieval и ANN approximation scores нельзя объединить в одну «accuracy encoder».

## 2. Benchmarks и uncertainty

[BEIR](https://arxiv.org/abs/2104.08663) предлагает heterogeneous retrieval tasks, [MTEB](https://arxiv.org/abs/2210.07316) — разнообразие embedding tasks, [MMTEB](https://arxiv.org/abs/2502.13595) — multilingual expansion. Средний leaderboard score относится к versions/subsets/protocol, не универсальному deployment ranking.

Для честного сравнения фиксируют dataset revisions, split, encoder/pooling/prefixes и search budget. Парное сравнение на одних queries учитывает общую difficulty. Bootstrap resamples query/document groups, когда именно они независимы; correlated token rows одного document не создают тысячи independent trials.

Seeds отражают training variation, dataset intervals — sampling uncertainty. Report effect size, slices и cost. Выбор maximum из множества configurations/seed по test делает test development set. Небольшой средний gain не отменяет серьёзную regression критичного rare slice.

## 3. Calibration и отказ

Temperature scaling classifier logits использует $p_T=\operatorname{softmax}(z/T)$, $T>0$, fit по calibration NLL. Положительный scalar T сохраняет argmax и ranking classes данного example; он не исправляет wrong class ordering или missing evidence. Для independent labels protocol calibration может отличаться.

Brier/NLL и reliability bins проверяют probabilities, а не только accuracy. Softmax confidence не universal OOD detector: unfamiliar input может получить confident prediction. Нужны held-out unknown/domain/time cases, а не только in-distribution histogram.

Selective prediction принимает set $\mathcal A_\tau=\{i:c_i\ge\tau\}$:

$$
\operatorname{Coverage}(\tau)=\frac{|\mathcal A_\tau|}n,
\qquad
\operatorname{Risk}(\tau)=
\frac{\sum_{i\in\mathcal A_\tau}\mathbf1[\widehat y_i\ne y_i]}{|\mathcal A_\tau|}.
$$

При пустом set risk не определён. Если из 100 inputs принято 60 и в них 3 ошибки, coverage 0.6, risk 0.05. Отказ от остальных 40 требует operational route: review, additional retrieval или явный no-answer. Нельзя сообщать только 95% accepted accuracy без coverage.

## 4. Robustness tests с ожидаемым поведением

Typos, OCR noise, code switching, длинный context, числа, quotations и negation проверяют разные risks. Harmless spelling change должен часто сохранять label, смена «есть» на «нет» может обязана его изменить. Blanket invariance ко всем perturbations неверна.

Test cases фиксируют expected outcome и reason. Если annotation policy неоднозначна, сначала разрешается label issue, а не подбор model confidence. Изменение time/domain distribution, missing classes и new entities оценивается отдельно; one temperature не устраняет произвольный concept drift.

## 5. Attribution, probes и causal diagnosis

Gradient/saliency показывает local sensitivity, attention inspection — внутренние weights выбранного layer/head. Ни одно не является готовым человеческим объяснением causal decision. Probe может декодировать feature hidden state, хотя main head его не использует.

Integrated gradients для baseline $x'$ и input $x$ задаёт attribution:

$$
\operatorname{IG}_i(x)=(x_i-x_i')
\int_0^1\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}\,d\alpha.
$$

Для text interpolation обычно идёт в embedding space, а не между integer IDs; baseline/path определяют interpretation. Нужны controls и actual perturbations: source metadata removal, matched negation pairs, label-preserving paraphrase. Decodability, sensitivity и causal use различаются. Первичный метод: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365).

Shortcut может быть длиной, source или entity name. Высокое качество random split тогда не переносится на new source. Для проверки гипотезы «использует отрицание» сравнивают minimally differing pairs и контроль без semantic change, а не только наличие token «не» среди ярких saliency features.

## 6. Data access и privacy в прикладном contract

Retriever и PII extractor тоже обрабатывают adversarial/untrusted input. Embedded instructions или special-token-like strings требуют task tests; encoder-only architecture не даёт абсолютной защищённости любой обвязки. ACL/filter применяется до выдачи недоступных results по contract системы.

PII head может пропустить entity. Удаление известных имён не доказывает полной anonymization: косвенные identifiers и context остаются. Для реального использования задаются retention/logging/masking policies и audit соответствующих failures. Это конкретные properties pipeline, а не автоматически получаемое качество checkpoint.

## 7. Контроль и практика модулей 35–37

1. Почему MRR может быть 1 при nonideal ranking? **Ответ:** он учитывает только first relevant rank.
2. Исправляет ли T scaling wrong argmax? **Ответ:** positive global temperature сохраняет argmax.
3. Почему probe не доказывает causal use? **Ответ:** информация может присутствовать в state, но не влиять на actual head decision.
4. Почему accepted accuracy без coverage неполна? **Ответ:** model могла отвергнуть большую часть трудных inputs.

**Практика:** посчитайте MRR/nDCG двух rankings, постройте coverage-risk для шести scores. Предложите три robustness pairs с expected labels и probe control. Подготовьте paired evaluation protocol с независимой calibration частью и объяснением limits, без универсального вывода из одного benchmark.
