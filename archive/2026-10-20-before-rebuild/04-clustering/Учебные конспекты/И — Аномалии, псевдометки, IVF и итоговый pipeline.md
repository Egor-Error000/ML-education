# И. Clustering как часть решения, а не самоцель

[Оглавление](README.md). Пункт 9 и итоговая практика. **Основа:** S6 обсуждает retrieval и embeddings; использование IVF/псевдометок/режимов — дополнительное углубление. Исторические ссылки плана на проекты не считаются подтверждением ваших выполненных результатов.

**Уровень:** основа ветки by task. **Предпосылки:** C1–C4 and chosen application. **Используют:** anomaly discovery, pseudo-label, ANN routing.

## 1. Аномалия — редкость относительно контекста

DBSCAN noise или большой distance до centre может служить anomaly score. Но редкий нормальный режим тоже может быть isolated, а массовый сбой — образовать плотный cluster. Поэтому anomaly label определяется operational consequences, а не одним density criterion.

В predictive maintenance разные режимы работы дают разные normal distributions. Сравнивать турбину в режиме запуска с нормой устойчивого полёта/работы может дать ложные alarms. Clustering regimes помогает condition analysis, если groups соответствуют физически meaningful states. Без labels/эксперта такая interpretation остаётся hypothesis.

Distance-based score $s(x)=\min_k\|z(x)-\mu_k\|$ чувствителен к scaling и cluster spreads. GMM negative log-likelihood учитывает density assumptions, но тоже не равен probability failure. Threshold выбирается по validation с известными cases/costs либо по declared alert budget, а затем проверяется на future data.

## 2. Псевдометки

Cluster assignment можно использовать как training target encoder/classifier. Это self-training signal: model обучается воспроизводить structure, найденную предыдущим этапом. Если initial clusters отражали source/camera, student будет учиться source/camera, а не нужному object type.

Confidence filtering оставляет надёжные examples, но может убрать редкие важные groups. Cluster size balancing меняет sampling distribution. Нужно сохранять coverage, class/cluster mass и manually reviewed cases. Высокая accuracy на pseudo-labels доказывает agreement с pseudo-teacher, а не correctness human taxonomy.

При итеративном reclustering numeric IDs могут переставиться. Если classifier head сохраняется, требуется mapping или reset по protocol; иначе label 0 внезапно означает другую group. Такой label permutation issue отличается от semantic drift самой group.

## 3. IVF: clusters ускоряют поиск

Inverted file index предварительно делит database vectors по coarse centres. Для query выбираются ближайшие cells, затем поиск выполняется внутри их inverted lists. Упрощённо:

$$
\mathcal K(q)=\operatorname{arg\,top}_{n_{\mathrm{probe}},k}
\bigl(-\|q-\mu_k\|^2\bigr),\qquad
\mathcal C(q)=\bigcup_{k\in\mathcal K(q)}
\{x_i:z_i=k\}.
$$

Search рассматривает candidates $\mathcal C(q)$ вместо всей базы. При примерно равных cells ожидаемый scan count порядка $N\,n_{\mathrm{probe}}/n_{\mathrm{list}}$, плюс стоимость coarse search. На практике cell imbalance и metadata filters меняют cost.

Для $N=1\,000\,000,n_{\mathrm{list}}=1000,n_{\mathrm{probe}}=10$ получается около 10 000 candidates, то есть 1% базы, без overhead. Но nearest true neighbour может лежать в непросмотренной cell, особенно около boundary. Больше probes обычно повышает recall и latency; при всех cells approximation этой части исчезает. [FAISS: index trade-offs](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

IVF grouping не обязано быть human semantic taxonomy. Оно оптимизирует candidate routing. Quantization/compression может добавить ещё одну approximation независимо от coarse partitioning. Нужны ANN recall относительно exact top-k и downstream answer/retrieval quality; хороший cluster silhouette не измеряет search recall.

## 4. Новые objects и drift

K-Means assign по nearest centre естественен, GMM даёт responsibilities. Для DBSCAN/agglomerative естественного universal `predict` может не быть: нужно refit, nearest-core rule, separate classifier либо supported approximate assignment extension с собственными limitations. Присвоение nearest centroid произвольному density cluster меняет semantics метода.

Новые points могут образовать новую group, которую fixed model не обнаружит. Drift требует мониторить distance distributions, coverage, cell loads и downstream quality. Refit index/clusterer меняет centres и иногда IDs; version, migration и audit сохраняются.

## 5. Единый эксперимент

Сначала паспорт данных и split/group/time; затем geometry/scaling/PCA, один fixed encoder baseline. Вручную проверить distance и toy algorithms. Выбрать compact-centre, density или probabilistic method под hypothesis, а не все metrics сразу под победителя.

Report содержит metrics с объявленной noise policy, stability seeds/subsampling, runtime/memory, representatives, borderline/noise cases и out-of-sample rule. При наличии truth labels сохраняется independent test; без них нужны domain audit и usefulness для конкретного downstream action. Сравнение не должно скрывать discarded difficult objects.

## 6. Контроль и практика

1. Равен ли noise failure label? **Ответ:** нет; редкость и опасность различаются.
2. Что измеряет pseudo-label accuracy? **Ответ:** agreement с generated labels, не независимую истинность.
3. Что происходит при увеличении nprobe? **Ответ:** обычно больше candidate coverage/recall за дополнительную стоимость; это search trade-off, не новый semantic class count.

**Практика:** для одного corpus спроектируйте три разных применения одних clusters: review anomalies, pseudo-label training и IVF routing. Для каждого задайте свою метрику успеха. Затем завершите общий report с explicit new-object strategy и примером вывода, который не следует из одного silhouette score.


**Схема курса:** fit representation/partition → detect outlier or low-confidence point → human/external validation → optional pseudo-label → IVF routing or stream update → monitor drift

