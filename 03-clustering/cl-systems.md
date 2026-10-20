# Аномалии, псевдометки, потоки и IVF

**Уровень:** основа ветки.

**Предпосылки:** [Качество разбиения: внутренние и внешние метрики](<cl-metrics.md>), [Выбор числа кластеров и устойчивость](<cl-stability.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как использовать кластеризацию как компонент системы и обрабатывать новый объект?





## Карта вычисления и границы этапов

![Аномалии, псевдометки, потоки и IVF: входы, преобразования и выходы](<../оформление/course/cl-systems.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:5239f01c2aabab1744 -->
<a id="u-5239f01c2aabab1744"></a>

## Аномалия — редкость относительно контекста

<!-- /unit:5239f01c2aabab1744 -->

<!-- unit:fad4dd682d2f662694 -->
<a id="u-fad4dd682d2f662694"></a>

DBSCAN noise или большой distance до centre может служить anomaly score. Но редкий нормальный режим тоже может быть isolated, а массовый сбой — образовать плотный cluster. Поэтому anomaly label определяется operational consequences, а не одним density criterion.

<!-- /unit:fad4dd682d2f662694 -->

<!-- unit:4ae2b81bcfe265e83c -->
<a id="u-4ae2b81bcfe265e83c"></a>

В predictive maintenance разные режимы работы дают разные normal distributions. Сравнивать турбину в режиме запуска с нормой устойчивого полёта/работы может дать ложные alarms. Clustering regimes помогает condition analysis, если groups соответствуют физически meaningful states. Без labels/эксперта такая interpretation остаётся hypothesis.

<!-- /unit:4ae2b81bcfe265e83c -->

<!-- unit:64570ddc38bb409d34 -->
<a id="u-64570ddc38bb409d34"></a>

Distance-based score $s(x)=\min_k\|z(x)-\mu_k\|$ чувствителен к scaling и cluster spreads. GMM negative log-likelihood учитывает density assumptions, но тоже не равен probability failure. Threshold выбирается по validation с известными cases/costs либо по declared alert budget, а затем проверяется на future data.

<!-- /unit:64570ddc38bb409d34 -->

<!-- unit:17cb7ddc9d445561f7 -->
<a id="u-17cb7ddc9d445561f7"></a>

## Псевдометки

<!-- /unit:17cb7ddc9d445561f7 -->

<!-- unit:c5976ba8747e887e7f -->
<a id="u-c5976ba8747e887e7f"></a>

Cluster assignment можно использовать как training target encoder/classifier. Это self-training signal: model обучается воспроизводить structure, найденную предыдущим этапом. Если initial clusters отражали source/camera, student будет учиться source/camera, а не нужному object type.

<!-- /unit:c5976ba8747e887e7f -->

<!-- unit:5d124ded271c5629b2 -->
<a id="u-5d124ded271c5629b2"></a>

Confidence filtering оставляет надёжные examples, но может убрать редкие важные groups. Cluster size balancing меняет sampling distribution. Нужно сохранять coverage, class/cluster mass и manually reviewed cases. Высокая accuracy на pseudo-labels доказывает agreement с pseudo-teacher, а не correctness human taxonomy.

<!-- /unit:5d124ded271c5629b2 -->

<!-- unit:93afcf579bd27a1ff7 -->
<a id="u-93afcf579bd27a1ff7"></a>

При итеративном reclustering numeric IDs могут переставиться. Если classifier head сохраняется, требуется mapping или reset по protocol; иначе label 0 внезапно означает другую group. Такой label permutation issue отличается от semantic drift самой group.

<!-- /unit:93afcf579bd27a1ff7 -->

<!-- unit:e5b3f5ea48d7172140 -->
<a id="u-e5b3f5ea48d7172140"></a>

## IVF: clusters ускоряют поиск

<!-- /unit:e5b3f5ea48d7172140 -->

<!-- unit:91130ebfe2daf291ee -->
<a id="u-91130ebfe2daf291ee"></a>

Inverted file index предварительно делит database vectors по coarse centres. Для query выбираются ближайшие cells, затем поиск выполняется внутри их inverted lists. Упрощённо:

<!-- /unit:91130ebfe2daf291ee -->

<!-- unit:ffc7313cdbf73c447d -->
<a id="u-ffc7313cdbf73c447d"></a>

$$
\mathcal K(q)=\operatorname{arg\,top}_{n_{\mathrm{probe}},k}
\bigl(-\|q-\mu_k\|^2\bigr),\qquad
\mathcal C(q)=\bigcup_{k\in\mathcal K(q)}
\{x_i:z_i=k\}.
$$

<!-- /unit:ffc7313cdbf73c447d -->

<!-- unit:730f7570c4340bb2c5 -->
<a id="u-730f7570c4340bb2c5"></a>

Search рассматривает candidates $\mathcal C(q)$ вместо всей базы. При примерно равных cells ожидаемый scan count порядка $N\,n_{\mathrm{probe}}/n_{\mathrm{list}}$, плюс стоимость coarse search. На практике cell imbalance и metadata filters меняют cost.

<!-- /unit:730f7570c4340bb2c5 -->

<!-- unit:c69d7e4a38c5b965f8 -->
<a id="u-c69d7e4a38c5b965f8"></a>

Для $N=1\,000\,000,n_{\mathrm{list}}=1000,n_{\mathrm{probe}}=10$ получается около 10 000 candidates, то есть 1% базы, без overhead. Но nearest true neighbour может лежать в непросмотренной cell, особенно около boundary. Больше probes обычно повышает recall и latency; при всех cells approximation этой части исчезает. [FAISS: index trade-offs](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

<!-- /unit:c69d7e4a38c5b965f8 -->

<!-- unit:55a3c97e5a493898cb -->
<a id="u-55a3c97e5a493898cb"></a>

IVF grouping не обязано быть human semantic taxonomy. Оно оптимизирует candidate routing. Quantization/compression может добавить ещё одну approximation независимо от coarse partitioning. Нужны ANN recall относительно exact top-k и downstream answer/retrieval quality; хороший cluster silhouette не измеряет search recall.

<!-- /unit:55a3c97e5a493898cb -->

<!-- unit:55da64d40c406a035d -->
<a id="u-55da64d40c406a035d"></a>

## Новые objects и drift

<!-- /unit:55da64d40c406a035d -->

<!-- unit:74924bba024f3845fb -->
<a id="u-74924bba024f3845fb"></a>

K-Means assign по nearest centre естественен, GMM даёт responsibilities. Для DBSCAN/agglomerative естественного universal `predict` может не быть: нужно refit, nearest-core rule, separate classifier либо supported approximate assignment extension с собственными limitations. Присвоение nearest centroid произвольному density cluster меняет semantics метода.

<!-- /unit:74924bba024f3845fb -->

<!-- unit:2beaa0c8988086bf8b -->
<a id="u-2beaa0c8988086bf8b"></a>

Новые points могут образовать новую group, которую fixed model не обнаружит. Drift требует мониторить distance distributions, coverage, cell loads и downstream quality. Refit index/clusterer меняет centres и иногда IDs; version, migration и audit сохраняются.

<!-- /unit:2beaa0c8988086bf8b -->

<!-- unit:4db4b82860066590e6 -->
<a id="u-4db4b82860066590e6"></a>

## Единый эксперимент

<!-- /unit:4db4b82860066590e6 -->

<!-- unit:f34f0a854c6cb209c7 -->
<a id="u-f34f0a854c6cb209c7"></a>

Сначала паспорт данных и split/group/time; затем geometry/scaling/PCA, один fixed encoder baseline. Вручную проверить distance и toy algorithms. Выбрать compact-centre, density или probabilistic method под hypothesis, а не все metrics сразу под победителя.

<!-- /unit:f34f0a854c6cb209c7 -->

<!-- unit:9091067a292f6ec8fa -->
<a id="u-9091067a292f6ec8fa"></a>

Report содержит metrics с объявленной noise policy, stability seeds/subsampling, runtime/memory, representatives, borderline/noise cases и out-of-sample rule. При наличии truth labels сохраняется independent test; без них нужны domain audit и usefulness для конкретного downstream action. Сравнение не должно скрывать discarded difficult objects.

<!-- /unit:9091067a292f6ec8fa -->

<!-- unit:98ccf3ab9eabe356f2 -->
<a id="u-98ccf3ab9eabe356f2"></a>

## Контроль и практика

<!-- /unit:98ccf3ab9eabe356f2 -->

<!-- unit:85a6e0bf5b5754c00d -->
<a id="u-85a6e0bf5b5754c00d"></a>

1. Равен ли noise failure label? **Ответ:** нет; редкость и опасность различаются.
2. Что измеряет pseudo-label accuracy? **Ответ:** agreement с generated labels, не независимую истинность.
3. Что происходит при увеличении nprobe? **Ответ:** обычно больше candidate coverage/recall за дополнительную стоимость; это search trade-off, не новый semantic class count.

<!-- /unit:85a6e0bf5b5754c00d -->

<!-- unit:750411399b1dc7e6a4 -->
<a id="u-750411399b1dc7e6a4"></a>

**Практика:** для одного corpus спроектируйте три разных применения одних clusters: review anomalies, pseudo-label training и IVF routing. Для каждого задайте свою метрику успеха. Затем завершите общий report с explicit new-object strategy и примером вывода, который не следует из одного silhouette score.

<!-- /unit:750411399b1dc7e6a4 -->

<!-- unit:7aae6bd00801c40894 -->
<a id="u-7aae6bd00801c40894"></a>

**Схема курса:** fit representation/partition → detect outlier or low-confidence point → human/external validation → optional pseudo-label → IVF routing or stream update → monitor drift

<!-- /unit:7aae6bd00801c40894 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как использовать кластеризацию как компонент системы и обрабатывать новый объект?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
