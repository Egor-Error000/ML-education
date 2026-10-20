# Масштабирование, PCA, t-SNE и UMAP

**Уровень:** общее ядро.

**Предпосылки:** [Похожесть, расстояния и постановка кластеризации](<cl-geometry.md>), [Числовые преобразования и масштабирование](<../01-data-features/02-features/feat-numeric.md>)

**Используют:** [Центроидные методы: K-Means и K-Means++](<cl-kmeans.md>), [Иерархическая кластеризация](<cl-hierarchy.md>), [Плотностные методы: DBSCAN и HDBSCAN](<cl-density.md>), [Вероятностные смеси и EM](<cl-gmm.md>)

**Учебный вопрос:** Какие отношения сохраняет проекция и как проверить искажения?





## Карта вычисления и границы этапов

![Масштабирование, PCA, t-SNE и UMAP: входы, преобразования и выходы](<../оформление/course/cl-projection.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:07498f702a6226fe19 -->
<a id="u-07498f702a6226fe19"></a>

## Высокая размерность не всегда бесполезна

<!-- /unit:07498f702a6226fe19 -->

<!-- unit:e78079808fd8327009 -->
<a id="u-e78079808fd8327009"></a>

Много irrelevant coordinates могут скрыть signal в distances, а neighbour search становится дороже. В некоторых random high-dimensional models relative distances концентрируются. Но это не универсальное утверждение, что любые 768-dimensional embeddings невозможно кластеризовать: intrinsic structure и training objective важны.

<!-- /unit:e78079808fd8327009 -->

<!-- unit:706d006196e06b4e82 -->
<a id="u-706d006196e06b4e82"></a>

Сокращение dimensions может удалить noise или одновременно удалить useful low-variance signal. Поэтому выбор dimension — часть representation experiment. Сначала scaling/normalization по задаче, затем transform fit на allowed data.

<!-- /unit:706d006196e06b4e82 -->

<!-- unit:96c569d48fa712d3ea -->
<a id="u-96c569d48fa712d3ea"></a>

## PCA через SVD

<!-- /unit:96c569d48fa712d3ea -->

<!-- unit:711ab35fa514503b78 -->
<a id="u-711ab35fa514503b78"></a>

Центрируем matrix $X\in\mathbb R^{n\times d}$ по train mean. SVD:

<!-- /unit:711ab35fa514503b78 -->

<!-- unit:45aacc94ea01787e4f -->
<a id="u-45aacc94ea01787e4f"></a>

$$
X=U\Sigma V^\top,\qquad
Z=XV_r=U_r\Sigma_r.
$$

<!-- /unit:45aacc94ea01787e4f -->

<!-- unit:b58a051aed47c2f815 -->
<a id="u-b58a051aed47c2f815"></a>

PCA сохраняет subspace максимальной variance, минимизируя squared reconstruction error среди rank-r linear projections. Explained variance ratio суммирует соответствующие squared singular values. Это критерий reconstruction/variance, а не labels или semantic separability.

<!-- /unit:b58a051aed47c2f815 -->

<!-- unit:f979e2c8f9cf8d5eba -->
<a id="u-f979e2c8f9cf8d5eba"></a>

Пример: features имеют variances 100 и 1, а true grouping различается только знаком второго feature. PCA до одной dimension может сохранить первую и потерять grouping. Standardization перед PCA может изменить выбор, но это уже другой geometry prior. Сохранять 95% variance не означает сохранять 95% информации о любой downstream task.

<!-- /unit:f979e2c8f9cf8d5eba -->

<!-- unit:ecb84980d099f484ab -->
<a id="u-ecb84980d099f484ab"></a>

Whitening дополнительно делит projected coordinates на estimated standard deviations. Это меняет distance weighting и усиливает слабые directions, включая noise; regularization нужна при small singular values. Утверждение «PCA и whitening одно и то же» неверно.

<!-- /unit:ecb84980d099f484ab -->

<!-- unit:2796de49f1e119379f -->
<a id="u-2796de49f1e119379f"></a>

## t-SNE: локальные соседства для карты

<!-- /unit:2796de49f1e119379f -->

<!-- unit:8126be070818da82a3 -->
<a id="u-8126be070818da82a3"></a>

t-SNE строит distributions pairwise similarities в исходном и low-dimensional spaces и минимизирует KL. Один из low-dimensional kernels:

<!-- /unit:8126be070818da82a3 -->

<!-- unit:15904035e5d3e4e85d -->
<a id="u-15904035e5d3e4e85d"></a>

$$
q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}
{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}},
\qquad
L=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

<!-- /unit:15904035e5d3e4e85d -->

<!-- unit:c678b4b4236a8f70c8 -->
<a id="u-c678b4b4236a8f70c8"></a>

Local structure имеет приоритет; relative island sizes и большие distances не обязаны faithfully отражать original geometry. Perplexity, initialization и optimization влияют на map. [Оригинальная статья t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html).

<!-- /unit:c678b4b4236a8f70c8 -->

<!-- unit:5658fe3b97fcb5c997 -->
<a id="u-5658fe3b97fcb5c997"></a>

Из двух separated islands нельзя непосредственно выводить наличие двух natural classes. Algorithm может разнести continuous manifold в visual patches. Обычный t-SNE fit не задаёт стандартный deterministic parametric mapping любого нового объекта; конкретные extensions имеют отдельный protocol.

<!-- /unit:5658fe3b97fcb5c997 -->

<!-- unit:89652e5c0944a6caf4 -->
<a id="u-89652e5c0944a6caf4"></a>

## UMAP для preprocessing

<!-- /unit:89652e5c0944a6caf4 -->

<!-- unit:c13f3a9865dd22e3fc -->
<a id="u-c13f3a9865dd22e3fc"></a>

UMAP строит neighbour graph и low-dimensional representation, приближённо сохраняющее выбранную локальную structure. `n_neighbors`, `min_dist`, metric и embedding dimension меняют geometry. Для clustering dimension не обязана быть 2: можно сохранять больше structure, чем нужно для картинки. [UMAP для clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html).

<!-- /unit:c13f3a9865dd22e3fc -->

<!-- unit:0e542bf3fa7e81ac7b -->
<a id="u-0e542bf3fa7e81ac7b"></a>

Проекция меняет densities и gaps. HDBSCAN после UMAP находит dense regions в transformed space, которые могут отличаться от исходных embeddings. Поэтому нужно проверять neighbours/assignments обратно в original space, seeds и nearby parameters. [Ограничения сохранения density](https://umap-learn.readthedocs.io/en/latest/faq.html).

<!-- /unit:0e542bf3fa7e81ac7b -->

<!-- unit:6a2e2742558cbacc62 -->
<a id="u-6a2e2742558cbacc62"></a>

UMAP transform новых inputs возможен в поддерживающей его implementation, но fitted graph/transform version сохраняются. Supervised UMAP с labels уже использует supervision: это нельзя смешивать с полностью unsupervised experiment без указания. Fit по full evaluation corpus относится к transductive setup и отличается от train-only transform.

<!-- /unit:6a2e2742558cbacc62 -->

<!-- unit:ade96265977fae06e4 -->
<a id="u-ade96265977fae06e4"></a>

## Протокол без визуального самообмана

<!-- /unit:ade96265977fae06e4 -->

<!-- unit:bf635ece4aaa5b5a9b -->
<a id="u-bf635ece4aaa5b5a9b"></a>

Сравните raw embeddings, L2 normalization, PCA нескольких dimensions и при необходимости UMAP. Зафиксируйте algorithm/K сначала, затем менять clustering. Для каждого сохраняйте fit scope, parameter config, runtime, stability и downstream measure. Visualization — инструмент осмотра, не единственная метрика выбора.

<!-- /unit:bf635ece4aaa5b5a9b -->

<!-- unit:4986b6fc379c3db2a9 -->
<a id="u-4986b6fc379c3db2a9"></a>

Проверка representatives полезна даже при хорошем silhouette: убедитесь, что clusters группируют нужное свойство, а не time/source. Если separation исчезает при другом seed projection, вывод о taxonomy слаб. Если downstream quality стабильно хороша, visual overlap не обязательно проблема: 2D map может терять существенные dimensions.

<!-- /unit:4986b6fc379c3db2a9 -->

<!-- unit:4eff977c3712a49686 -->
<a id="u-4eff977c3712a49686"></a>

## Контроль и практика

<!-- /unit:4eff977c3712a49686 -->

<!-- unit:7df10c711c61a78c16 -->
<a id="u-7df10c711c61a78c16"></a>

1. Что максимизирует PCA? **Ответ:** retained variance выбранного centred/scaled representation, не semantic accuracy.
2. Почему нельзя выбрать K по t-SNE islands? **Ответ:** map distortion и hyperparameters могут создавать/менять видимые gaps.
3. Что проверять после UMAP+HDBSCAN? **Ответ:** исходные neighbours/семантику, density distortion, seeds, параметры и coverage.

<!-- /unit:7df10c711c61a78c16 -->

<!-- unit:59d62931ce2c9206d6 -->
<a id="u-59d62931ce2c9206d6"></a>

**Практика:** придумайте high-variance nuisance и low-variance target feature. Опишите PCA result до/после scaling. Составьте plan сравнения четырёх representations и проверьте, что качество считается в сопоставимом space либо различие spaces явно объявлено.

<!-- /unit:59d62931ce2c9206d6 -->

<!-- unit:1a9c882ab541ace077 -->
<a id="u-1a9c882ab541ace077"></a>

**Схема курса:** high-dimensional points → projection objective → 2D coordinates → visualize local/global neighborhoods separately → validate clustering in original space

<!-- /unit:1a9c882ab541ace077 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Какие отношения сохраняет проекция и как проверить искажения?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Центроидные методы: K-Means и K-Means++](<cl-kmeans.md>), [Иерархическая кластеризация](<cl-hierarchy.md>), [Плотностные методы: DBSCAN и HDBSCAN](<cl-density.md>), [Вероятностные смеси и EM](<cl-gmm.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
