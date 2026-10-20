# З. Снижение размерности может помочь и может создать иллюзию

[Оглавление](README.md). Пункт 8 плана; читать после А до основных algorithms. **Происхождение:** дополнение к обсуждённым embeddings, с отдельным разбором preprocessing и visualization.

**Уровень:** общее ядро как diagnostic; t-SNE/UMAP theory — углубление. **Предпосылки:** C1, covariance/SVD. **Используют:** C5–C7 and reports.

## 1. Высокая размерность не всегда бесполезна

Много irrelevant coordinates могут скрыть signal в distances, а neighbour search становится дороже. В некоторых random high-dimensional models relative distances концентрируются. Но это не универсальное утверждение, что любые 768-dimensional embeddings невозможно кластеризовать: intrinsic structure и training objective важны.

Сокращение dimensions может удалить noise или одновременно удалить useful low-variance signal. Поэтому выбор dimension — часть representation experiment. Сначала scaling/normalization по задаче, затем transform fit на allowed data.

## 2. PCA через SVD

Центрируем matrix $X\in\mathbb R^{n\times d}$ по train mean. SVD:

$$
X=U\Sigma V^\top,\qquad
Z=XV_r=U_r\Sigma_r.
$$

PCA сохраняет subspace максимальной variance, минимизируя squared reconstruction error среди rank-r linear projections. Explained variance ratio суммирует соответствующие squared singular values. Это критерий reconstruction/variance, а не labels или semantic separability.

Пример: features имеют variances 100 и 1, а true grouping различается только знаком второго feature. PCA до одной dimension может сохранить первую и потерять grouping. Standardization перед PCA может изменить выбор, но это уже другой geometry prior. Сохранять 95% variance не означает сохранять 95% информации о любой downstream task.

Whitening дополнительно делит projected coordinates на estimated standard deviations. Это меняет distance weighting и усиливает слабые directions, включая noise; regularization нужна при small singular values. Утверждение «PCA и whitening одно и то же» неверно.

## 3. t-SNE: локальные соседства для карты

t-SNE строит distributions pairwise similarities в исходном и low-dimensional spaces и минимизирует KL. Один из low-dimensional kernels:

$$
q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}
{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}},
\qquad
L=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

Local structure имеет приоритет; relative island sizes и большие distances не обязаны faithfully отражать original geometry. Perplexity, initialization и optimization влияют на map. [Оригинальная статья t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html).

Из двух separated islands нельзя непосредственно выводить наличие двух natural classes. Algorithm может разнести continuous manifold в visual patches. Обычный t-SNE fit не задаёт стандартный deterministic parametric mapping любого нового объекта; конкретные extensions имеют отдельный protocol.

## 4. UMAP для preprocessing

UMAP строит neighbour graph и low-dimensional representation, приближённо сохраняющее выбранную локальную structure. `n_neighbors`, `min_dist`, metric и embedding dimension меняют geometry. Для clustering dimension не обязана быть 2: можно сохранять больше structure, чем нужно для картинки. [UMAP для clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html).

Проекция меняет densities и gaps. HDBSCAN после UMAP находит dense regions в transformed space, которые могут отличаться от исходных embeddings. Поэтому нужно проверять neighbours/assignments обратно в original space, seeds и nearby parameters. [Ограничения сохранения density](https://umap-learn.readthedocs.io/en/latest/faq.html).

UMAP transform новых inputs возможен в поддерживающей его implementation, но fitted graph/transform version сохраняются. Supervised UMAP с labels уже использует supervision: это нельзя смешивать с полностью unsupervised experiment без указания. Fit по full evaluation corpus относится к transductive setup и отличается от train-only transform.

## 5. Протокол без визуального самообмана

Сравните raw embeddings, L2 normalization, PCA нескольких dimensions и при необходимости UMAP. Зафиксируйте algorithm/K сначала, затем менять clustering. Для каждого сохраняйте fit scope, parameter config, runtime, stability и downstream measure. Visualization — инструмент осмотра, не единственная метрика выбора.

Проверка representatives полезна даже при хорошем silhouette: убедитесь, что clusters группируют нужное свойство, а не time/source. Если separation исчезает при другом seed projection, вывод о taxonomy слаб. Если downstream quality стабильно хороша, visual overlap не обязательно проблема: 2D map может терять существенные dimensions.

## 6. Контроль и практика

1. Что максимизирует PCA? **Ответ:** retained variance выбранного centred/scaled representation, не semantic accuracy.
2. Почему нельзя выбрать K по t-SNE islands? **Ответ:** map distortion и hyperparameters могут создавать/менять видимые gaps.
3. Что проверять после UMAP+HDBSCAN? **Ответ:** исходные neighbours/семантику, density distortion, seeds, параметры и coverage.

**Практика:** придумайте high-variance nuisance и low-variance target feature. Опишите PCA result до/после scaling. Составьте plan сравнения четырёх representations и проверьте, что качество считается в сопоставимом space либо различие spaces явно объявлено.


**Схема курса:** high-dimensional points → projection objective → 2D coordinates → visualize local/global neighborhoods separately → validate clustering in original space

