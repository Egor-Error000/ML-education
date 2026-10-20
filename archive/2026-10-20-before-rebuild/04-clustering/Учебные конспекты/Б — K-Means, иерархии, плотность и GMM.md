# Б. Разные алгоритмы ищут разные виды групп

[Оглавление](README.md). Пункт 2 и практики K-Means/GMM/плотности. **Происхождение:** дополнительная глава по плану; все числа ниже — toy calculations.

**Уровень:** общее ядро. **Предпосылки:** C1, expectation/variance. **Используют:** C3–C4/C9.

## 1. K-Means и два ручных шага

Objective:

$$
J(z,\mu)=\sum_{i=1}^n\|x_i-\mu_{z_i}\|_2^2.
$$

При fixed centres каждый point назначается ближайшему. При fixed assignments mean minimizes squared distances. Эти два exact updates не увеличивают objective, но приводят к local optimum/стационарному решению, а не гарантированно global best.

Для points $(0,1,4,5)$ и initial centres $(0,1)$ первое assignment: $(\{0\},\{1,4,5\})$. New centres $(0,10/3)$. На следующем assignment point 1 переходит налево, получаем $(\{0,1\},\{4,5\})$, new centres $(0.5,4.5)$. Inertia после второго complete update равна 1. Initial inertia была 25. Empty cluster требует explicit reinitialization/policy; tie-breaking тоже часть implementation.

Outlier может сильно сдвинуть mean, потому что squared distance растёт быстро. Неравные densities/sizes и curved structures могут разбиваться неудобно. K-Means всё равно отдаст $K$ groups даже на одной continuous cloud; наличие partitions не доказывает естественные semantic classes.

K-Means++ выбирает первый centre случайно, subsequent — с probability proportional $D(x)^2$, где $D$ — distance до ближайшего уже выбранного centre. Это улучшает initialization quality, но не даёт global guarantee отдельного fit. Нужны несколько starts/seeds и контроль budget. Mini-batch вариант приближает updates для больших data; результат может отличаться от full-batch.

## 2. Hierarchical clustering

Agglomerative начинается с singletons и объединяет ближайшие groups, divisive начинает с большой group и разделяет. Dendrogram показывает hierarchy merges; срез определяет granularity. Height meaningful только относительно chosen linkage/distance.

Single linkage — minimum pair distance, склонен к цепочкам; complete — maximum, предпочитает bounded diameter; average — mean pair distance. Ward выбирает минимальный прирост within-cluster squared error:

$$
\Delta(A,B)=\frac{|A||B|}{|A|+|B|}\|\mu_A-\mu_B\|_2^2.
$$

Ward связан с Euclidean squared-error geometry и не является arbitrary-metric linkage. На больших $n$ хранение pairwise distances может требовать quadratic memory; precise complexity зависит от algorithm/linkage/implementation. Новый point обычно нельзя добавить простым `predict` без отдельного assignment rule или refit.

## 3. DBSCAN: core, border, noise

В выбранной convention epsilon-neighbourhood включает сам point. Core имеет не меньше `min_samples` neighbours; border не core, но находится рядом с core; noise не присоединён к density-connected cluster. Cluster расширяется через core connectivity. Border point сам не соединяет две dense groups как полноценный core bridge.

Toy data $(0,0.1,0.2,0.4,2)$, $\epsilon=0.21$, min_samples=3: 0,0.1,0.2 — core; 0.4 — border, потому что рядом с core 0.2, но имеет только два neighbours вместе с собой; 2 — noise. При увеличении epsilon отдельные области могут слиться, при уменьшении — распасться/стать noise.

Число clusters не задаётся, но quality зависит от radius, neighbour count и geometry. Разные densities плохо описываются одним radius. Noise — результат density criterion, а не автоматически истинная anomaly label. Memory worst case некоторых implementations quadratic при больших neighbourhoods. [DBSCAN API и conventions](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

## 4. HDBSCAN

HDBSCAN строит density hierarchy, используя core distances и mutual reachability, например:

$$
d_{\mathrm{mr}}(a,b)=\max\{d_k(a),d_k(b),d(a,b)\}.
$$

Затем hierarchy конденсируется по minimum cluster size и выбираются persistent groups. Это позволяет работать с несколькими density scales, но не устраняет все проблемы variable-density data и high dimensions.

`min_cluster_size` задаёт минимальный размер meaningful group, `min_samples` влияет на conservativeness/core-density estimate. Их нельзя считать одной настройкой; default связывает их только в конкретной implementation. Packages могут различаться в учёте самой точки при k-neighbour definition. Отчёт включает noise fraction и выбранный selection method. [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).

## 5. Gaussian Mixture Model и EM

GMM задаёт density:

$$
p(x)=\sum_{k=1}^K\pi_k\mathcal N(x\mid\mu_k,\Sigma_k),
\quad \pi_k\ge0,\quad\sum_k\pi_k=1.
$$

E-step computes responsibilities:

$$
r_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}.
$$

M-step с $N_k=\sum_i r_{ik}$:

$$
\pi_k^+=\frac{N_k}n,\quad
\mu_k^+=\frac{\sum_i r_{ik}x_i}{N_k},\quad
\Sigma_k^+=\frac{\sum_i r_{ik}(x_i-\mu_k^+)(x_i-\mu_k^+)^\top}{N_k}.
$$

Для 2D point $(0,0)$, centres $(0,0)$ и $(2,0)$, equal priors и identity covariances likelihood ratio равен $e^2$. Responsibility первого $1/(1+e^{-2})\approx0.8808$. Для point $(2,0)$ она симметрично 0.1192. Если обновлять только means по этим двум points, centres становятся примерно $(0.2384,0)$ и $(1.7616,0)$.

Full covariance допускает rotated ellipses, diagonal — оси features, spherical — scalar variance per component, tied — shared covariance. Gaussian components не обязательно совпадают semantic classes: несколько components могут описывать один сложный class.

Mixture likelihood может расти неограниченно при collapse covariance вокруг одной точки. Нужны covariance regularization, минимальные counts и несколько initializations. AIC $=2p-2\log\widehat L$, BIC $=p\log n-2\log\widehat L$ сравнивают fit/complexity при same data/representation и assumptions; lower лучше по данному criterion. Scores после разных dimensionality transforms нельзя бездумно сравнивать как одну и ту же likelihood.

## 6. Контроль и практика

1. Почему K-Means не находит произвольные curved groups? **Ответ:** nearest-centroid squared-error objective задаёт другую geometry partitions.
2. Является ли DBSCAN noise доказанным anomaly? **Ответ:** нет; это отсутствие нужной локальной плотности при chosen parameters.
3. Почему GMM нужны covariance safeguards? **Ответ:** component может схлопнуться на point и патологически повысить likelihood.

**Практика:** повторите два K-Means updates и DBSCAN labels. Вычислите обе GMM responsibilities в 2D. Сравните shapes/assumptions algorithms и отдельно запишите, как будете присваивать group новому объекту, не предполагая `predict` у каждого метода.


**Схема курса:** representation → choose structure assumption → centroid updates / linkage merge / density neighborhood / GMM responsibilities → labels, hierarchy or probabilities

