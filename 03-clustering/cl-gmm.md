# Вероятностные смеси и EM

**Уровень:** общее ядро.

**Предпосылки:** [Похожесть, расстояния и постановка кластеризации](<cl-geometry.md>), [Масштабирование, PCA, t-SNE и UMAP](<cl-projection.md>)

**Используют:** [Качество разбиения: внутренние и внешние метрики](<cl-metrics.md>)

**Учебный вопрос:** Как мягкая принадлежность и обновление параметров повышают likelihood?




## Мягкие компоненты вместо жёстких центров

Gaussian mixture задаёт $p(x)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)$, где веса неотрицательны и суммируются в 1. E-step вычисляет responsibilities $r_{ik}=\pi_k\mathcal N_k(x_i)/\sum_j\pi_j\mathcal N_j(x_i)$. M-step обновляет веса, средние и covariance с этими responsibilities. Component — часть вероятностной модели, а не автоматически реальный класс.

При двух одномерных компонентах с равными priors, variance=1 и means 0 и 2 точка x=0 имеет относительные densities 1 и exp(−2). Responsibilities приблизительно 0.881 и 0.119. Для x=1 они равны 0.5/0.5. Общий нормировочный множитель Gaussian сокращается в этом частном расчёте; при разных covariance его игнорировать нельзя.

Likelihood может патологически расти при схлопывании covariance на отдельной точке. Нужны ограничения covariance, regularization и диагностика. AIC/BIC сравнивают модели при своих предпосылках и числе параметров; разные covariance families имеют разную сложность. Новый объект получает posterior responsibilities при фиксированных обученных параметрах.


## Карта вычисления и границы этапов

![Вероятностные смеси и EM: входы, преобразования и выходы](<../оформление/course/cl-gmm.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:9d3490bba57ad6d197 -->
<a id="u-9d3490bba57ad6d197"></a>

## Gaussian Mixture Model и EM

<!-- /unit:9d3490bba57ad6d197 -->

<!-- unit:3c28b7570baf3f322e -->
<a id="u-3c28b7570baf3f322e"></a>

GMM задаёт density:

<!-- /unit:3c28b7570baf3f322e -->

<!-- unit:0d6528325979f36849 -->
<a id="u-0d6528325979f36849"></a>

$$
p(x)=\sum_{k=1}^K\pi_k\mathcal N(x\mid\mu_k,\Sigma_k),
\quad \pi_k\ge0,\quad\sum_k\pi_k=1.
$$

<!-- /unit:0d6528325979f36849 -->

<!-- unit:bb4b4883329016557e -->
<a id="u-bb4b4883329016557e"></a>

E-step computes responsibilities:

<!-- /unit:bb4b4883329016557e -->

<!-- unit:372812deefc9436471 -->
<a id="u-372812deefc9436471"></a>

$$
r_{ik}=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}.
$$

<!-- /unit:372812deefc9436471 -->

<!-- unit:515c2f07eb86bbe122 -->
<a id="u-515c2f07eb86bbe122"></a>

M-step с $N_k=\sum_i r_{ik}$:

<!-- /unit:515c2f07eb86bbe122 -->

<!-- unit:c4ca4113b34e3770e0 -->
<a id="u-c4ca4113b34e3770e0"></a>

$$
\pi_k^+=\frac{N_k}n,\quad
\mu_k^+=\frac{\sum_i r_{ik}x_i}{N_k},\quad
\Sigma_k^+=\frac{\sum_i r_{ik}(x_i-\mu_k^+)(x_i-\mu_k^+)^\top}{N_k}.
$$

<!-- /unit:c4ca4113b34e3770e0 -->

<!-- unit:ef4933bd82a005e557 -->
<a id="u-ef4933bd82a005e557"></a>

Для 2D point $(0,0)$, centres $(0,0)$ и $(2,0)$, equal priors и identity covariances likelihood ratio равен $e^2$. Responsibility первого $1/(1+e^{-2})\approx0.8808$. Для point $(2,0)$ она симметрично 0.1192. Если обновлять только means по этим двум points, centres становятся примерно $(0.2384,0)$ и $(1.7616,0)$.

<!-- /unit:ef4933bd82a005e557 -->

<!-- unit:1555905ea75d34e2d3 -->
<a id="u-1555905ea75d34e2d3"></a>

Full covariance допускает rotated ellipses, diagonal — оси features, spherical — scalar variance per component, tied — shared covariance. Gaussian components не обязательно совпадают semantic classes: несколько components могут описывать один сложный class.

<!-- /unit:1555905ea75d34e2d3 -->

<!-- unit:37010379c07e02c671 -->
<a id="u-37010379c07e02c671"></a>

Mixture likelihood может расти неограниченно при collapse covariance вокруг одной точки. Нужны covariance regularization, минимальные counts и несколько initializations. AIC $=2p-2\log\widehat L$, BIC $=p\log n-2\log\widehat L$ сравнивают fit/complexity при same data/representation и assumptions; lower лучше по данному criterion. Scores после разных dimensionality transforms нельзя бездумно сравнивать как одну и ту же likelihood.

<!-- /unit:37010379c07e02c671 -->

## Самостоятельная практика

**Практика:** Для двух Gaussian components с means 0 и 2, variance=1 и равными priors вычислите responsibilities x=0 и x=1. Затем измените prior первой компоненты на 0.8 и пересчитайте нормировку. Объясните отличие posterior probability от расстояния до центра.

## Результат главы и переход дальше

Ответьте на вопрос: **Как мягкая принадлежность и обновление параметров повышают likelihood?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Качество разбиения: внутренние и внешние метрики](<cl-metrics.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
