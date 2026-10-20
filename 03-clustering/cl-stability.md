# Выбор числа кластеров и устойчивость

**Уровень:** общее ядро.

**Предпосылки:** [Качество разбиения: внутренние и внешние метрики](<cl-metrics.md>)

**Используют:** [Кластеризация замороженных представлений](<cl-frozen.md>), [Аномалии, псевдометки, потоки и IVF](<cl-systems.md>)

**Учебный вопрос:** Как выбрать разрешение разбиения без универсального оптимального K?





## Карта вычисления и границы этапов

![Выбор числа кластеров и устойчивость: входы, преобразования и выходы](<../оформление/course/cl-stability.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:e44c109ab166cb2ecd -->
<a id="u-e44c109ab166cb2ecd"></a>

## Почему inertia не выбирает K

<!-- /unit:e44c109ab166cb2ecd -->

<!-- unit:c119bd3e10d2488cad -->
<a id="u-c119bd3e10d2488cad"></a>

При увеличении числа centres K-Means может оставить прежние centres и добавить новый, поэтому optimum inertia не увеличивается. При $K=n$ каждый point может стать своим centre и inertia равна нулю. Следовательно, minimum training inertia всегда предпочитает максимально разрешённую granularity, а не meaningful grouping.

<!-- /unit:c119bd3e10d2488cad -->

<!-- unit:89b55a0a609062687a -->
<a id="u-89b55a0a609062687a"></a>

Для $(0,1,4,5)$ один centre 2.5 даёт inertia 17. Два centres 0.5/4.5 дают 1; четыре — 0. Переход от 1 к 2 действительно крупный, но one example не доказывает universal elbow rule. В сложных data может не быть выраженного излома или их может быть несколько.

<!-- /unit:89b55a0a609062687a -->

<!-- unit:017b52091fc9f121f5 -->
<a id="u-017b52091fc9f121f5"></a>

Elbow curve — диагностический plot gain от K, который зависит от scale/representation и quality local fits. Нельзя сравнивать poorly initialized K=2 с удачным K=3 и приписывать весь improvement числу clusters. При каждом K нужны adequate restarts и comparable solver quality.

<!-- /unit:017b52091fc9f121f5 -->

<!-- unit:7c6ea77df98564dcd8 -->
<a id="u-7c6ea77df98564dcd8"></a>

## Silhouette и geometry preference

<!-- /unit:7c6ea77df98564dcd8 -->

<!-- unit:7378bff7f258af0deb -->
<a id="u-7378bff7f258af0deb"></a>

Для point $i$ пусть $a_i$ — mean distance к другим members своего cluster, $b_i$ — минимальная mean distance к members другого cluster:

<!-- /unit:7378bff7f258af0deb -->

<!-- unit:66e5e0c19e53961482 -->
<a id="u-66e5e0c19e53961482"></a>

$$
s_i=\frac{b_i-a_i}{\max(a_i,b_i)}.
$$

<!-- /unit:66e5e0c19e53961482 -->

<!-- unit:85874df23b61df436d -->
<a id="u-85874df23b61df436d"></a>

Большой positive score означает компактность относительно соседнего cluster в этой distance. Отрицательный говорит, что point в среднем ближе к другой group. Singleton convention часто задаёт score 0; все-singleton и one-cluster results не подходят стандартному global silhouette. [Определение и допустимые counts](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

<!-- /unit:85874df23b61df436d -->

<!-- unit:04393c6af82b939918 -->
<a id="u-04393c6af82b939918"></a>

Для point 0 в group {0,1}, соседней {4,5}: $a=1,b=4.5,s=7/9$. Для point 1: $a=1,b=3.5,s=5/7$. Mean четырёх points около 0.7460. У curved density clusters silhouette может быть ниже, чем у компактного искусственного partition, поэтому maximum silhouette — preference metric, а не oracle semantics.

<!-- /unit:04393c6af82b939918 -->

<!-- unit:56c9795564d69b6a1d -->
<a id="u-56c9795564d69b6a1d"></a>

Все сравнения делают в одном space/distance или явно признают, что оценивают разные geometries. Silhouette на t-SNE islands не является доказательством separation исходных embeddings.

<!-- /unit:56c9795564d69b6a1d -->

<!-- unit:8c9bc216d693479d57 -->
<a id="u-8c9bc216d693479d57"></a>

## Gap statistic

<!-- /unit:8c9bc216d693479d57 -->

<!-- unit:c5bf29bd94bff6362b -->
<a id="u-c5bf29bd94bff6362b"></a>

Gap сравнивает observed within-cluster dispersion $W_K$ с reference data без предполагаемой cluster structure:

<!-- /unit:c5bf29bd94bff6362b -->

<!-- unit:906df8c680fcf1f320 -->
<a id="u-906df8c680fcf1f320"></a>

$$
\operatorname{Gap}(K)=
\frac1B\sum_{b=1}^B\log W^*_{Kb}-\log W_K.
$$

<!-- /unit:906df8c680fcf1f320 -->

<!-- unit:dce04372684d869755 -->
<a id="u-dce04372684d869755"></a>

Reference distribution, например uniform в выбранной bounding region, должна соответствовать null hypothesis и feature scaling. Выбор region/rotation влияет на результат. Если $W_K=0$, logarithm требует отдельного handling: нельзя вычислять такую formula механически на singleton solution.

<!-- /unit:dce04372684d869755 -->

<!-- unit:bade9d34d8006f20f8 -->
<a id="u-bade9d34d8006f20f8"></a>

С Monte Carlo spread $s_K=\sqrt{1+1/B}\,\operatorname{sd}(\log W^*_{Kb})$ распространённое правило выбирает smallest K, для которого $\operatorname{Gap}(K)\ge\operatorname{Gap}(K+1)-s_{K+1}$. Это учитывает uncertainty reference simulation, но не делает K объективно истинным вне модели null. [Оригинальная работа gap statistic](https://statistics.stanford.edu/technical-reports/estimating-number-clusters-dataset-gap-statistic).

<!-- /unit:bade9d34d8006f20f8 -->

<!-- unit:088b91c4973a887152 -->
<a id="u-088b91c4973a887152"></a>

Toy values: Gap(2)=0.8, Gap(3)=0.85, $s_3=0.1$. Условие для K=2 выполнено, потому что $0.8\ge0.75$. Если это первое выполненное условие в просмотренном диапазоне, выбирается 2. Это иллюстрация selection rule, а не рассчитанная simulation ваших data.

<!-- /unit:088b91c4973a887152 -->

<!-- unit:3f1b55be56bec77a9f -->
<a id="u-3f1b55be56bec77a9f"></a>

## Bayesian mixtures и цена granularity

<!-- /unit:3f1b55be56bec77a9f -->

<!-- unit:9489f29fd7212ecb6d -->
<a id="u-9489f29fd7212ecb6d"></a>

Finite GMM можно сравнивать по held-out likelihood/AIC/BIC при одном representation. Dirichlet Process mixture допускает потенциально неограниченное число components с prior, а фактическое количество занятых components зависит от data и concentration parameter. «Не нужно фиксировать K» не означает отсутствие hyperparameters и prior assumptions.

<!-- /unit:9489f29fd7212ecb6d -->

<!-- unit:5eba45ff5a26d8ce0c -->
<a id="u-5eba45ff5a26d8ce0c"></a>

Component count не равен class count. Один non-Gaussian class может потребовать несколько Gaussian components, а overlapping semantic classes — не разделиться density model. Нужны domain review и downstream criterion: segmentations, пригодные для action, могут иметь другую granularity, чем natural density modes.

<!-- /unit:5eba45ff5a26d8ce0c -->

<!-- unit:4fa7b2afb11caf820d -->
<a id="u-4fa7b2afb11caf820d"></a>

## Stability

<!-- /unit:4fa7b2afb11caf820d -->

<!-- unit:95585229ac34d6202e -->
<a id="u-95585229ac34d6202e"></a>

Повторяем seeds, subsampling, small perturbations features и nearby hyperparameters. Assignments сравниваем на common objects permutation-invariant metrics, например ARI. Стабильный result может быть consistently wrong по semantics, но нестабильный result требует осторожных выводов.

<!-- /unit:95585229ac34d6202e -->

<!-- unit:ce2ab39f250870b652 -->
<a id="u-ce2ab39f250870b652"></a>

Не все algorithms имеют random seed: deterministic agglomerative тоже меняется при data perturbations. Если refit меняет labels, numeric IDs необходимо сопоставлять только через partitions/overlap, а не literal equality. При наличии truth labels отдельные validation labels могут помогать выбрать K, но final test нельзя многократно использовать для этого.

<!-- /unit:ce2ab39f250870b652 -->

<!-- unit:36f40480a3e2d5a326 -->
<a id="u-36f40480a3e2d5a326"></a>

## Контроль и практика

<!-- /unit:36f40480a3e2d5a326 -->

<!-- unit:5de49922efae4bf509 -->
<a id="u-5de49922efae4bf509"></a>

1. Почему minimum inertia не выбирает полезный K? **Ответ:** objective падает при дроблении вплоть до singletons.
2. Даёт ли gap единственный истинный K? **Ответ:** нет; он сравнивает с выбранным reference model и учитывает конкретную granularity.
3. Что означает высокий stability? **Ответ:** устойчивость partition к указанным изменениям, а не доказанное semantic correctness.

<!-- /unit:5de49922efae4bf509 -->

<!-- unit:846ca4735c614e43ba -->
<a id="u-846ca4735c614e43ba"></a>

**Практика:** рассчитайте inertia для K=1/2/4 toy dataset. Найдите silhouette каждого point. Спроектируйте seeds/subsampling report, сохраняющий common IDs и долю noise; выберите K по заранее объявленному criterion и проверьте downstream usefulness.

<!-- /unit:846ca4735c614e43ba -->

<!-- unit:db9dbf3a46055fa90b -->
<a id="u-db9dbf3a46055fa90b"></a>

**Схема курса:** candidate K + repeated subsamples/reference null → fit partitions → inertia/silhouette/gap/stability → compare criteria → report uncertainty, not a single true K

<!-- /unit:db9dbf3a46055fa90b -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как выбрать разрешение разбиения без универсального оптимального K?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Кластеризация замороженных представлений](<cl-frozen.md>), [Аномалии, псевдометки, потоки и IVF](<cl-systems.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
