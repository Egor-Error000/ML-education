# Качество разбиения: внутренние и внешние метрики

**Уровень:** общее ядро.

**Предпосылки:** [Центроидные методы: K-Means и K-Means++](<cl-kmeans.md>), [Иерархическая кластеризация](<cl-hierarchy.md>), [Плотностные методы: DBSCAN и HDBSCAN](<cl-density.md>), [Вероятностные смеси и EM](<cl-gmm.md>)

**Используют:** [Выбор числа кластеров и устойчивость](<cl-stability.md>), [Кластеризация замороженных представлений](<cl-frozen.md>), [Аномалии, псевдометки, потоки и IVF](<cl-systems.md>)

**Учебный вопрос:** Что измеряет метрика и как noise/coverage меняют сравнение?





## Карта вычисления и границы этапов

![Качество разбиения: внутренние и внешние метрики: входы, преобразования и выходы](<../оформление/course/cl-metrics.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:eebdbb5384c2f9460e -->
<a id="u-eebdbb5384c2f9460e"></a>

## Не сравнивать номера labels как классы

<!-- /unit:eebdbb5384c2f9460e -->

<!-- unit:7b5a590d1127bb0b14 -->
<a id="u-7b5a590d1127bb0b14"></a>

Предсказанные cluster IDs условны. True labels (A,A,B,B) и clusters (7,7,3,3) дают идеальное совпадение partitions. Literal accuracy после arbitrary mapping IDs может показать ноль. External clustering metrics сравнивают структуру без зависимости от permutation labels. Если используется matched accuracy, mapping обучается/определяется по заявленному protocol; она имеет другой смысл, чем готовая classifier accuracy.

<!-- /unit:7b5a590d1127bb0b14 -->

<!-- unit:6015f1d243535f7f22 -->
<a id="u-6015f1d243535f7f22"></a>

Contingency matrix $n_{kc}$ считает objects cluster $k$ и true class $c$. Row sums $a_k$, column sums $b_c$, total $n$. Эти counts позволяют вычислить several external measures.

<!-- /unit:6015f1d243535f7f22 -->

<!-- unit:b5852f620c7cf6ca05 -->
<a id="u-b5852f620c7cf6ca05"></a>

## Purity и её ловушка

<!-- /unit:b5852f620c7cf6ca05 -->

<!-- unit:c9cb9b0579c8be01e8 -->
<a id="u-c9cb9b0579c8be01e8"></a>

$$
\operatorname{Purity}=\frac1n\sum_k\max_c n_{kc}.
$$

<!-- /unit:c9cb9b0579c8be01e8 -->

<!-- unit:cc4e98087aadff553b -->
<a id="u-cc4e98087aadff553b"></a>

Если каждому point присвоить отдельный cluster, purity равна 1 независимо от meaningful structure. Поэтому purity не штрафует excessive fragmentation. Её следует сопровождать cluster count/size distribution и complementary measures. Для true A,A,B,B и crossed clusters (1,2,1,2) каждый cluster содержит по A/B, purity равна 0.5.

<!-- /unit:cc4e98087aadff553b -->

<!-- unit:3941d7640f83a4d693 -->
<a id="u-3941d7640f83a4d693"></a>

## ARI: согласие pairs с поправкой на случайность

<!-- /unit:3941d7640f83a4d693 -->

<!-- unit:568ca5e786d47b6ab2 -->
<a id="u-568ca5e786d47b6ab2"></a>

Пусть $\binom m2=m(m-1)/2$. Adjusted Rand Index:

<!-- /unit:568ca5e786d47b6ab2 -->

<!-- unit:b331880b947cc497c8 -->
<a id="u-b331880b947cc497c8"></a>

$$
\operatorname{ARI}=
\frac{\displaystyle\sum_{k,c}\binom{n_{kc}}2
-\dfrac{\left(\sum_k\binom{a_k}2\right)\left(\sum_c\binom{b_c}2\right)}{\binom n2}}
{\displaystyle\frac12\left[\sum_k\binom{a_k}2+\sum_c\binom{b_c}2\right]
-\dfrac{\left(\sum_k\binom{a_k}2\right)\left(\sum_c\binom{b_c}2\right)}{\binom n2}}.
$$

<!-- /unit:b331880b947cc497c8 -->

<!-- unit:e25d87f61db0a33405 -->
<a id="u-e25d87f61db0a33405"></a>

Adjustment относится к определённому random partition model с fixed marginals. ARI=1 для identical partitions, около 0 для chance agreement по этому baseline, возможны negative values. Это не probability и не обязано лежать в [0,1]. Degenerate denominators требуют library convention.

<!-- /unit:e25d87f61db0a33405 -->

<!-- unit:d69c6e768771024194 -->
<a id="u-d69c6e768771024194"></a>

Для crossed four-point example все $n_{kc}=1$, pair sum 0; row/column pair sums по 2, total pairs 6. Expected term $2/3$, denominator $4/3$, поэтому ARI=-0.5. Для четырёх singleton clusters при двух balanced true classes ARI=0, несмотря на purity=1.

<!-- /unit:d69c6e768771024194 -->

<!-- unit:c0ede2f92f6cbd4531 -->
<a id="u-c0ede2f92f6cbd4531"></a>

## NMI и информация

<!-- /unit:c0ede2f92f6cbd4531 -->

<!-- unit:983c97a128b61b15a1 -->
<a id="u-983c97a128b61b15a1"></a>

Mutual information:

<!-- /unit:983c97a128b61b15a1 -->

<!-- unit:5db07a1c48bcd6eba2 -->
<a id="u-5db07a1c48bcd6eba2"></a>

$$
I(K;C)=\sum_{k,c}\frac{n_{kc}}n
\log\frac{n\,n_{kc}}{a_kb_c}.
$$

<!-- /unit:5db07a1c48bcd6eba2 -->

<!-- unit:c2c78f2e86019dfcd2 -->
<a id="u-c2c78f2e86019dfcd2"></a>

Zero cells дают нулевой вклад. Одна convention NMI — arithmetic normalization:

<!-- /unit:c2c78f2e86019dfcd2 -->

<!-- unit:409d069908c7e4360b -->
<a id="u-409d069908c7e4360b"></a>

$$
\operatorname{NMI}=\frac{2I(K;C)}{H(K)+H(C)}.
$$

<!-- /unit:409d069908c7e4360b -->

<!-- unit:2b167f3353b5332145 -->
<a id="u-2b167f3353b5332145"></a>

Другие normalizers дают другие values; convention сообщается. NMI не является chance-adjusted: при большом количестве clusters может завышаться. Для singleton prediction и balanced two-class truth $I=\log2$, $H(K)=\log4$, $H(C)=\log2$, NMI равна $2/3$, а не 1. При необходимости рассматривают AMI как отдельную adjusted measure.

<!-- /unit:2b167f3353b5332145 -->

<!-- unit:726271eb09906fab72 -->
<a id="u-726271eb09906fab72"></a>

## Internal scores

<!-- /unit:726271eb09906fab72 -->

<!-- unit:f41947fb5395c28305 -->
<a id="u-f41947fb5395c28305"></a>

Silhouette раскрыт в главе [«Выбор числа кластеров и устойчивость»](<cl-stability.md>) и зависит от pairwise distance. Davies–Bouldin использует within scatter $S_k$ и centre distance $M_{kj}$:

<!-- /unit:f41947fb5395c28305 -->

<!-- unit:4a97dab4353f370479 -->
<a id="u-4a97dab4353f370479"></a>

$$
\operatorname{DB}=\frac1K\sum_k\max_{j\ne k}
\frac{S_k+S_j}{M_{kj}}.
$$

<!-- /unit:4a97dab4353f370479 -->

<!-- unit:f0ebc5beabdab71f4d -->
<a id="u-f0ebc5beabdab71f4d"></a>

Lower лучше по этому criterion. При стандартной Euclidean convention $S_k$ — mean distance до centre. Calinski–Harabasz сравнивает between и within dispersion:

<!-- /unit:f0ebc5beabdab71f4d -->

<!-- unit:0ce45d1970bb92a19b -->
<a id="u-0ce45d1970bb92a19b"></a>

$$
W=\sum_k\sum_{i\in C_k}\|x_i-\mu_k\|^2,
\quad
B=\sum_k n_k\|\mu_k-\mu\|^2,
\quad
\operatorname{CH}=\frac{B/(K-1)}{W/(n-K)}.
$$

<!-- /unit:0ce45d1970bb92a19b -->

<!-- unit:154d4aa234ccb7fdb6 -->
<a id="u-154d4aa234ccb7fdb6"></a>

Higher лучше при допустимых K и nondegenerate denominator. Для groups {0,1}/{4,5}: $S_1=S_2=0.5$, centre distance 4, DB=0.25. $W=1,B=16,n=4,K=2$, CH=32. Эти measures предпочитают определённую compact geometry и не оценивают universal semantics. Definitions: [DB](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html), [CH](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html).

<!-- /unit:154d4aa234ccb7fdb6 -->

<!-- unit:b4bdce91a3b67473b9 -->
<a id="u-b4bdce91a3b67473b9"></a>

## Noise и selection bias

<!-- /unit:b4bdce91a3b67473b9 -->

<!-- unit:b19a3061cf6b456285 -->
<a id="u-b19a3061cf6b456285"></a>

У DBSCAN/HDBSCAN noise часто имеет ID -1. Передав его обычной metric function, можно неосознанно считать все unrelated noise points одним cluster. Исключив noise, можно получить прекрасную метрику на 5% лёгких objects. Поэтому report содержит coverage/noise fraction, правило оценки noise и исходное число examples.

<!-- /unit:b19a3061cf6b456285 -->

<!-- unit:f5f2f06f0bdca58fdc -->
<a id="u-f5f2f06f0bdca58fdc"></a>

External evaluation использует independent labels; если по ним выбирали K/representation, они validation, а не test. Без labels добавляются expert audit типичных/пограничных cases, stability и downstream utility. Не нужно объединять разные internal scales в одно «среднее качество» без осмысленной normalization.

<!-- /unit:f5f2f06f0bdca58fdc -->

<!-- unit:3c88730b1675cd6ec3 -->
<a id="u-3c88730b1675cd6ec3"></a>

## Контроль и практика

<!-- /unit:3c88730b1675cd6ec3 -->

<!-- unit:bf779c5fd85012c13b -->
<a id="u-bf779c5fd85012c13b"></a>

1. Почему K-Means не находит произвольные curved groups? **Ответ:** nearest-centroid squared-error objective задаёт другую geometry partitions.
2. Является ли DBSCAN noise доказанным anomaly? **Ответ:** нет; это отсутствие нужной локальной плотности при chosen parameters.
3. Почему GMM нужны covariance safeguards? **Ответ:** component может схлопнуться на point и патологически повысить likelihood.

<!-- /unit:bf779c5fd85012c13b -->

<!-- unit:9e50d4a68dcbef6475 -->
<a id="u-9e50d4a68dcbef6475"></a>

**Практика:** повторите два K-Means updates и DBSCAN labels. Вычислите обе GMM responsibilities в 2D. Сравните shapes/assumptions algorithms и отдельно запишите, как будете присваивать group новому объекту, не предполагая `predict` у каждого метода.

<!-- /unit:9e50d4a68dcbef6475 -->

<!-- unit:4ba09b56c9c9d475de -->
<a id="u-4ba09b56c9c9d475de"></a>

**Схема курса:** representation → choose structure assumption → centroid updates / linkage merge / density neighborhood / GMM responsibilities → labels, hierarchy or probabilities

<!-- /unit:4ba09b56c9c9d475de -->

<!-- unit:b1137e16d375c5add3 -->
<a id="u-b1137e16d375c5add3"></a>

1. Почему purity 1 не доказывает хорошую clustering? **Ответ:** singletons всегда дают purity 1.
2. Почему ARI может быть отрицательным? **Ответ:** agreement хуже ожидаемого по chance baseline.
3. Что сообщать при исключении noise? **Ответ:** долю coverage/noise и правило отбора; score условен на оставшихся points.

<!-- /unit:b1137e16d375c5add3 -->

<!-- unit:b6d0b18ab57c8242ed -->
<a id="u-b6d0b18ab57c8242ed"></a>

**Практика:** вычислите purity/ARI для perfect permutation, crossed groups и singletons на четырёх objects. Затем DB/CH для toy points. Составьте report, где algorithm A покрывает все objects, B — половину; объясните, почему нельзя выбирать B только по silhouette оставшихся.

<!-- /unit:b6d0b18ab57c8242ed -->

<!-- unit:01e219df8bcc132e76 -->
<a id="u-01e219df8bcc132e76"></a>

**Схема курса:** partition + hidden evaluation labels or geometry → external contingency metric / internal metric → noise and coverage report → no test-label tuning

<!-- /unit:01e219df8bcc132e76 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Что измеряет метрика и как noise/coverage меняют сравнение?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Выбор числа кластеров и устойчивость](<cl-stability.md>), [Кластеризация замороженных представлений](<cl-frozen.md>), [Аномалии, псевдометки, потоки и IVF](<cl-systems.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
