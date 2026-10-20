# В. Сколько групп нужно задаче

[Оглавление](README.md). Пункт 3 и практика stability. **Происхождение:** дополнение; ни одно число K не объявляется истинным только по красивой картинке.

**Уровень:** общее ядро. **Предпосылки:** C2 and sampling. **Используют:** any result reporting.

## 1. Почему inertia не выбирает K

При увеличении числа centres K-Means может оставить прежние centres и добавить новый, поэтому optimum inertia не увеличивается. При $K=n$ каждый point может стать своим centre и inertia равна нулю. Следовательно, minimum training inertia всегда предпочитает максимально разрешённую granularity, а не meaningful grouping.

Для $(0,1,4,5)$ один centre 2.5 даёт inertia 17. Два centres 0.5/4.5 дают 1; четыре — 0. Переход от 1 к 2 действительно крупный, но one example не доказывает universal elbow rule. В сложных data может не быть выраженного излома или их может быть несколько.

Elbow curve — диагностический plot gain от K, который зависит от scale/representation и quality local fits. Нельзя сравнивать poorly initialized K=2 с удачным K=3 и приписывать весь improvement числу clusters. При каждом K нужны adequate restarts и comparable solver quality.

## 2. Silhouette и geometry preference

Для point $i$ пусть $a_i$ — mean distance к другим members своего cluster, $b_i$ — минимальная mean distance к members другого cluster:

$$
s_i=\frac{b_i-a_i}{\max(a_i,b_i)}.
$$

Большой positive score означает компактность относительно соседнего cluster в этой distance. Отрицательный говорит, что point в среднем ближе к другой group. Singleton convention часто задаёт score 0; все-singleton и one-cluster results не подходят стандартному global silhouette. [Определение и допустимые counts](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

Для point 0 в group {0,1}, соседней {4,5}: $a=1,b=4.5,s=7/9$. Для point 1: $a=1,b=3.5,s=5/7$. Mean четырёх points около 0.7460. У curved density clusters silhouette может быть ниже, чем у компактного искусственного partition, поэтому maximum silhouette — preference metric, а не oracle semantics.

Все сравнения делают в одном space/distance или явно признают, что оценивают разные geometries. Silhouette на t-SNE islands не является доказательством separation исходных embeddings.

## 3. Gap statistic

Gap сравнивает observed within-cluster dispersion $W_K$ с reference data без предполагаемой cluster structure:

$$
\operatorname{Gap}(K)=
\frac1B\sum_{b=1}^B\log W^*_{Kb}-\log W_K.
$$

Reference distribution, например uniform в выбранной bounding region, должна соответствовать null hypothesis и feature scaling. Выбор region/rotation влияет на результат. Если $W_K=0$, logarithm требует отдельного handling: нельзя вычислять такую formula механически на singleton solution.

С Monte Carlo spread $s_K=\sqrt{1+1/B}\,\operatorname{sd}(\log W^*_{Kb})$ распространённое правило выбирает smallest K, для которого $\operatorname{Gap}(K)\ge\operatorname{Gap}(K+1)-s_{K+1}$. Это учитывает uncertainty reference simulation, но не делает K объективно истинным вне модели null. [Оригинальная работа gap statistic](https://statistics.stanford.edu/technical-reports/estimating-number-clusters-dataset-gap-statistic).

Toy values: Gap(2)=0.8, Gap(3)=0.85, $s_3=0.1$. Условие для K=2 выполнено, потому что $0.8\ge0.75$. Если это первое выполненное условие в просмотренном диапазоне, выбирается 2. Это иллюстрация selection rule, а не рассчитанная simulation ваших data.

## 4. Bayesian mixtures и цена granularity

Finite GMM можно сравнивать по held-out likelihood/AIC/BIC при одном representation. Dirichlet Process mixture допускает потенциально неограниченное число components с prior, а фактическое количество занятых components зависит от data и concentration parameter. «Не нужно фиксировать K» не означает отсутствие hyperparameters и prior assumptions.

Component count не равен class count. Один non-Gaussian class может потребовать несколько Gaussian components, а overlapping semantic classes — не разделиться density model. Нужны domain review и downstream criterion: segmentations, пригодные для action, могут иметь другую granularity, чем natural density modes.

## 5. Stability

Повторяем seeds, subsampling, small perturbations features и nearby hyperparameters. Assignments сравниваем на common objects permutation-invariant metrics, например ARI. Стабильный result может быть consistently wrong по semantics, но нестабильный result требует осторожных выводов.

Не все algorithms имеют random seed: deterministic agglomerative тоже меняется при data perturbations. Если refit меняет labels, numeric IDs необходимо сопоставлять только через partitions/overlap, а не literal equality. При наличии truth labels отдельные validation labels могут помогать выбрать K, но final test нельзя многократно использовать для этого.

## 6. Контроль и практика

1. Почему minimum inertia не выбирает полезный K? **Ответ:** objective падает при дроблении вплоть до singletons.
2. Даёт ли gap единственный истинный K? **Ответ:** нет; он сравнивает с выбранным reference model и учитывает конкретную granularity.
3. Что означает высокий stability? **Ответ:** устойчивость partition к указанным изменениям, а не доказанное semantic correctness.

**Практика:** рассчитайте inertia для K=1/2/4 toy dataset. Найдите silhouette каждого point. Спроектируйте seeds/subsampling report, сохраняющий common IDs и долю noise; выберите K по заранее объявленному criterion и проверьте downstream usefulness.


**Схема курса:** candidate K + repeated subsamples/reference null → fit partitions → inertia/silhouette/gap/stability → compare criteria → report uncertainty, not a single true K

