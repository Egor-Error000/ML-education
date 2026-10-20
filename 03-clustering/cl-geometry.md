# Похожесть, расстояния и постановка кластеризации

**Уровень:** общее ядро.

**Предпосылки:** [Постановка задачи и паспорт данных](<../01-data-features/01-reliability/df-task.md>), [Split, cross-validation и утечки](<../01-data-features/01-reliability/df-split.md>)

**Используют:** [Масштабирование, PCA, t-SNE и UMAP](<cl-projection.md>), [Центроидные методы: K-Means и K-Means++](<cl-kmeans.md>), [Иерархическая кластеризация](<cl-hierarchy.md>), [Плотностные методы: DBSCAN и HDBSCAN](<cl-density.md>), [Вероятностные смеси и EM](<cl-gmm.md>)

**Учебный вопрос:** Какую структуру данных мы называем кластером и почему это зависит от представления?

Векторное представление вводится внутри этой главы. Для прохождения модуля не требуется знать Transformer или обучение языковой модели.



## Карта вычисления и границы этапов

![Похожесть, расстояния и постановка кластеризации: входы, преобразования и выходы](<../оформление/course/cl-geometry.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:e33e213cc4cd87c11c -->
<a id="u-e33e213cc4cd87c11c"></a>

## Класс и кластер — разные понятия

<!-- /unit:e33e213cc4cd87c11c -->

<!-- unit:6eeeafe08da9e0966c -->
<a id="u-6eeeafe08da9e0966c"></a>

Класс определяется разметкой задачи; кластер — алгоритмом и геометрией данных. Изображения собак могут разделиться по фону, а документы об одной теме — по языку. Это не обязательно computational ошибка: features и distance выражали другое понятие similarity. До алгоритма нужно сформулировать, какая группировка полезна.

<!-- /unit:6eeeafe08da9e0966c -->

<!-- unit:28cc94ad0d749b0da3 -->
<a id="u-28cc94ad0d749b0da3"></a>

Объекты, единица независимости, будущие новые данные и downstream use задаются явно. Exploratory transductive analysis может fit по всему имеющемуся набору; deployment clustering на будущих объектах требует отдельного out-of-sample protocol. «Без labels» не означает «без evaluation».

<!-- /unit:28cc94ad0d749b0da3 -->

<!-- unit:383a4c7da695ac0ecc -->
<a id="u-383a4c7da695ac0ecc"></a>

### Представление и embedding

<!-- /unit:383a4c7da695ac0ecc -->

<!-- unit:9cd44f7ac06b3601ce -->
<a id="u-9cd44f7ac06b3601ce"></a>

Объект $x$ можно представить вектором признаков $z=f(x)\in\mathbb R^d$. Такое векторное представление называют embedding, если оно кодирует объект в координатах выбранного пространства; оно может быть вычислено простыми признаками или получено обученным encoder. Для кластеризации важен не только исходный объект, но и то, какие различия функция $f$ сохраняет. Transformer для этого определения не требуется.

<!-- /unit:9cd44f7ac06b3601ce -->

<!-- unit:48a27c03337a33ce24 -->
<a id="u-48a27c03337a33ce24"></a>

Например, пусть два текста получили векторы $(1,0)$ и $(0.9,0.1)$, а третий — $(0,1)$. По cosine первые два близки по направлению; это утверждение только о заданных координатах, не о метках и не о вероятностях. Одинаковая размерность двух независимо обученных embeddings также не гарантирует совместимые axes: один encoder может быть повёрнут относительно другого без изменения внутренних расстояний.

<!-- /unit:48a27c03337a33ce24 -->

<!-- unit:f3955a4a24a8978113 -->
<a id="u-f3955a4a24a8978113"></a>

## Евклидово и Manhattan distances

<!-- /unit:f3955a4a24a8978113 -->

<!-- unit:1ba14e248330ef32b2 -->
<a id="u-1ba14e248330ef32b2"></a>

Для $x,y\in\mathbb R^d$:

<!-- /unit:1ba14e248330ef32b2 -->

<!-- unit:e06291e8d9ea7d2249 -->
<a id="u-e06291e8d9ea7d2249"></a>

$$
d_2(x,y)=\sqrt{\sum_j(x_j-y_j)^2},
\qquad
d_1(x,y)=\sum_j|x_j-y_j|.
$$

<!-- /unit:e06291e8d9ea7d2249 -->

<!-- unit:719a9bd4a5df9df71a -->
<a id="u-719a9bd4a5df9df71a"></a>

Квадрат $d_2$ сильнее подчёркивает большие coordinate deviations. Manhattan даёт другой compromise, но не превращает любой method автоматически в robust estimator. K-Means minimises squared Euclidean distances и выбирает means; при sum absolute deviations natural centre — coordinatewise median. Смена distance без соответствующего centre update может изменить objective несогласованно.

<!-- /unit:719a9bd4a5df9df71a -->

<!-- unit:1e57bc63a80294d899 -->
<a id="u-1e57bc63a80294d899"></a>

Euclidean geometry не требует spherical clusters сама по себе: DBSCAN с Euclidean distance может найти изогнутую плотную область. Spherical/convex preference K-Means связан с nearest-centroid objective и Voronoi partition. «K-Means предполагает одинаковые spheres» — полезная интуиция ограничений, но не буквальный probabilistic generative assumption каждого запуска.

<!-- /unit:1e57bc63a80294d899 -->

<!-- unit:8d29685f2c5ea4f3dc -->
<a id="u-8d29685f2c5ea4f3dc"></a>

## Scale определяет вклад feature

<!-- /unit:8d29685f2c5ea4f3dc -->

<!-- unit:e5631a7e1f885acc47 -->
<a id="u-e5631a7e1f885acc47"></a>

### Числовой пример: масштаб меняет ближайшую точку

<!-- /unit:e5631a7e1f885acc47 -->

<!-- unit:f60a509cb45877352b -->
<a id="u-f60a509cb45877352b"></a>

Пусть A=(0,0), B=(1,100), C=(10,0). Raw distances $d(A,B)\approx100.005$, $d(A,C)=10$: C ближе. Если второй feature измерен в сотнях и делится на 100, B becomes (1,1), distance $\sqrt2$, и B ближе. Никакой алгоритм не узнает правильные units сам по двум числам.

<!-- /unit:f60a509cb45877352b -->

<!-- unit:889642b33b7e61d73c -->
<a id="u-889642b33b7e61d73c"></a>

Standardization использует train mean/std; robust scaling — другие estimates; domain weights выражают semantic importance. Features с нулевой variance требуют handling. Для prospective evaluation scaler/PCA fit только по allowed training data. Если задача intentionally transductive, использование всего unlabeled набора объявляется как часть protocol.

<!-- /unit:889642b33b7e61d73c -->

<!-- unit:6cc312f4800e35589a -->
<a id="u-6cc312f4800e35589a"></a>

## Cosine и normalization

<!-- /unit:6cc312f4800e35589a -->

<!-- unit:8ce7bc9a620704dc57 -->
<a id="u-8ce7bc9a620704dc57"></a>

Cosine similarity:

<!-- /unit:8ce7bc9a620704dc57 -->

<!-- unit:3ce074486a73dc51d5 -->
<a id="u-3ce074486a73dc51d5"></a>

$$
\cos(x,y)=\frac{x^\top y}{\|x\|_2\|y\|_2}.
$$

<!-- /unit:3ce074486a73dc51d5 -->

<!-- unit:54bd1b1857b72ea835 -->
<a id="u-54bd1b1857b72ea835"></a>

Она не определена для zero vectors без отдельной convention. Common cosine distance $1-\cos$ не обязана удовлетворять triangle inequality, поэтому слово «метрика» используется с осторожностью для algorithms, требующих настоящий metric.

<!-- /unit:54bd1b1857b72ea835 -->

<!-- unit:739f5026205272e983 -->
<a id="u-739f5026205272e983"></a>

Для unit vectors $u,v$:

<!-- /unit:739f5026205272e983 -->

<!-- unit:4d90fdc8e99d626b50 -->
<a id="u-4d90fdc8e99d626b50"></a>

$$
\|u-v\|_2^2=u^\top u+v^\top v-2u^\top v
=2[1-\cos(u,v)].
$$

<!-- /unit:4d90fdc8e99d626b50 -->

<!-- unit:f25357690485400615 -->
<a id="u-f25357690485400615"></a>

Так ranking pairwise distances эквивалентен cosine ranking после L2 normalization. Но standard K-Means mean centres не остаются unit norm. Spherical K-Means нормирует centres и оптимизирует другой closely related objective. Нельзя из identity pairwise distances автоматически заключить equivalence любых centroid algorithms.

<!-- /unit:f25357690485400615 -->

<!-- unit:17c49728cb875cf640 -->
<a id="u-17c49728cb875cf640"></a>

## Mahalanobis и корреляции

<!-- /unit:17c49728cb875cf640 -->

<!-- unit:e4d7ea3c3f97e9a15c -->
<a id="u-e4d7ea3c3f97e9a15c"></a>

Если covariance $\Sigma$ positive definite:

<!-- /unit:e4d7ea3c3f97e9a15c -->

<!-- unit:dffb7e64c72292ef07 -->
<a id="u-dffb7e64c72292ef07"></a>

$$
d_M(x,y)=\sqrt{(x-y)^\top\Sigma^{-1}(x-y)}.
$$

<!-- /unit:dffb7e64c72292ef07 -->

<!-- unit:5f3a581e69113a2a34 -->
<a id="u-5f3a581e69113a2a34"></a>

Это Euclidean distance после соответствующего whitening. Направление с большой естественной variance получает меньший weight; correlated features не дублируют сигнал так же, как при raw Euclidean. Но estimate covariance может быть шумным или singular при $d\ge n$; нужны regularization, reduced space или pseudoinverse с ясными consequences.

<!-- /unit:5f3a581e69113a2a34 -->

<!-- unit:ea2a707da21a8e2c5f -->
<a id="u-ea2a707da21a8e2c5f"></a>

Глобальная covariance может включать межкластерный signal, который whitening ослабит. Поэтому «учёт корреляций всегда лучше» неверен. Выбор distance проверяется через stability, domain review и downstream usefulness.

<!-- /unit:ea2a707da21a8e2c5f -->

<!-- unit:2b61c16f7c78446cb9 -->
<a id="u-2b61c16f7c78446cb9"></a>

## Контроль и практика

<!-- /unit:2b61c16f7c78446cb9 -->

<!-- unit:9af3d54e80eeb170c2 -->
<a id="u-9af3d54e80eeb170c2"></a>

1. Доказывает ли cluster совпадение semantic class? **Ответ:** нет; он отражает chosen representation/algorithm.
2. Почему unit-vector identity не делает ordinary K-Means spherical? **Ответ:** mean update создаёт centres разной нормы.
3. Что делать с zero embedding при cosine? **Ответ:** отдельная declared policy и диагностика причины, а не division by zero.

<!-- /unit:9af3d54e80eeb170c2 -->

<!-- unit:c98e501a50f1481824 -->
<a id="u-c98e501a50f1481824"></a>

**Практика:** повторите scale example, выведите unit-vector identity и выпишите expected similarity вашей задачи. Укажите nuisance factors, которые не должны определять clusters, и проверьте, представлены ли они слишком сильно в features.

<!-- /unit:c98e501a50f1481824 -->

<!-- unit:f9ae6ae3e62af5b50a -->
<a id="u-f9ae6ae3e62af5b50a"></a>

**Схема курса:** objects + feature scale → representation/normalization → distance or similarity → neighborhood/algorithm input → inspect neighbors in original space

<!-- /unit:f9ae6ae3e62af5b50a -->

## Результат главы и переход дальше

Ответьте на вопрос: **Какую структуру данных мы называем кластером и почему это зависит от представления?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Масштабирование, PCA, t-SNE и UMAP](<cl-projection.md>), [Центроидные методы: K-Means и K-Means++](<cl-kmeans.md>), [Иерархическая кластеризация](<cl-hierarchy.md>), [Плотностные методы: DBSCAN и HDBSCAN](<cl-density.md>), [Вероятностные смеси и EM](<cl-gmm.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
