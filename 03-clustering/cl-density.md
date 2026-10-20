# Плотностные методы: DBSCAN и HDBSCAN

**Уровень:** общее ядро.

**Предпосылки:** [Похожесть, расстояния и постановка кластеризации](<cl-geometry.md>), [Масштабирование, PCA, t-SNE и UMAP](<cl-projection.md>)

**Используют:** [Качество разбиения: внутренние и внешние метрики](<cl-metrics.md>)

**Учебный вопрос:** Как связность плотных областей отличается от близости к центру?




## Плотность и связность

DBSCAN задаёт радиус ε и min_samples. Здесь точка считается входящей в собственную ε-окрестность. Core имеет не менее min_samples соседей; border достижима от core, но сама не имеет нужной плотности; noise не отнесена к найденной плотной области. Это результат выбранных параметров, а не доказательство аномальности.

Пусть точки на прямой равны 0,0.1,0.2,2.0, ε=0.15, min_samples=3. У 0.1 соседи {0,0.1,0.2}, значит это core. У крайних 0 и 0.2 только по два соседа; они border. Точка 2.0 — noise. Если уменьшить ε до 0.05, все точки станут noise при том же threshold.

HDBSCAN рассматривает иерархию плотностных уровней и выбирает устойчивые группы. Параметры всё ещё задают допустимый масштаб и минимальный размер. В высокой размерности выбранная метрика может перестать различать локальную плотность. Новые объекты назначают только по явно выбранной процедуре; не следует считать batch clustering и streaming prediction одним интерфейсом.


## Карта вычисления и границы этапов

![Плотностные методы: DBSCAN и HDBSCAN: входы, преобразования и выходы](<../оформление/course/cl-density.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:8b4e2ec47475b9fe20 -->
<a id="u-8b4e2ec47475b9fe20"></a>

## DBSCAN: core, border, noise

<!-- /unit:8b4e2ec47475b9fe20 -->

<!-- unit:6580d0cdbfda9a073e -->
<a id="u-6580d0cdbfda9a073e"></a>

В выбранной convention epsilon-neighbourhood включает сам point. Core имеет не меньше `min_samples` neighbours; border не core, но находится рядом с core; noise не присоединён к density-connected cluster. Cluster расширяется через core connectivity. Border point сам не соединяет две dense groups как полноценный core bridge.

<!-- /unit:6580d0cdbfda9a073e -->

<!-- unit:f87c79b0d128330644 -->
<a id="u-f87c79b0d128330644"></a>

Toy data $(0,0.1,0.2,0.4,2)$, $\epsilon=0.21$, min_samples=3: 0,0.1,0.2 — core; 0.4 — border, потому что рядом с core 0.2, но имеет только два neighbours вместе с собой; 2 — noise. При увеличении epsilon отдельные области могут слиться, при уменьшении — распасться/стать noise.

<!-- /unit:f87c79b0d128330644 -->

<!-- unit:ab14c85036dba35c97 -->
<a id="u-ab14c85036dba35c97"></a>

Число clusters не задаётся, но quality зависит от radius, neighbour count и geometry. Разные densities плохо описываются одним radius. Noise — результат density criterion, а не автоматически истинная anomaly label. Memory worst case некоторых implementations quadratic при больших neighbourhoods. [DBSCAN API и conventions](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

<!-- /unit:ab14c85036dba35c97 -->

<!-- unit:1142987d09e5e84dae -->
<a id="u-1142987d09e5e84dae"></a>

## HDBSCAN

<!-- /unit:1142987d09e5e84dae -->

<!-- unit:93004a5b8cf528c580 -->
<a id="u-93004a5b8cf528c580"></a>

HDBSCAN строит density hierarchy, используя core distances и mutual reachability, например:

<!-- /unit:93004a5b8cf528c580 -->

<!-- unit:e78df040c7a90e6528 -->
<a id="u-e78df040c7a90e6528"></a>

$$
d_{\mathrm{mr}}(a,b)=\max\{d_k(a),d_k(b),d(a,b)\}.
$$

<!-- /unit:e78df040c7a90e6528 -->

<!-- unit:87ac40d7f40934be1d -->
<a id="u-87ac40d7f40934be1d"></a>

Затем hierarchy конденсируется по minimum cluster size и выбираются persistent groups. Это позволяет работать с несколькими density scales, но не устраняет все проблемы variable-density data и high dimensions.

<!-- /unit:87ac40d7f40934be1d -->

<!-- unit:f91e9523d3692eedf2 -->
<a id="u-f91e9523d3692eedf2"></a>

`min_cluster_size` задаёт минимальный размер meaningful group, `min_samples` влияет на conservativeness/core-density estimate. Их нельзя считать одной настройкой; default связывает их только в конкретной implementation. Packages могут различаться в учёте самой точки при k-neighbour definition. Отчёт включает noise fraction и выбранный selection method. [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).

<!-- /unit:f91e9523d3692eedf2 -->

## Самостоятельная практика

**Практика:** Для 0,0.1,0.2,2.0 повторите core/border/noise при ε=0.15 и min_samples=3, считая саму точку соседом. Измените min_samples на 2 и объясните, почему taxonomy меняется. Отдельно задайте noise policy метрики.

## Результат главы и переход дальше

Ответьте на вопрос: **Как связность плотных областей отличается от близости к центру?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Качество разбиения: внутренние и внешние метрики](<cl-metrics.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
