# Наборы категорий и pooling

**Уровень:** основа ветки.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Обучаемые embeddings и перенос представлений](<feat-embeddings.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как отличить множество от последовательности и что теряет агрегирование?





## Карта вычисления и границы этапов

![Наборы категорий и pooling: входы, преобразования и выходы](<../../оформление/course/feat-sets.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:183d2b7cdb7b339243 -->
<a id="u-183d2b7cdb7b339243"></a>

## Set, multiset или sequence

<!-- /unit:183d2b7cdb7b339243 -->

<!-- unit:624c333e685b1655e0 -->
<a id="u-624c333e685b1655e0"></a>

Список интересов пользователя может быть set: порядок не важен, повтор не добавляет смысла. История покупок — multiset, где counts важны, либо sequence, где важны время и порядок. Выбор representation должен сохранить информацию, нужную target. Называть все три случая «несколько tags» недостаточно.

<!-- /unit:624c333e685b1655e0 -->

<!-- unit:7638cc320883fa7311 -->
<a id="u-7638cc320883fa7311"></a>

Multi-hot vector имеет единицу в каждой присутствующей category. Для A,B,A при vocabulary A/B/C он равен $(1,1,0)$. Count representation даёт $(2,1,0)$. Sequence [A,B,A] отличается от [A,A,B], но обе representations его порядок стирают. Если recency критична, нужны temporal features или sequence model.

<!-- /unit:7638cc320883fa7311 -->

<!-- unit:27a86a599f9844576f -->
<a id="u-27a86a599f9844576f"></a>

Для multi-label target тоже бывает multi-hot output, но это другая сторона задачи. Input tags и independent output classes не следует смешивать: softmax target предполагает конкурирующие mutually exclusive classes, sigmoid/BCE может моделировать несколько simultaneous labels.

<!-- /unit:27a86a599f9844576f -->

<!-- unit:fd35614d4346154cd3 -->
<a id="u-fd35614d4346154cd3"></a>

## Pooling embeddings

<!-- /unit:fd35614d4346154cd3 -->

<!-- unit:89bcabd709892b996e -->
<a id="u-89bcabd709892b996e"></a>

Пусть active categories $c_1,\ldots,c_m$ имеют vectors $e_{c_i}$. Базовые варианты:

<!-- /unit:89bcabd709892b996e -->

<!-- unit:95cbc89ba7afdc9bb6 -->
<a id="u-95cbc89ba7afdc9bb6"></a>

$$
z_{\mathrm{sum}}=\sum_{i=1}^m e_{c_i},
\qquad
z_{\mathrm{mean}}=\frac1m\sum_{i=1}^m e_{c_i},
\qquad
(z_{\max})_j=\max_i(e_{c_i})_j.
$$

<!-- /unit:95cbc89ba7afdc9bb6 -->

<!-- unit:dc8a69d2be6e5e10f7 -->
<a id="u-dc8a69d2be6e5e10f7"></a>

Sum сохраняет влияние числа occurrences, mean нормирует length, max выбирает coordinatewise extremes. Ни один не универсально лучший. Для A=(1,0), B=(0,2), списка A,B,A получаем sum $(2,2)$, mean $(2/3,2/3)$, max $(1,2)$. Если сначала dedup set, sum станет $(1,2)$, mean $(0.5,1)$. Это разные assumptions о значимости repetitions.

<!-- /unit:dc8a69d2be6e5e10f7 -->

<!-- unit:fd73f79f74ca8f7707 -->
<a id="u-fd73f79f74ca8f7707"></a>

Empty set требует explicit vector/indicator. Mean нельзя делить на ноль; max пустого множества не имеет обычного конечного vector. PAD tokens должны исключаться из denominator и max, иначе длина padding изменит representation. Для sum нулевой PAD естественен только при корректной row/update policy.

<!-- /unit:fd73f79f74ca8f7707 -->

<!-- unit:228285db3cebf85902 -->
<a id="u-228285db3cebf85902"></a>

## Веса и обучаемый pooling

<!-- /unit:228285db3cebf85902 -->

<!-- unit:7a574f879d6e8ebe53 -->
<a id="u-7a574f879d6e8ebe53"></a>

Weighted pooling использует weights, например recency или confidence:

<!-- /unit:7a574f879d6e8ebe53 -->

<!-- unit:e2418775fd634804a5 -->
<a id="u-e2418775fd634804a5"></a>

$$
z=\frac{\sum_i w_i e_{c_i}}{\sum_i w_i},\qquad w_i\ge0.
$$

<!-- /unit:e2418775fd634804a5 -->

<!-- unit:269d1142dad8a72611 -->
<a id="u-269d1142dad8a72611"></a>

Сумма weights тоже должна быть ненулевой. Attention pooling вычисляет normalized weights по representations и, возможно, query. Если scoring зависит только от каждого set element и aggregation symmetric, result permutation-invariant. Position embeddings или order-dependent preprocessing меняют это свойство.

<!-- /unit:269d1142dad8a72611 -->

<!-- unit:35408e12f6404183a1 -->
<a id="u-35408e12f6404183a1"></a>

В mean pooling gradient одного occurrence масштабируется $1/m$; длинный список может ослаблять вклад каждого tag. При sum magnitude растёт с длиной. Добавление отдельного count feature позволяет mean representation явно учитывать size, но не восстанавливает весь потерянный multiset. Max передаёт gradient только selected coordinates, с отдельным поведением на ties.

<!-- /unit:35408e12f6404183a1 -->

<!-- unit:1c1af7d13b122265e3 -->
<a id="u-1c1af7d13b122265e3"></a>

## Что проверять в data pipeline

<!-- /unit:1c1af7d13b122265e3 -->

<!-- unit:822af82be62c7ff746 -->
<a id="u-822af82be62c7ff746"></a>

Canonicalize spelling/namespace; различать missing list и известный empty list. Ограничение длины/обрезка может терять редкие важные tags; порядок truncation следует определить. Выбор «первые 20» для неупорядоченного set может зависеть от случайной сериализации и давать нестабильные predictions.

<!-- /unit:822af82be62c7ff746 -->

<!-- unit:afaa54feafb563a16a -->
<a id="u-afaa54feafb563a16a"></a>

Новые tags mapped в UNK/OTHER по сохранённому contract. Если несколько unknown схлопываются, sum может отражать только их count, а mean — одинаковый неизвестный vector. Это нужно проверить отдельно, а не считать semantic generalization.

<!-- /unit:afaa54feafb563a16a -->

<!-- unit:01786ec07fceec227c -->
<a id="u-01786ec07fceec227c"></a>

Для event history preprocessing должен учитывать prediction time: будущая покупка в списке interests — leakage независимо от pooling method. Split по пользователям/времени остаётся основой honest evaluation.

<!-- /unit:01786ec07fceec227c -->

<!-- unit:d748fd4e8b70adb25a -->
<a id="u-d748fd4e8b70adb25a"></a>

## Контроль и практика

<!-- /unit:d748fd4e8b70adb25a -->

<!-- unit:a7fbc2036b3823b60d -->
<a id="u-a7fbc2036b3823b60d"></a>

1. Чем multi-hot отличается от counts? **Ответ:** multi-hot фиксирует presence, counts сохраняет multiplicity.
2. Сохраняет ли mean порядок? **Ответ:** нет; sum/mean/max являются symmetric set aggregations.
3. Как кодировать empty и missing? **Ответ:** явно по task contract; они могут иметь разные indicators/vectors и не должны приводить к division by zero.

<!-- /unit:a7fbc2036b3823b60d -->

<!-- unit:b9271f57ee94bad74f -->
<a id="u-b9271f57ee94bad74f"></a>

**Практика:** для A,B,A вычислите три pooling варианта с dedup и без. Переставьте элементы, добавьте PAD и empty case; проверьте ожидаемую invariance. Затем решите, нужен ли вашей задаче set, multiset или sequence, и перечислите информацию, потерянную выбранным encoding.

<!-- /unit:b9271f57ee94bad74f -->

<!-- unit:8450c646ad2e3131c0 -->
<a id="u-8450c646ad2e3131c0"></a>

**Схема курса:** set of categories → multi-hot/count/weighted IDs → sum/mean/attention pooling → fixed-size representation → downstream prediction

<!-- /unit:8450c646ad2e3131c0 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как отличить множество от последовательности и что теряет агрегирование?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
