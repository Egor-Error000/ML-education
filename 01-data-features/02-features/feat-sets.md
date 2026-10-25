# Наборы категорий и pooling

**Уровень:** основа ветки.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Обучаемые embeddings и перенос представлений](<feat-embeddings.md>)

**Используют:** Завершение выбранной ветки: самостоятельный эксперимент по практике этой главы.

**Учебный вопрос:** Как отличить множество от последовательности и что теряет агрегирование?

## Карта вычисления и границы этапов

![Наборы категорий и pooling: входы, преобразования и выходы](<../../оформление/course/feat-sets.svg>)

Сумма, среднее и максимум сохраняют разные свойства множества. Пустой набор и повтор элемента требуют явного решения: перестановочная инвариантность не определяет обработку кратности.

## Set, multiset или sequence

Список интересов пользователя может быть set: порядок не важен, повтор не добавляет смысла. История покупок — multiset, где counts важны, либо sequence, где важны время и порядок. Выбор representation должен сохранить информацию, нужную target. Называть все три случая «несколько tags» недостаточно.

Multi-hot vector имеет единицу в каждой присутствующей category. Для A,B,A при vocabulary A/B/C он равен $(1,1,0)$. Count representation даёт $(2,1,0)$. Sequence [A,B,A] отличается от [A,A,B], но обе representations его порядок стирают. Если recency критична, нужны temporal features (признаки) или sequence model.

Для multi-label target тоже бывает multi-hot output, но это другая сторона задачи. Input tags и independent output classes не следует смешивать: softmax target предполагает конкурирующие mutually exclusive classes, sigmoid/BCE может моделировать несколько simultaneous labels (метки).

## Pooling (агрегирование состояний) embeddings (векторные представления)

Пусть active categories $c_1,\ldots,c_m$ имеют vectors $e_{c_i}$. Базовые варианты:

$$
z_{\mathrm{sum}}=\sum_{i=1}^m e_{c_i},
\qquad
z_{\mathrm{mean}}=\frac1m\sum_{i=1}^m e_{c_i},
\qquad
(z_{\max})_j=\max_i(e_{c_i})_j.
$$

Sum сохраняет влияние числа occurrences, mean нормирует length, max выбирает coordinatewise extremes. Ни один не универсально лучший. Для A=(1,0), B=(0,2), списка A,B,A получаем sum $(2,2)$, mean $(2/3,2/3)$, max $(1,2)$. Если сначала dedup set, sum станет $(1,2)$, mean $(0.5,1)$. Это разные assumptions о значимости repetitions.

Empty set требует explicit vector/indicator. Mean нельзя делить на ноль; max пустого множества не имеет обычного конечного vector. PAD tokens (токены) должны исключаться из denominator и max, иначе длина padding (дополнение до общей длины) изменит representation. Для sum нулевой PAD естественен только при корректной row/update policy.

## Веса и обучаемый pooling

Weighted pooling использует weights, например recency или confidence (уверенность по принятой оценке):

$$
z=\frac{\sum_i w_i e_{c_i}}{\sum_i w_i},\qquad w_i\ge0.
$$

Сумма weights тоже должна быть ненулевой. Attention (механизм внимания) pooling вычисляет normalized weights по representations и, возможно, query (запрос). Если scoring зависит только от каждого set element и aggregation symmetric, result permutation-invariant. Position embeddings или order-dependent preprocessing меняют это свойство.

В mean pooling (усреднение выбранных состояний) gradient (градиент) одного occurrence масштабируется $1/m$; длинный список может ослаблять вклад каждого tag. При sum magnitude растёт с длиной. Добавление отдельного count feature (признак) позволяет mean representation явно учитывать size, но не восстанавливает весь потерянный multiset. Max передаёт gradient только selected coordinates, с отдельным поведением на ties.

## Что проверять в data pipeline

Canonicalize spelling/namespace; различать missing list и известный empty list. Ограничение длины/обрезка может терять редкие важные tags; порядок truncation следует определить. Выбор «первые 20» для неупорядоченного set может зависеть от случайной сериализации и давать нестабильные predictions.

Новые tags mapped в UNK/OTHER по сохранённому contract. Если несколько unknown схлопываются, sum может отражать только их count, а mean — одинаковый неизвестный vector. Это нужно проверить отдельно, а не считать semantic generalization.

Для event history preprocessing должен учитывать prediction time: будущая покупка в списке interests — leakage (утечка недоступной информации) независимо от pooling method. Split (разбиение данных) по пользователям/времени остаётся основой honest evaluation.

## Контроль и практика

1. Чем multi-hot отличается от counts? **Ответ:** multi-hot фиксирует presence, counts сохраняет multiplicity.
2. Сохраняет ли mean порядок? **Ответ:** нет; sum/mean/max являются symmetric set aggregations.
3. Как кодировать empty и missing? **Ответ:** явно по task contract; они могут иметь разные indicators/vectors и не должны приводить к division by zero.

**Практика:** для A,B,A вычислите три pooling варианта с dedup и без. Переставьте элементы, добавьте PAD и empty case; проверьте ожидаемую invariance. Затем решите, нужен ли вашей задаче set, multiset или sequence, и перечислите информацию, потерянную выбранным encoding.

**Схема курса:** set of categories → multi-hot/count/weighted IDs → sum/mean/attention pooling → fixed-size representation → downstream prediction

## Результат главы и переход дальше

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
