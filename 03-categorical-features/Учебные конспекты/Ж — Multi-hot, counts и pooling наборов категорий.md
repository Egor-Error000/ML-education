# Ж. Если одно наблюдение имеет несколько значений field

[Оглавление](README.md). Пункт 7 плана. **Происхождение:** дополнение. Сходство с token embeddings из LLM относится к lookup, но set pooling и sequence encoding задают разные модели.

## 1. Set, multiset или sequence

Список интересов пользователя может быть set: порядок не важен, повтор не добавляет смысла. История покупок — multiset, где counts важны, либо sequence, где важны время и порядок. Выбор representation должен сохранить информацию, нужную target. Называть все три случая «несколько tags» недостаточно.

Multi-hot vector имеет единицу в каждой присутствующей category. Для A,B,A при vocabulary A/B/C он равен $(1,1,0)$. Count representation даёт $(2,1,0)$. Sequence [A,B,A] отличается от [A,A,B], но обе representations его порядок стирают. Если recency критична, нужны temporal features или sequence model.

Для multi-label target тоже бывает multi-hot output, но это другая сторона задачи. Input tags и independent output classes не следует смешивать: softmax target предполагает конкурирующие mutually exclusive classes, sigmoid/BCE может моделировать несколько simultaneous labels.

## 2. Pooling embeddings

Пусть active categories $c_1,\ldots,c_m$ имеют vectors $e_{c_i}$. Базовые варианты:

$$
z_{\mathrm{sum}}=\sum_{i=1}^m e_{c_i},
\qquad
z_{\mathrm{mean}}=\frac1m\sum_{i=1}^m e_{c_i},
\qquad
(z_{\max})_j=\max_i(e_{c_i})_j.
$$

Sum сохраняет влияние числа occurrences, mean нормирует length, max выбирает coordinatewise extremes. Ни один не универсально лучший. Для A=(1,0), B=(0,2), списка A,B,A получаем sum $(2,2)$, mean $(2/3,2/3)$, max $(1,2)$. Если сначала dedup set, sum станет $(1,2)$, mean $(0.5,1)$. Это разные assumptions о значимости repetitions.

Empty set требует explicit vector/indicator. Mean нельзя делить на ноль; max пустого множества не имеет обычного конечного vector. PAD tokens должны исключаться из denominator и max, иначе длина padding изменит representation. Для sum нулевой PAD естественен только при корректной row/update policy.

## 3. Веса и обучаемый pooling

Weighted pooling использует weights, например recency или confidence:

$$
z=\frac{\sum_i w_i e_{c_i}}{\sum_i w_i},\qquad w_i\ge0.
$$

Сумма weights тоже должна быть ненулевой. Attention pooling вычисляет normalized weights по representations и, возможно, query. Если scoring зависит только от каждого set element и aggregation symmetric, result permutation-invariant. Position embeddings или order-dependent preprocessing меняют это свойство.

В mean pooling gradient одного occurrence масштабируется $1/m$; длинный список может ослаблять вклад каждого tag. При sum magnitude растёт с длиной. Добавление отдельного count feature позволяет mean representation явно учитывать size, но не восстанавливает весь потерянный multiset. Max передаёт gradient только selected coordinates, с отдельным поведением на ties.

## 4. Что проверять в data pipeline

Canonicalize spelling/namespace; различать missing list и известный empty list. Ограничение длины/обрезка может терять редкие важные tags; порядок truncation следует определить. Выбор «первые 20» для неупорядоченного set может зависеть от случайной сериализации и давать нестабильные predictions.

Новые tags mapped в UNK/OTHER по сохранённому contract. Если несколько unknown схлопываются, sum может отражать только их count, а mean — одинаковый неизвестный vector. Это нужно проверить отдельно, а не считать semantic generalization.

Для event history preprocessing должен учитывать prediction time: будущая покупка в списке interests — leakage независимо от pooling method. Split по пользователям/времени остаётся основой honest evaluation.

## 5. Контроль и практика

1. Чем multi-hot отличается от counts? **Ответ:** multi-hot фиксирует presence, counts сохраняет multiplicity.
2. Сохраняет ли mean порядок? **Ответ:** нет; sum/mean/max являются symmetric set aggregations.
3. Как кодировать empty и missing? **Ответ:** явно по task contract; они могут иметь разные indicators/vectors и не должны приводить к division by zero.

**Практика:** для A,B,A вычислите три pooling варианта с dedup и без. Переставьте элементы, добавьте PAD и empty case; проверьте ожидаемую invariance. Затем решите, нужен ли вашей задаче set, multiset или sequence, и перечислите информацию, потерянную выбранным encoding.
