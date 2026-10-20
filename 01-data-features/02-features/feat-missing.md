# Missing, unknown, rare и стабильный словарь

**Уровень:** общее ядро.

**Предпосылки:** [Типы признаков и контракты представлений](<feat-types.md>), [Качество данных, пропуски, выбросы и разметка](<../01-reliability/df-quality.md>)

**Используют:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>)

**Учебный вопрос:** Как различать отсутствие значения, новую категорию и редкую известную категорию?





## Карта вычисления и границы этапов

![Missing, unknown, rare и стабильный словарь: входы, преобразования и выходы](<../../оформление/course/feat-missing.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:e5fb851ee6c4015e71 -->
<a id="u-e5fb851ee6c4015e71"></a>

## Различать причины fallback

<!-- /unit:e5fb851ee6c4015e71 -->

<!-- unit:0b6766db5e932b82a4 -->
<a id="u-0b6766db5e932b82a4"></a>

Missing означает, что значение не получено или неизвестно источнику. Unknown — значение получено, но не встречалось в vocabulary fit. Rare — известное значение, по которому мало observations. Empty string может быть валидной category или sentinel; это определяется схемой. Объединять эти состояния можно только как сознательное упрощение.

<!-- /unit:0b6766db5e932b82a4 -->

<!-- unit:bbe32f6fb46df53470 -->
<a id="u-bbe32f6fb46df53470"></a>

Пример: город не указан; город «Новый» появился после training; город «Старый» встречался один раз. Первое может отражать процесс заполнения анкеты, второе — expansion бизнеса, третье — слабую статистику. Один общий marker стирает эти различия и может изменить поведение при drift.

<!-- /unit:bbe32f6fb46df53470 -->

<!-- unit:9aa702b9dc70649403 -->
<a id="u-9aa702b9dc70649403"></a>

## Missing как category

<!-- /unit:9aa702b9dc70649403 -->

<!-- unit:fbe21d0e37b970cb17 -->
<a id="u-fbe21d0e37b970cb17"></a>

Explicit MISSING bucket — хороший baseline для nominal field. Он позволяет learned effect отсутствия, если training содержит такие cases. Для one-hot добавляется column, для embedding — row, для native categorical — canonical value согласно API contract. Если строка `"MISSING"` реально встречается как значение, sentinel должен иметь отдельное безопасное namespace/representation.

<!-- /unit:fbe21d0e37b970cb17 -->

<!-- unit:853671eea9ef6ca4f6 -->
<a id="u-853671eea9ef6ca4f6"></a>

Такой bucket не восстанавливает истинную category. Он моделирует outcome условно на наблюдаемом отсутствии. Если процесс сбора поменялся, прежний missing effect может не переноситься. Поэтому missing rate и quality по этому срезу отслеживаются отдельно.

<!-- /unit:853671eea9ef6ca4f6 -->

<!-- unit:144106f778673712f1 -->
<a id="u-144106f778673712f1"></a>

## Импутация модой и indicator

<!-- /unit:144106f778673712f1 -->

<!-- unit:207d7076423910366f -->
<a id="u-207d7076423910366f"></a>

Mode imputation заменяет null наиболее частой train category. Это создаёт доступный input и простой baseline, но увеличивает observed mass этой category. Даже при MCAR она не восстанавливает индивидуальные истинные values и может искажать dependencies.

<!-- /unit:207d7076423910366f -->

<!-- unit:cc08b7cfec5015afb1 -->
<a id="u-cc08b7cfec5015afb1"></a>

Пусть observed values A,A,B и ещё два missing. Mode fill даёт A,A,B,A,A: A теперь 4 из 5 вместо 2 из 3 observed. Если missing имеют другой target distribution, одна A coefficient смешает genuine A и imputed A. Indicator $M$ помогает различить эти cases:

<!-- /unit:cc08b7cfec5015afb1 -->

<!-- unit:c7da6a78c173ddcf0e -->
<a id="u-c7da6a78c173ddcf0e"></a>

$$
\widehat y=b+w_A\mathbf1[\widetilde C=A]+w_B\mathbf1[\widetilde C=B]+\gamma M.
$$

<!-- /unit:c7da6a78c173ddcf0e -->

<!-- unit:d8064b8166b598ced7 -->
<a id="u-d8064b8166b598ced7"></a>

Но additive indicator не всегда выражает все desired interactions, особенно при нескольких imputed fields. Explicit missing category и mode+indicator не обязаны давать одинаковый hypothesis class при каждой architecture/penalty. Сравнение должно быть task-specific.

<!-- /unit:d8064b8166b598ced7 -->

<!-- unit:b2d567d2cbdcdf2f6a -->
<a id="u-b2d567d2cbdcdf2f6a"></a>

## Vocabulary contract

<!-- /unit:b2d567d2cbdcdf2f6a -->

<!-- unit:585047a81591d7022b -->
<a id="u-585047a81591d7022b"></a>

Canonical type, whitespace/case normalization, namespace, known categories, rare threshold, sentinel и unknown policy сохраняются с model. Train dtype integer и inference strings могут выглядеть одинаково человеку и быть разными categories для system. Чрезмерная normalization также опасна: различия регистра/ведущих нулей иногда значимы для IDs.

<!-- /unit:585047a81591d7022b -->

<!-- unit:565be3bc1c1fd4f655 -->
<a id="u-565be3bc1c1fd4f655"></a>

Rare grouping fit только по train counts. Например, threshold 3 объединяет A с count 2 в OTHER, но не должен использовать будущие counts для решения, станет ли A достаточно частой. Unknown может mapping в OTHER, если group обучена на реальных rare values; это перенос общего поведения rare pool, а не знание нового значения.

<!-- /unit:565be3bc1c1fd4f655 -->

<!-- unit:03c056af8ae0376aef -->
<a id="u-03c056af8ae0376aef"></a>

Код -1 не имеет универсальной semantics. В одном encoder он обозначает unknown, в другом недопустим, в LightGBM categorical treated as missing. В embedding lookup отрицательный index может быть ошибкой API. Нужно проверять end-to-end contract, а не переносить convention по привычке.

<!-- /unit:03c056af8ae0376aef -->

<!-- unit:cc8ed48022f0625294 -->
<a id="u-cc8ed48022f0625294"></a>

## Round-trip и monitoring

<!-- /unit:cc8ed48022f0625294 -->

<!-- unit:015af318a314563512 -->
<a id="u-015af318a314563512"></a>

Проверочные examples: известная category; известная, отсутствующая в текущем inference batch; unseen; missing; rare; строка, похожая на sentinel; случайно изменённый dtype. После serialization/reload predictions и mapping должны сохраняться в рамках объявленной numerical tolerance.

<!-- /unit:015af318a314563512 -->

<!-- unit:920655b49e12207e36 -->
<a id="u-920655b49e12207e36"></a>

Отчёт показывает unknown/missing rates и metric по ним. Если unknown растёт, overall quality может деградировать из-за coverage vocabulary, а не из-за optimizer. Refit dictionary требует согласованного обновления model weights; нельзя изменить IDs одной части pipeline и оставить старую embedding matrix.

<!-- /unit:920655b49e12207e36 -->

<!-- unit:4afe9463a30c8fd07e -->
<a id="u-4afe9463a30c8fd07e"></a>

## Контроль и практика

<!-- /unit:4afe9463a30c8fd07e -->

<!-- unit:37189a8f1fb9e62816 -->
<a id="u-37189a8f1fb9e62816"></a>

1. Что знает model об explicit UNK row без training examples? **Ответ:** обычный supervised signal её effect не выучил; инициализация/fallback остаются непроверенным поведением.
2. Почему mode fill меняет распределение? **Ответ:** все missing искусственно добавляются самой частой category.
3. Можно ли строить rare groups по всем данным? **Ответ:** для inductive evaluation counts fit только по train, иначе используется future distribution.

<!-- /unit:37189a8f1fb9e62816 -->

<!-- unit:9efe013945da64df5d -->
<a id="u-9efe013945da64df5d"></a>

**Практика:** опишите schema для одного field с missing/unknown/rare. Вручную выполните mode fill с indicator и explicit missing representation. Подготовьте семь round-trip cases и объясните expected fallback каждого до запуска model.

<!-- /unit:9efe013945da64df5d -->

<!-- unit:6fcdb6510f803e0645 -->
<a id="u-6fcdb6510f803e0645"></a>

**Схема курса:** raw value + training vocabulary → canonicalize/missing/rare/unknown branch → stable encoded value → serialized schema → inference parity and fallback

<!-- /unit:6fcdb6510f803e0645 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как различать отсутствие значения, новую категорию и редкую известную категорию?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
