# З. Отсутствующая категория — отдельное состояние данных

[Оглавление](README.md). Пункт 8 плана и inference edge cases. **Происхождение:** дополнение; общая теория missingness находится в [данных](../../01-data/Учебные%20конспекты/README.md).

**Уровень:** основа ветки для missing/unknown handling. **Предпосылки:** D3, F1–F3. **Используют:** D3, F3/F7 и production vocabulary/fallback monitoring.

## 1. Различать причины fallback

Missing означает, что значение не получено или неизвестно источнику. Unknown — значение получено, но не встречалось в vocabulary fit. Rare — известное значение, по которому мало observations. Empty string может быть валидной category или sentinel; это определяется схемой. Объединять эти состояния можно только как сознательное упрощение.

Пример: город не указан; город «Новый» появился после training; город «Старый» встречался один раз. Первое может отражать процесс заполнения анкеты, второе — expansion бизнеса, третье — слабую статистику. Один общий marker стирает эти различия и может изменить поведение при drift.

## 2. Missing как category

Explicit MISSING bucket — хороший baseline для nominal field. Он позволяет learned effect отсутствия, если training содержит такие cases. Для one-hot добавляется column, для embedding — row, для native categorical — canonical value согласно API contract. Если строка `"MISSING"` реально встречается как значение, sentinel должен иметь отдельное безопасное namespace/representation.

Такой bucket не восстанавливает истинную category. Он моделирует outcome условно на наблюдаемом отсутствии. Если процесс сбора поменялся, прежний missing effect может не переноситься. Поэтому missing rate и quality по этому срезу отслеживаются отдельно.

## 3. Импутация модой и indicator

Mode imputation заменяет null наиболее частой train category. Это создаёт доступный input и простой baseline, но увеличивает observed mass этой category. Даже при MCAR она не восстанавливает индивидуальные истинные values и может искажать dependencies.

Пусть observed values A,A,B и ещё два missing. Mode fill даёт A,A,B,A,A: A теперь 4 из 5 вместо 2 из 3 observed. Если missing имеют другой target distribution, одна A coefficient смешает genuine A и imputed A. Indicator $M$ помогает различить эти cases:

$$
\widehat y=b+w_A\mathbf1[\widetilde C=A]+w_B\mathbf1[\widetilde C=B]+\gamma M.
$$

Но additive indicator не всегда выражает все desired interactions, особенно при нескольких imputed fields. Explicit missing category и mode+indicator не обязаны давать одинаковый hypothesis class при каждой architecture/penalty. Сравнение должно быть task-specific.

## 4. Vocabulary contract

Canonical type, whitespace/case normalization, namespace, known categories, rare threshold, sentinel и unknown policy сохраняются с model. Train dtype integer и inference strings могут выглядеть одинаково человеку и быть разными categories для system. Чрезмерная normalization также опасна: различия регистра/ведущих нулей иногда значимы для IDs.

Rare grouping fit только по train counts. Например, threshold 3 объединяет A с count 2 в OTHER, но не должен использовать будущие counts для решения, станет ли A достаточно частой. Unknown может mapping в OTHER, если group обучена на реальных rare values; это перенос общего поведения rare pool, а не знание нового значения.

Код -1 не имеет универсальной semantics. В одном encoder он обозначает unknown, в другом недопустим, в LightGBM categorical treated as missing. В embedding lookup отрицательный index может быть ошибкой API. Нужно проверять end-to-end contract, а не переносить convention по привычке.

## 5. Round-trip и monitoring

Проверочные examples: известная category; известная, отсутствующая в текущем inference batch; unseen; missing; rare; строка, похожая на sentinel; случайно изменённый dtype. После serialization/reload predictions и mapping должны сохраняться в рамках объявленной numerical tolerance.

Отчёт показывает unknown/missing rates и metric по ним. Если unknown растёт, overall quality может деградировать из-за coverage vocabulary, а не из-за optimizer. Refit dictionary требует согласованного обновления model weights; нельзя изменить IDs одной части pipeline и оставить старую embedding matrix.

## 6. Контроль и практика

1. Что знает model об explicit UNK row без training examples? **Ответ:** обычный supervised signal её effect не выучил; инициализация/fallback остаются непроверенным поведением.
2. Почему mode fill меняет распределение? **Ответ:** все missing искусственно добавляются самой частой category.
3. Можно ли строить rare groups по всем данным? **Ответ:** для inductive evaluation counts fit только по train, иначе используется future distribution.

**Практика:** опишите schema для одного field с missing/unknown/rare. Вручную выполните mode fill с indicator и explicit missing representation. Подготовьте семь round-trip cases и объясните expected fallback каждого до запуска model.


**Схема курса:** raw value + training vocabulary → canonicalize/missing/rare/unknown branch → stable encoded value → serialized schema → inference parity and fallback

