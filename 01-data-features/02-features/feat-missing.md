# Missing, unknown, rare и стабильный словарь

**Уровень:** общее ядро.

**Предпосылки:** [Типы признаков и контракты представлений](<feat-types.md>), [Качество данных, пропуски, выбросы и разметка](<../01-reliability/df-quality.md>)

**Используют:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>)

**Учебный вопрос:** Как различать отсутствие значения, новую категорию и редкую известную категорию?

## Карта вычисления и границы этапов

![Missing, unknown, rare и стабильный словарь: входы, преобразования и выходы](<../../оформление/course/feat-missing.svg>)

Отсутствующее значение и ранее не встречавшаяся категория — разные состояния. Индикатор сохраняет информацию о пропуске, которую одно заполненное число иначе уничтожило бы.

## Различать причины fallback

Missing означает, что значение не получено или неизвестно источнику. Unknown — значение получено, но не встречалось в vocabulary fit. Rare — известное значение, по которому мало observations. Empty string может быть валидной category или sentinel; это определяется схемой. Объединять эти состояния можно только как сознательное упрощение.

Пример: город не указан; город «Новый» появился после training; город «Старый» встречался один раз. Первое может отражать процесс заполнения анкеты, второе — expansion бизнеса, третье — слабую статистику. Один общий marker стирает эти различия и может изменить поведение при drift (изменение распределения или зависимости со временем).

## Missing как category

Explicit MISSING bucket — хороший baseline (базовый вариант сравнения) для nominal field. Он позволяет learned effect отсутствия, если training содержит такие cases. Для one-hot добавляется column, для embedding (векторное представление) — row, для native categorical — canonical value согласно API contract. Если строка `"MISSING"` реально встречается как значение, sentinel должен иметь отдельное безопасное namespace/representation.

Такой bucket не восстанавливает истинную category. Он моделирует outcome условно на наблюдаемом отсутствии. Если процесс сбора поменялся, прежний missing effect может не переноситься. Поэтому missing rate и quality по этому срезу отслеживаются отдельно.

## Импутация модой и indicator

Mode imputation заменяет null наиболее частой train category. Это создаёт доступный input и простой baseline, но увеличивает observed mass этой category. Даже при MCAR она не восстанавливает индивидуальные истинные values и может искажать dependencies.

Пусть observed values A,A,B и ещё два missing. Mode fill даёт A,A,B,A,A: A теперь 4 из 5 вместо 2 из 3 observed. Если missing имеют другой target distribution, одна A coefficient смешает genuine A и imputed A. Indicator $M$ помогает различить эти cases:

$$
\widehat y=b+w_A\mathbf1[\widetilde C=A]+w_B\mathbf1[\widetilde C=B]+\gamma M.
$$

Но additive indicator не всегда выражает все desired interactions, особенно при нескольких imputed fields. Explicit missing category и mode+indicator не обязаны давать одинаковый hypothesis class при каждой architecture/penalty. Сравнение должно быть task-specific.

## Vocabulary contract

Canonical type, whitespace/case normalization (нормировка по выбранному правилу), namespace, known categories, rare threshold, sentinel и unknown policy сохраняются с model. Train dtype integer и inference (применение обученной модели) strings могут выглядеть одинаково человеку и быть разными categories для system. Чрезмерная normalization также опасна: различия регистра/ведущих нулей иногда значимы для IDs.

Rare grouping fit только по train counts. Например, threshold 3 объединяет A с count 2 в OTHER, но не должен использовать будущие counts для решения, станет ли A достаточно частой. Unknown может mapping в OTHER, если group обучена на реальных rare values; это перенос общего поведения rare pool, а не знание нового значения.

Код -1 не имеет универсальной semantics. В одном encoder (кодировщик входа) он обозначает unknown, в другом недопустим, в LightGBM categorical treated as missing. В embedding lookup отрицательный index может быть ошибкой API. Нужно проверять end-to-end contract, а не переносить convention по привычке.

## Round-trip и monitoring

Проверочные examples: известная category; известная, отсутствующая в текущем inference batch (пакет примеров); unseen; missing; rare; строка, похожая на sentinel; случайно изменённый dtype. После serialization/reload predictions и mapping должны сохраняться в рамках объявленной numerical tolerance.

Отчёт показывает unknown/missing rates и metric по ним. Если unknown растёт, overall quality может деградировать из-за coverage vocabulary, а не из-за optimizer (оптимизатор обновления параметров). Refit dictionary требует согласованного обновления model weights; нельзя изменить IDs одной части pipeline и оставить старую embedding matrix.

## Контроль и практика

1. Что знает model об explicit UNK row без training examples? **Ответ:** обычный supervised signal её effect не выучил; инициализация/fallback остаются непроверенным поведением.
2. Почему mode fill меняет распределение? **Ответ:** все missing искусственно добавляются самой частой category.
3. Можно ли строить rare groups по всем данным? **Ответ:** для inductive evaluation counts fit только по train, иначе используется future distribution.

**Практика:** опишите schema (схема допустимых данных) для одного field с missing/unknown/rare. Вручную выполните mode fill с indicator и explicit missing representation. Подготовьте семь round-trip cases и объясните expected fallback каждого до запуска model.

**Схема курса:** raw value + training vocabulary → canonicalize/missing/rare/unknown branch → stable encoded value → serialized schema → inference parity and fallback

## Результат главы и переход дальше

Следующие темы: [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
