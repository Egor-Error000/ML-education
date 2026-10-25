# Nominal, ordinal, one-hot и dummy encoding

**Уровень:** общее ядро.

**Предпосылки:** [Типы признаков и контракты представлений](<feat-types.md>), [Missing, unknown, rare и стабильный словарь](<feat-missing.md>)

**Используют:** [Кардинальность, частотные кодировки и hashing](<feat-hashing.md>), [Target statistics, smoothing, cross-fitting и WoE](<feat-target.md>), [Обучаемые embeddings и перенос представлений](<feat-embeddings.md>), [Нативные категории в бустинге](<feat-native.md>), [Взаимодействия признаков, FM и FFM](<feat-interactions.md>), [Наборы категорий и pooling](<feat-sets.md>), [Отбор признаков и устойчивость выбора](<feat-selection.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>)

**Учебный вопрос:** Как представить категорию, не приписав ей ложный числовой порядок?

## Карта вычисления и границы этапов

![Nominal, ordinal, one-hot и dummy encoding: входы, преобразования и выходы](<../../оформление/course/feat-categorical.svg>)

Числовой код категории не задаёт расстояние между её значениями. Кодировка выбирается под модель; путь неизвестного значения должен быть определён до применения на новых данных.

## Категория, измерение и ID

Город, цвет и ID товара — nominal categories: между кодами нет арифметического расстояния. Размер одежды или уровень образования могут иметь ordinal structure, но даже естественный порядок не гарантирует равных интервалов. Код «малый=1, средний=2, большой=3» допускает ordering, но линейная модель дополнительно предполагает одинаковый эффект соседнего шага.

При nominal encoding A=1,B=2,C=3 линейный predictor $\widehat y=b+wx$ ставит B ровно посередине A/C. Это искусственное ограничение. Decision tree с numeric thresholds может отделить A от B/C или A/B от C, но не A/C от B одним split (разбиение данных). Native categorical split умеет задавать subset; просто integer codes с numeric dtype — другой механизм.

Для target labels (метки) integer IDs допустимы как identifiers classes, если loss (функция потерь)/evaluator интерпретируют их именно так. Не следует смешивать label (метка) encoding target и nominal feature (признак) preprocessing.

### Сначала определить допустимые операции

Тип хранения не определяет смысл признака. Строка «42» может быть номером маршрута, возрастом или кодом товара; integer 42 тоже не сообщает, можно ли прибавить единицу и получить осмысленное соседнее значение. Для nominal категории допустима проверка равенства, но не требуется порядок. Для ordinal есть порядок, однако интервалы между уровнями не обязаны быть равными. Для количественной величины arithmetic и units имеют содержательный смысл.

Это свойство постановки, а не только dtype. Например, размер одежды имеет порядок, но переход S→M не обязан влиять на target так же, как M→L. Номер района не становится расстоянием от центра. Почтовый индекс может содержать структуру, но её надо явно извлечь и проверить, а не считать, что обычная арифметика всех цифр автоматически её использует.

Target classes тоже имеют IDs, но loss интерпретирует их как адреса классов, а не значения числовой регрессии. Если nominal feature передать линейной модели как число, она получает совсем другой контракт. Поэтому encoder (кодировщик входа) features (признаки) и mapping target labels сохраняют отдельно.

### Полностью вычисленный провал произвольного порядка

Возьмём по одному наблюдению A/B/C с outcomes 10,13,8 и numeric codes 1,2,3. Predictor b+wc обязан иметь одинаковую разность между соседними кодами, значит prediction B всегда является средним prediction A и C. Для истинных outcomes это неверно: среднее A/C равно 9, а B=13.

OLS по трём точкам даёт w=−1, b=37/3. Predictions равны 34/3,31/3,28/3, то есть примерно 11.333,10.333,9.333; MSE=32/9. Full one-hot без intercept может точно задать три coefficients 10,13,8. Разница здесь вызвана доступным классом функций, а не тем, что один набор IDs «красивее» другого.

Перестановка nominal IDs меняет соседство и fit numeric модели, хотя task не изменилась. Для one-hot перестановка category columns с такой же перестановкой coefficients сохраняет predictions. Это полезная проверка того, является ли arbitrary ordering частью нежелательного prior.

### Настоящий порядок не равен линейному эффекту

Для уровней S<M<L можно кодировать cumulative indicators $I[c\ge M]$ и $I[c\ge L]$. Векторы S/M/L равны (0,0), (1,0), (1,1). Если b=10 и increments равны 1 и 9, predictions получаются 10,11,20. Numeric code с единственным slope не мог бы выразить эти неравные increments точно.

Если требуется монотонный рост response, increments можно ограничить неотрицательностью. Без такого ограничения cumulative coding само по себе не запрещает убывающие участки. Порядок входных категорий также не означает, что target обязательно монотонен по ним: это отдельное предположение задачи.

У numeric decision tree thresholds используют порядок, но не обязаны давать одинаковый effect каждого шага. Однако для nominal A/B/C/D с кодами 1/2/3/4 один threshold не отделит subset {A,C} от {B,D}. Native categorical partition способен задать такой subset непосредственно. One-hot splits и несколько numeric thresholds могут выразить похожую функцию с другой структурой и стоимостью дерева; это не универсальная гарантия преимущества одного варианта.

## One-hot и геометрия

Для $K$ categories значение $c$ кодируется vector $e_c\in\{0,1\}^K$. Например, A/B/C дают $(1,0,0),(0,1,0),(0,0,1)$. Линейный model:

$$
\widehat y=b+w^\top e_c=b+w_c.
$$

Теперь каждый class имеет отдельный effect. Euclidean distance между любыми двумя различными one-hot vectors равна $\sqrt{2}$. Representation не содержит semantic closeness, но downstream model может выучить разные effects и interactions; это не запрет на learning сходства вообще.

Категории fit на train. New category требует policy: error, explicit unknown bucket, all-zero либо иной предусмотренный fallback. Missing и unknown имеют разный смысл: missing — значение не получено, unknown — получено новое значение вне train vocabulary. Если они кодируются одинаково, это сознательная потеря информации.

### Почему расстояние $\sqrt{2}$ не является знанием о смысле

Для двух разных basis vectors $\|e_a-e_b\|^2=1+1=2$. Это означает, что исходное representation не различает степени семантической близости. Линейная модель может выучить похожие coefficients для двух категорий, embedding (векторное представление) — близкие vectors, а взаимодействия — зависимость от других fields. Равная исходная дистанция не запрещает такое обучение.

Для нескольких single-valued fields при простом concatenation каждая несовпадающая категория добавляет 2 к squared Euclidean distance. Один field с тысячей уровней не получает автоматически в тысячу раз больший вклад, чем field с двумя уровнями: у обоих активен один basis coordinate. Веса fields, scaling и multi-valued representation могут изменить это соотношение.

### Dummy coding меняет расстояния

При baseline (базовый вариант сравнения) A и оставшихся columns B/C vectors равны A=(0,0), B=(1,0), C=(0,1). Теперь расстояния A–B и A–C равны 1, а B–C — $\sqrt{2}$. Это нормально как определённый contrast для линейного predictor, но уже другая геометрия для nearest neighbors, clustering или distance kernel. Удаление column не является универсально нейтральным preprocessing для любого estimator.

Если binary indicators дополнительно делят на их standard deviations, редкие levels могут получить большой вес. Для категорий с probabilities 0.5 и 0.01 squared distance после такого scaling содержит $1/[0.5(1-0.5)]+1/[0.01(1-0.01)]\approx105.01$ вместо 2. Это прямой результат выбранных units representation, а не автоматически более правильное расстояние.

Центрирование разреженной матрицы также может сделать почти все нулевые элементы ненулевыми. Поэтому numeric scaling, geometry и sparse storage проектируют совместно. «Все признаки надо привести к variance 1» не является безусловным правилом категориального pipeline.

## Intercept и идентифицируемость

При полном one-hot каждая строка удовлетворяет $\sum_{k=1}^K e_{c,k}=1$. Столбец intercept линейно зависим от dummy columns. Для unregularized linear least squares coefficients неидентифицируемы: к intercept можно добавить $a$, ко всем category coefficients вычесть $a$, predictions не изменятся.

Dummy encoding удаляет одну baseline category. Для baseline A:

$$
\widehat y=b+\beta_B\mathbf1[c=B]+\beta_C\mathbf1[c=C].
$$

Тогда $b$ — effect A, $\beta_B$ и $\beta_C$ — differences от A. Если средние outcomes A/B/C равны 10/13/8, coefficients составляют $b=10,\beta_B=3,\beta_C=-2$. Существуют и другие contrast codings; baseline выбран для интерпретации, а не потому, что A важнее.

Удалять column обязательно не для каждого estimator. Regularized model может иметь уникальное penalized solution и с full one-hot; trees не требуют full-rank design matrix. При penalty choice baseline может менять symmetry regularization, поэтому drop-first не универсально нейтрален.

### Почему coefficients могут быть неоднозначными

Для A/B/C design с intercept содержит строки `(1,1,0,0)`, `(1,0,1,0)`, `(1,0,0,1)`. Ранг равен 3 при четырёх columns, поскольку intercept равен сумме category indicators. Coefficients b=0,w=(10,13,8) и b=10,w=(0,3,−2) дают одинаковые predictions. Сдвиг b на a и всех wk на −a сохраняет каждую строку.

Без intercept full one-hot из трёх наблюдавшихся уровней имеет rank 3, и category coefficients в этой игрушечной задаче определяются однозначно. Если одновременно удалить baseline-column и не добавить intercept, baseline получит prediction 0 в модели только с этим field. Поэтому «всегда удалять первый столбец» без учёта estimator может испортить постановку.

Некоторые penalties устраняют неоднозначность полного design, выбирая предпочтительное представление. Например, ridge с unpenalized intercept может иметь единственное решение, хотя unregularized design был зависим. Trees вообще не решают normal equations этой матрицы. Требование full rank нельзя механически переносить на все модели. Доступные encoder options описаны в [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

### Одна функция и разные штрафы baseline

Зафиксируем predictions категорий 10,13,8. С baseline A получаем b=10, beta_B=3, beta_C=−2 и сумму squared coefficients 13, если intercept не штрафуется. С baseline C получаем b=8, beta_A=2, beta_B=5 и сумму квадратов 29. Predictions идентичны, а penalty — разная.

В полном one-hot среди представлений этих же predictions минимальная сумма squared category coefficients достигается при b=31/3, w=(−1/3,8/3,−7/3), сумма=38/3. Здесь рассматривается выбор параметризации уже фиксированной функции; конечный penalized fit может изменить и predictions. Пример показывает, почему drop-first при регуляризации не всегда симметричен относительно категорий.

### Смысл coefficient зависит от link и других features

В модели только с category и identity link baseline b можно читать как средний response A. С числовыми covariates это response при reference values остальных features, а с interactions эффект category может зависеть от контекста. Нельзя автоматически называть каждый coefficient безусловным средним всей группы.

В logistic regression b — baseline log-odds, а beta_B — разность log-odds. Если p_A=0.2 и p_B=0.5, b=−log4, beta_B=log4; odds ratio равен 4. Это не увеличение probability на четыре единицы и не гарантированный causal effect. Кодирование помогает представлению ассоциации, но не заменяет постановку причинного вопроса.

## Missing, unknown и словарь

Если baseline A закодирован всеми нулями и unseen D тоже получает zero vector, model будет трактовать D как A. Это допустимый fallback только при осознанном решении, а не следствие того, что модель «поняла новый город». Explicit unknown bucket позволяет отдельное поведение, но если он ни разу не встречался в training, его coefficient/embedding не выучен обычными данными.

Rare grouping объединяет малочисленные categories в OTHER по train counts. Это уменьшает variance, но стирает различия; threshold выбирается на validation (проверочная часть для выбора решения). Нельзя построить словарь по train+test лишь потому, что labels не используются: для inductive evaluation distribution тоже должна быть доступна только по protocol.

Feature names, category ordering и dtype сохраняются вместе с model. Если при inference (применение обученной модели) B получил index A, prediction ошибочен, хотя shapes совпадают. Serialization round-trip должен включать encoder, а не только final estimator.

### Пять состояний одного field

Учебный train vocabulary содержит A/B/C. Можно дополнительно зарезервировать MISSING и UNKNOWN, получив пять отдельных basis coordinates. A/B/C — известные значения; MISSING — значение отсутствует; D на inference — значение есть, но его нет в vocabulary, поэтому используется UNKNOWN. Если literal string «MISSING» может быть настоящим значением, нужен недвусмысленный способ отличать её от служебного состояния.

Можно выбрать и другой contract: ошибка на unseen, all-zero, либо объединение с обученной rare-группой. При full one-hot all-zero означает отсутствие category contribution и оставляет intercept плюс другие features, а не автоматически population-average prediction. При drop-first тот же zero совпадает с baseline A. Это два разных fallback при внешне одинаковом векторе.

Reserved UNKNOWN-column позволяет отдельно хранить состояние, но её coefficient не выучен из обычных labels, если column никогда не активировалась в train. Regularization или initialization зададут некоторое поведение, но не знание о новом значении D. Можно обучать объявленный fallback на редких/искусственно скрытых категориях, проверяя, насколько этот режим соответствует реальному unseen; подробный контракт рассматривается в главе [«Missing, unknown, rare и стабильный словарь»](<feat-missing.md>).

### Где проходит граница fit

Если vocabulary и rare threshold выводятся из текущих данных, их fit находится внутри train каждого fold. Отсутствие использования y не даёт права читать будущую popularity или состав закрытого test в индуктивном эксперименте. В то же время заранее заданный внешний справочник допустимых категорий — иной источник знания: его можно зафиксировать независимо от текущего sample и описать как schema (схема допустимых данных)/prior.

Для ordinal features порядок уровней лучше задавать содержательно, а не доверять alphabetic sort. Unknown и missing codes должны быть отличимы от нормальных levels, если такой контракт выбран. Numeric sentinel −1, переданный как обычный ordinal feature, также получает геометрический смысл для модели; он не становится «ничем» автоматически. Точные options проверяют в [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

### Что сохраняется с моделью

Сохраняют имена и порядок fields, category ordering, dtype/canonicalization, rare threshold, baseline, special states, feature names и final estimator. Inference не должен заново сортировать только встретившиеся в новом batch (пакет примеров) значения. Иначе B может незаметно получить column A при прежнем shape, и model weights будут применены к другому смыслу.

Round-trip проверяют на известных categories, missing, unseen, пустых строках и значениях разных типов, если они разрешены схемой. Inverse transform all-zero может возвращать baseline или unknown-marker в зависимости от configuration; восстановление строки не является универсально однозначным. Важнее соответствие сохранённому контракту, чем отсутствие ошибки сериализации.

## Контроль и практика

1. Почему arbitrary integer codes вредят linear model? **Ответ:** вводят порядок и равные расстояния, которых nominal feature не имеет.
2. Что устраняет drop-first? **Ответ:** зависимость dummy columns с intercept в полной линейной design matrix.
3. Почему unknown нельзя молча приравнивать missing? **Ответ:** это разные процессы; объединение теряет возможный сигнал.

**Практика:** закодируйте пять строк с A/B/C, missing и unseen D. Для full/dummy схем выпишите predictions и fallback. Покажите две разные группы coefficients, дающие одни predictions при intercept+full one-hot. Сохраните vocabulary и проверьте восстановление column order.

**Схема курса:** raw category + declared semantics → nominal/ordinal decision → fitted vocabulary → one-hot/dummy columns → unknown policy → model input



1. **Почему естественный порядок не задаёт равных intervals?** Он сообщает сравнение levels, но не величину эффекта каждого перехода.
2. **Когда dummy coding меняет не только интерпретацию?** При distance-based модели и при penalties, зависящих от выбранных coordinates.
3. **Почему all-zero unknown не равен «модель поняла новый класс»?** Это фиксированный fallback без обученного индивидуального effect.
4. **Почему одинаковый output shape не доказывает правильный encoder?** Значения и columns могли поменять смысл при другом mapping.

Практика: воспроизвести numeric OLS, full/dummy predictions и два penalty values при разных baselines. Для пяти состояний field выписать точные vectors и fallback. Сохранить encoder с моделью и проверить поведение после восстановления. Все строки и coefficients главы учебные; реальный пользовательский dataset не перекодировался.

## Результат главы и переход дальше

Следующие темы: [Кардинальность, частотные кодировки и hashing](<feat-hashing.md>), [Target statistics, smoothing, cross-fitting и WoE](<feat-target.md>), [Обучаемые embeddings и перенос представлений](<feat-embeddings.md>), [Нативные категории в бустинге](<feat-native.md>), [Взаимодействия признаков, FM и FFM](<feat-interactions.md>), [Наборы категорий и pooling](<feat-sets.md>), [Отбор признаков и устойчивость выбора](<feat-selection.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
