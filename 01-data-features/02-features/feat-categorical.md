# Nominal, ordinal, one-hot и dummy encoding

**Уровень:** общее ядро.

**Предпосылки:** [Типы признаков и контракты представлений](<feat-types.md>), [Missing, unknown, rare и стабильный словарь](<feat-missing.md>)

**Используют:** [Кардинальность, частотные кодировки и hashing](<feat-hashing.md>), [Target statistics, smoothing, cross-fitting и WoE](<feat-target.md>), [Обучаемые embeddings и перенос представлений](<feat-embeddings.md>), [Нативные категории в бустинге](<feat-native.md>), [Взаимодействия признаков, FM и FFM](<feat-interactions.md>), [Наборы категорий и pooling](<feat-sets.md>), [Отбор признаков и устойчивость выбора](<feat-selection.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>)

**Учебный вопрос:** Как представить категорию, не приписав ей ложный числовой порядок?





## Карта вычисления и границы этапов

![Nominal, ordinal, one-hot и dummy encoding: входы, преобразования и выходы](<../../оформление/course/feat-categorical.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:b6427bd5c9892b2a3e -->
<a id="u-b6427bd5c9892b2a3e"></a>

## Категория, измерение и ID

<!-- /unit:b6427bd5c9892b2a3e -->

<!-- unit:034b21630ca78c0cad -->
<a id="u-034b21630ca78c0cad"></a>

Город, цвет и ID товара — nominal categories: между кодами нет арифметического расстояния. Размер одежды или уровень образования могут иметь ordinal structure, но даже естественный порядок не гарантирует равных интервалов. Код «малый=1, средний=2, большой=3» допускает ordering, но линейная модель дополнительно предполагает одинаковый эффект соседнего шага.

<!-- /unit:034b21630ca78c0cad -->

<!-- unit:c82c5dca576b9209d4 -->
<a id="u-c82c5dca576b9209d4"></a>

При nominal encoding A=1,B=2,C=3 линейный predictor $\widehat y=b+wx$ ставит B ровно посередине A/C. Это искусственное ограничение. Decision tree с numeric thresholds может отделить A от B/C или A/B от C, но не A/C от B одним split. Native categorical split умеет задавать subset; просто integer codes с numeric dtype — другой механизм.

<!-- /unit:c82c5dca576b9209d4 -->

<!-- unit:2b55983bbfbb5eedf5 -->
<a id="u-2b55983bbfbb5eedf5"></a>

Для target labels integer IDs допустимы как identifiers classes, если loss/evaluator интерпретируют их именно так. Не следует смешивать label encoding target и nominal feature preprocessing.

<!-- /unit:2b55983bbfbb5eedf5 -->

<!-- unit:b0037493c416abca92 -->
<a id="u-b0037493c416abca92"></a>

### Сначала определить допустимые операции

<!-- /unit:b0037493c416abca92 -->

<!-- unit:e25eb27cd155040829 -->
<a id="u-e25eb27cd155040829"></a>

Тип хранения не определяет смысл признака. Строка «42» может быть номером маршрута, возрастом или кодом товара; integer 42 тоже не сообщает, можно ли прибавить единицу и получить осмысленное соседнее значение. Для nominal категории допустима проверка равенства, но не требуется порядок. Для ordinal есть порядок, однако интервалы между уровнями не обязаны быть равными. Для количественной величины arithmetic и units имеют содержательный смысл.

<!-- /unit:e25eb27cd155040829 -->

<!-- unit:d30f6d490b53829dda -->
<a id="u-d30f6d490b53829dda"></a>

Это свойство постановки, а не только dtype. Например, размер одежды имеет порядок, но переход S→M не обязан влиять на target так же, как M→L. Номер района не становится расстоянием от центра. Почтовый индекс может содержать структуру, но её надо явно извлечь и проверить, а не считать, что обычная арифметика всех цифр автоматически её использует.

<!-- /unit:d30f6d490b53829dda -->

<!-- unit:f2360bd1bad1325950 -->
<a id="u-f2360bd1bad1325950"></a>

Target classes тоже имеют IDs, но loss интерпретирует их как адреса классов, а не значения числовой регрессии. Если nominal feature передать линейной модели как число, она получает совсем другой контракт. Поэтому encoder features и mapping target labels сохраняют отдельно.

<!-- /unit:f2360bd1bad1325950 -->

<!-- unit:81a9b8ab8339250be2 -->
<a id="u-81a9b8ab8339250be2"></a>

### Полностью вычисленный провал произвольного порядка

<!-- /unit:81a9b8ab8339250be2 -->

<!-- unit:1d998c4bc40882434b -->
<a id="u-1d998c4bc40882434b"></a>

Возьмём по одному наблюдению A/B/C с outcomes 10,13,8 и numeric codes 1,2,3. Predictor b+wc обязан иметь одинаковую разность между соседними кодами, значит prediction B всегда является средним prediction A и C. Для истинных outcomes это неверно: среднее A/C равно 9, а B=13.

<!-- /unit:1d998c4bc40882434b -->

<!-- unit:a623cacd59b4d0d0f5 -->
<a id="u-a623cacd59b4d0d0f5"></a>

OLS по трём точкам даёт w=−1, b=37/3. Predictions равны 34/3,31/3,28/3, то есть примерно 11.333,10.333,9.333; MSE=32/9. Full one-hot без intercept может точно задать три coefficients 10,13,8. Разница здесь вызвана доступным классом функций, а не тем, что один набор IDs «красивее» другого.

<!-- /unit:a623cacd59b4d0d0f5 -->

<!-- unit:168f48414b005d99f9 -->
<a id="u-168f48414b005d99f9"></a>

Перестановка nominal IDs меняет соседство и fit numeric модели, хотя task не изменилась. Для one-hot перестановка category columns с такой же перестановкой coefficients сохраняет predictions. Это полезная проверка того, является ли arbitrary ordering частью нежелательного prior.

<!-- /unit:168f48414b005d99f9 -->

<!-- unit:966ad1c2635b92bd7e -->
<a id="u-966ad1c2635b92bd7e"></a>

### Настоящий порядок не равен линейному эффекту

<!-- /unit:966ad1c2635b92bd7e -->

<!-- unit:cbaf99802f8e1ffc94 -->
<a id="u-cbaf99802f8e1ffc94"></a>

Для уровней S<M<L можно кодировать cumulative indicators $I[c\ge M]$ и $I[c\ge L]$. Векторы S/M/L равны (0,0), (1,0), (1,1). Если b=10 и increments равны 1 и 9, predictions получаются 10,11,20. Numeric code с единственным slope не мог бы выразить эти неравные increments точно.

<!-- /unit:cbaf99802f8e1ffc94 -->

<!-- unit:9f81fcc8959940b7e7 -->
<a id="u-9f81fcc8959940b7e7"></a>

Если требуется монотонный рост response, increments можно ограничить неотрицательностью. Без такого ограничения cumulative coding само по себе не запрещает убывающие участки. Порядок входных категорий также не означает, что target обязательно монотонен по ним: это отдельное предположение задачи.

<!-- /unit:9f81fcc8959940b7e7 -->

<!-- unit:1b2a76ef0ff8d83898 -->
<a id="u-1b2a76ef0ff8d83898"></a>

У numeric decision tree thresholds используют порядок, но не обязаны давать одинаковый effect каждого шага. Однако для nominal A/B/C/D с кодами 1/2/3/4 один threshold не отделит subset {A,C} от {B,D}. Native categorical partition способен задать такой subset непосредственно. One-hot splits и несколько numeric thresholds могут выразить похожую функцию с другой структурой и стоимостью дерева; это не универсальная гарантия преимущества одного варианта.

<!-- /unit:1b2a76ef0ff8d83898 -->

<!-- unit:d13a440c20718a0085 -->
<a id="u-d13a440c20718a0085"></a>

## One-hot и геометрия

<!-- /unit:d13a440c20718a0085 -->

<!-- unit:ea7fb6d2488b9aead3 -->
<a id="u-ea7fb6d2488b9aead3"></a>

Для $K$ categories значение $c$ кодируется vector $e_c\in\{0,1\}^K$. Например, A/B/C дают $(1,0,0),(0,1,0),(0,0,1)$. Линейный model:

<!-- /unit:ea7fb6d2488b9aead3 -->

<!-- unit:ccf17a6901b896fac3 -->
<a id="u-ccf17a6901b896fac3"></a>

$$
\widehat y=b+w^\top e_c=b+w_c.
$$

<!-- /unit:ccf17a6901b896fac3 -->

<!-- unit:9d131d9b05cd4c3623 -->
<a id="u-9d131d9b05cd4c3623"></a>

Теперь каждый class имеет отдельный effect. Euclidean distance между любыми двумя различными one-hot vectors равна $\sqrt{2}$. Representation не содержит semantic closeness, но downstream model может выучить разные effects и interactions; это не запрет на learning сходства вообще.

<!-- /unit:9d131d9b05cd4c3623 -->

<!-- unit:b53e9c79ad2c7124f8 -->
<a id="u-b53e9c79ad2c7124f8"></a>

Категории fit на train. New category требует policy: error, explicit unknown bucket, all-zero либо иной предусмотренный fallback. Missing и unknown имеют разный смысл: missing — значение не получено, unknown — получено новое значение вне train vocabulary. Если они кодируются одинаково, это сознательная потеря информации.

<!-- /unit:b53e9c79ad2c7124f8 -->

<!-- unit:a3b412731c02493e7c -->
<a id="u-a3b412731c02493e7c"></a>

### Почему расстояние $\sqrt{2}$ не является знанием о смысле

<!-- /unit:a3b412731c02493e7c -->

<!-- unit:d3dad71f531f4973da -->
<a id="u-d3dad71f531f4973da"></a>

Для двух разных basis vectors $\|e_a-e_b\|^2=1+1=2$. Это означает, что исходное representation не различает степени семантической близости. Линейная модель может выучить похожие coefficients для двух категорий, embedding — близкие vectors, а взаимодействия — зависимость от других fields. Равная исходная дистанция не запрещает такое обучение.

<!-- /unit:d3dad71f531f4973da -->

<!-- unit:0db9c688d4cbccc6d2 -->
<a id="u-0db9c688d4cbccc6d2"></a>

Для нескольких single-valued fields при простом concatenation каждая несовпадающая категория добавляет 2 к squared Euclidean distance. Один field с тысячей уровней не получает автоматически в тысячу раз больший вклад, чем field с двумя уровнями: у обоих активен один basis coordinate. Веса fields, scaling и multi-valued representation могут изменить это соотношение.

<!-- /unit:0db9c688d4cbccc6d2 -->

<!-- unit:091b5e0581050c92e8 -->
<a id="u-091b5e0581050c92e8"></a>

### Dummy coding меняет расстояния

<!-- /unit:091b5e0581050c92e8 -->

<!-- unit:596c8fa7fba3fef8c7 -->
<a id="u-596c8fa7fba3fef8c7"></a>

При baseline A и оставшихся columns B/C vectors равны A=(0,0), B=(1,0), C=(0,1). Теперь расстояния A–B и A–C равны 1, а B–C — $\sqrt{2}$. Это нормально как определённый contrast для линейного predictor, но уже другая геометрия для nearest neighbors, clustering или distance kernel. Удаление column не является универсально нейтральным preprocessing для любого estimator.

<!-- /unit:596c8fa7fba3fef8c7 -->

<!-- unit:a128e28756cf401916 -->
<a id="u-a128e28756cf401916"></a>

Если binary indicators дополнительно делят на их standard deviations, редкие levels могут получить большой вес. Для категорий с probabilities 0.5 и 0.01 squared distance после такого scaling содержит $1/[0.5(1-0.5)]+1/[0.01(1-0.01)]\approx105.01$ вместо 2. Это прямой результат выбранных units representation, а не автоматически более правильное расстояние.

<!-- /unit:a128e28756cf401916 -->

<!-- unit:a47c63b4019c601815 -->
<a id="u-a47c63b4019c601815"></a>

Центрирование разреженной матрицы также может сделать почти все нулевые элементы ненулевыми. Поэтому numeric scaling, geometry и sparse storage проектируют совместно. «Все признаки надо привести к variance 1» не является безусловным правилом категориального pipeline.

<!-- /unit:a47c63b4019c601815 -->

<!-- unit:c8d0d17ebc3c3b413c -->
<a id="u-c8d0d17ebc3c3b413c"></a>

## Intercept и идентифицируемость

<!-- /unit:c8d0d17ebc3c3b413c -->

<!-- unit:11ffffdc7230290cd0 -->
<a id="u-11ffffdc7230290cd0"></a>

При полном one-hot каждая строка удовлетворяет $\sum_{k=1}^K e_{c,k}=1$. Столбец intercept линейно зависим от dummy columns. Для unregularized linear least squares coefficients неидентифицируемы: к intercept можно добавить $a$, ко всем category coefficients вычесть $a$, predictions не изменятся.

<!-- /unit:11ffffdc7230290cd0 -->

<!-- unit:8b559d50ca0b59f199 -->
<a id="u-8b559d50ca0b59f199"></a>

Dummy encoding удаляет одну baseline category. Для baseline A:

<!-- /unit:8b559d50ca0b59f199 -->

<!-- unit:d41ae01e74646b40ec -->
<a id="u-d41ae01e74646b40ec"></a>

$$
\widehat y=b+\beta_B\mathbf1[c=B]+\beta_C\mathbf1[c=C].
$$

<!-- /unit:d41ae01e74646b40ec -->

<!-- unit:34a7f6e30f125db593 -->
<a id="u-34a7f6e30f125db593"></a>

Тогда $b$ — effect A, $\beta_B$ и $\beta_C$ — differences от A. Если средние outcomes A/B/C равны 10/13/8, coefficients составляют $b=10,\beta_B=3,\beta_C=-2$. Существуют и другие contrast codings; baseline выбран для интерпретации, а не потому, что A важнее.

<!-- /unit:34a7f6e30f125db593 -->

<!-- unit:1d1cefca019f10bd99 -->
<a id="u-1d1cefca019f10bd99"></a>

Удалять column обязательно не для каждого estimator. Regularized model может иметь уникальное penalized solution и с full one-hot; trees не требуют full-rank design matrix. При penalty choice baseline может менять symmetry regularization, поэтому drop-first не универсально нейтрален.

<!-- /unit:1d1cefca019f10bd99 -->

<!-- unit:6737b7a829f6911b78 -->
<a id="u-6737b7a829f6911b78"></a>

### Почему coefficients могут быть неоднозначными

<!-- /unit:6737b7a829f6911b78 -->

<!-- unit:bf249fc612a89b94cb -->
<a id="u-bf249fc612a89b94cb"></a>

Для A/B/C design с intercept содержит строки `(1,1,0,0)`, `(1,0,1,0)`, `(1,0,0,1)`. Ранг равен 3 при четырёх columns, поскольку intercept равен сумме category indicators. Coefficients b=0,w=(10,13,8) и b=10,w=(0,3,−2) дают одинаковые predictions. Сдвиг b на a и всех wk на −a сохраняет каждую строку.

<!-- /unit:bf249fc612a89b94cb -->

<!-- unit:b9406c9e54b2bc6616 -->
<a id="u-b9406c9e54b2bc6616"></a>

Без intercept full one-hot из трёх наблюдавшихся уровней имеет rank 3, и category coefficients в этой игрушечной задаче определяются однозначно. Если одновременно удалить baseline-column и не добавить intercept, baseline получит prediction 0 в модели только с этим field. Поэтому «всегда удалять первый столбец» без учёта estimator может испортить постановку.

<!-- /unit:b9406c9e54b2bc6616 -->

<!-- unit:ea5ecc872dc8d0346d -->
<a id="u-ea5ecc872dc8d0346d"></a>

Некоторые penalties устраняют неоднозначность полного design, выбирая предпочтительное представление. Например, ridge с unpenalized intercept может иметь единственное решение, хотя unregularized design был зависим. Trees вообще не решают normal equations этой матрицы. Требование full rank нельзя механически переносить на все модели. Доступные encoder options описаны в [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

<!-- /unit:ea5ecc872dc8d0346d -->

<!-- unit:c933d3e8d6296f003d -->
<a id="u-c933d3e8d6296f003d"></a>

### Одна функция и разные штрафы baseline

<!-- /unit:c933d3e8d6296f003d -->

<!-- unit:b13454909703dbd1f2 -->
<a id="u-b13454909703dbd1f2"></a>

Зафиксируем predictions категорий 10,13,8. С baseline A получаем b=10, beta_B=3, beta_C=−2 и сумму squared coefficients 13, если intercept не штрафуется. С baseline C получаем b=8, beta_A=2, beta_B=5 и сумму квадратов 29. Predictions идентичны, а penalty — разная.

<!-- /unit:b13454909703dbd1f2 -->

<!-- unit:45a40afa3459aefdf2 -->
<a id="u-45a40afa3459aefdf2"></a>

В полном one-hot среди представлений этих же predictions минимальная сумма squared category coefficients достигается при b=31/3, w=(−1/3,8/3,−7/3), сумма=38/3. Здесь рассматривается выбор параметризации уже фиксированной функции; конечный penalized fit может изменить и predictions. Пример показывает, почему drop-first при регуляризации не всегда симметричен относительно категорий.

<!-- /unit:45a40afa3459aefdf2 -->

<!-- unit:6b4c9d0cb735ac6518 -->
<a id="u-6b4c9d0cb735ac6518"></a>

### Смысл coefficient зависит от link и других features

<!-- /unit:6b4c9d0cb735ac6518 -->

<!-- unit:d07018fe83d54c447a -->
<a id="u-d07018fe83d54c447a"></a>

В модели только с category и identity link baseline b можно читать как средний response A. С числовыми covariates это response при reference values остальных features, а с interactions эффект category может зависеть от контекста. Нельзя автоматически называть каждый coefficient безусловным средним всей группы.

<!-- /unit:d07018fe83d54c447a -->

<!-- unit:23bb131b72ef3b6493 -->
<a id="u-23bb131b72ef3b6493"></a>

В logistic regression b — baseline log-odds, а beta_B — разность log-odds. Если p_A=0.2 и p_B=0.5, b=−log4, beta_B=log4; odds ratio равен 4. Это не увеличение probability на четыре единицы и не гарантированный causal effect. Кодирование помогает представлению ассоциации, но не заменяет постановку причинного вопроса.

<!-- /unit:23bb131b72ef3b6493 -->

<!-- unit:596d22de1a5288aa6d -->
<a id="u-596d22de1a5288aa6d"></a>

## Missing, unknown и словарь

<!-- /unit:596d22de1a5288aa6d -->

<!-- unit:e459649658a8d05b31 -->
<a id="u-e459649658a8d05b31"></a>

Если baseline A закодирован всеми нулями и unseen D тоже получает zero vector, model будет трактовать D как A. Это допустимый fallback только при осознанном решении, а не следствие того, что модель «поняла новый город». Explicit unknown bucket позволяет отдельное поведение, но если он ни разу не встречался в training, его coefficient/embedding не выучен обычными данными.

<!-- /unit:e459649658a8d05b31 -->

<!-- unit:f3036aea8265de70cd -->
<a id="u-f3036aea8265de70cd"></a>

Rare grouping объединяет малочисленные categories в OTHER по train counts. Это уменьшает variance, но стирает различия; threshold выбирается на validation. Нельзя построить словарь по train+test лишь потому, что labels не используются: для inductive evaluation distribution тоже должна быть доступна только по protocol.

<!-- /unit:f3036aea8265de70cd -->

<!-- unit:04e14e08df40e4b839 -->
<a id="u-04e14e08df40e4b839"></a>

Feature names, category ordering и dtype сохраняются вместе с model. Если при inference B получил index A, prediction ошибочен, хотя shapes совпадают. Serialization round-trip должен включать encoder, а не только final estimator.

<!-- /unit:04e14e08df40e4b839 -->

<!-- unit:aa73a81fc2b8406c14 -->
<a id="u-aa73a81fc2b8406c14"></a>

### Пять состояний одного field

<!-- /unit:aa73a81fc2b8406c14 -->

<!-- unit:42a12bd024a5004ce0 -->
<a id="u-42a12bd024a5004ce0"></a>

Учебный train vocabulary содержит A/B/C. Можно дополнительно зарезервировать MISSING и UNKNOWN, получив пять отдельных basis coordinates. A/B/C — известные значения; MISSING — значение отсутствует; D на inference — значение есть, но его нет в vocabulary, поэтому используется UNKNOWN. Если literal string «MISSING» может быть настоящим значением, нужен недвусмысленный способ отличать её от служебного состояния.

<!-- /unit:42a12bd024a5004ce0 -->

<!-- unit:198d15d7571a762d9c -->
<a id="u-198d15d7571a762d9c"></a>

Можно выбрать и другой contract: ошибка на unseen, all-zero, либо объединение с обученной rare-группой. При full one-hot all-zero означает отсутствие category contribution и оставляет intercept плюс другие features, а не автоматически population-average prediction. При drop-first тот же zero совпадает с baseline A. Это два разных fallback при внешне одинаковом векторе.

<!-- /unit:198d15d7571a762d9c -->

<!-- unit:35ada9cf625e847c04 -->
<a id="u-35ada9cf625e847c04"></a>

Reserved UNKNOWN-column позволяет отдельно хранить состояние, но её coefficient не выучен из обычных labels, если column никогда не активировалась в train. Regularization или initialization зададут некоторое поведение, но не знание о новом значении D. Можно обучать объявленный fallback на редких/искусственно скрытых категориях, проверяя, насколько этот режим соответствует реальному unseen; подробный контракт рассматривается в главе [«Missing, unknown, rare и стабильный словарь»](<feat-missing.md>).

<!-- /unit:35ada9cf625e847c04 -->

<!-- unit:edac6e75732cd0de31 -->
<a id="u-edac6e75732cd0de31"></a>

### Где проходит граница fit

<!-- /unit:edac6e75732cd0de31 -->

<!-- unit:8b5e6448684a5ce5bd -->
<a id="u-8b5e6448684a5ce5bd"></a>

Если vocabulary и rare threshold выводятся из текущих данных, их fit находится внутри train каждого fold. Отсутствие использования y не даёт права читать будущую popularity или состав закрытого test в индуктивном эксперименте. В то же время заранее заданный внешний справочник допустимых категорий — иной источник знания: его можно зафиксировать независимо от текущего sample и описать как schema/prior.

<!-- /unit:8b5e6448684a5ce5bd -->

<!-- unit:16201f3295235e6c9c -->
<a id="u-16201f3295235e6c9c"></a>

Для ordinal features порядок уровней лучше задавать содержательно, а не доверять alphabetic sort. Unknown и missing codes должны быть отличимы от нормальных levels, если такой контракт выбран. Numeric sentinel −1, переданный как обычный ordinal feature, также получает геометрический смысл для модели; он не становится «ничем» автоматически. Точные options проверяют в [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

<!-- /unit:16201f3295235e6c9c -->

<!-- unit:1dacd25fde858b5a14 -->
<a id="u-1dacd25fde858b5a14"></a>

### Что сохраняется с моделью

<!-- /unit:1dacd25fde858b5a14 -->

<!-- unit:80cda4e0743f771e77 -->
<a id="u-80cda4e0743f771e77"></a>

Сохраняют имена и порядок fields, category ordering, dtype/canonicalization, rare threshold, baseline, special states, feature names и final estimator. Inference не должен заново сортировать только встретившиеся в новом batch значения. Иначе B может незаметно получить column A при прежнем shape, и model weights будут применены к другому смыслу.

<!-- /unit:80cda4e0743f771e77 -->

<!-- unit:2fb07d459c7185a1ff -->
<a id="u-2fb07d459c7185a1ff"></a>

Round-trip проверяют на известных categories, missing, unseen, пустых строках и значениях разных типов, если они разрешены схемой. Inverse transform all-zero может возвращать baseline или unknown-marker в зависимости от configuration; восстановление строки не является универсально однозначным. Важнее соответствие сохранённому контракту, чем отсутствие ошибки сериализации.

<!-- /unit:2fb07d459c7185a1ff -->

<!-- unit:ad133a03ebbfeb9f16 -->
<a id="u-ad133a03ebbfeb9f16"></a>

## Контроль и практика

<!-- /unit:ad133a03ebbfeb9f16 -->

<!-- unit:4b925888b1737fd1bb -->
<a id="u-4b925888b1737fd1bb"></a>

1. Почему arbitrary integer codes вредят linear model? **Ответ:** вводят порядок и равные расстояния, которых nominal feature не имеет.
2. Что устраняет drop-first? **Ответ:** зависимость dummy columns с intercept в полной линейной design matrix.
3. Почему unknown нельзя молча приравнивать missing? **Ответ:** это разные процессы; объединение теряет возможный сигнал.

<!-- /unit:4b925888b1737fd1bb -->

<!-- unit:b77d8f441d6589cb87 -->
<a id="u-b77d8f441d6589cb87"></a>

**Практика:** закодируйте пять строк с A/B/C, missing и unseen D. Для full/dummy схем выпишите predictions и fallback. Покажите две разные группы coefficients, дающие одни predictions при intercept+full one-hot. Сохраните vocabulary и проверьте восстановление column order.

<!-- /unit:b77d8f441d6589cb87 -->

<!-- unit:66993bf5ed64854d78 -->
<a id="u-66993bf5ed64854d78"></a>

**Схема курса:** raw category + declared semantics → nominal/ordinal decision → fitted vocabulary → one-hot/dummy columns → unknown policy → model input

<!-- /unit:66993bf5ed64854d78 -->

<!-- unit:c9697848961eae581c -->
<a id="u-c9697848961eae581c"></a>

<!-- source-depth-categorical-А -->

<!-- /unit:c9697848961eae581c -->

<!-- unit:7f0f8ac64147158da0 -->
<a id="u-7f0f8ac64147158da0"></a>

1. **Почему естественный порядок не задаёт равных intervals?** Он сообщает сравнение levels, но не величину эффекта каждого перехода.
2. **Когда dummy coding меняет не только интерпретацию?** При distance-based модели и при penalties, зависящих от выбранных coordinates.
3. **Почему all-zero unknown не равен «модель поняла новый класс»?** Это фиксированный fallback без обученного индивидуального effect.
4. **Почему одинаковый output shape не доказывает правильный encoder?** Значения и columns могли поменять смысл при другом mapping.

<!-- /unit:7f0f8ac64147158da0 -->

<!-- unit:a2059ada9b71e26c46 -->
<a id="u-a2059ada9b71e26c46"></a>

Практика: воспроизвести numeric OLS, full/dummy predictions и два penalty values при разных baselines. Для пяти состояний field выписать точные vectors и fallback. Сохранить encoder с моделью и проверить поведение после восстановления. Все строки и coefficients главы учебные; реальный пользовательский dataset не перекодировался.

<!-- /unit:a2059ada9b71e26c46 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как представить категорию, не приписав ей ложный числовой порядок?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Кардинальность, частотные кодировки и hashing](<feat-hashing.md>), [Target statistics, smoothing, cross-fitting и WoE](<feat-target.md>), [Обучаемые embeddings и перенос представлений](<feat-embeddings.md>), [Нативные категории в бустинге](<feat-native.md>), [Взаимодействия признаков, FM и FFM](<feat-interactions.md>), [Наборы категорий и pooling](<feat-sets.md>), [Отбор признаков и устойчивость выбора](<feat-selection.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
