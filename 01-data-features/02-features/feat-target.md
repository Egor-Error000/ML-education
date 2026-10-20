# Target statistics, smoothing, cross-fitting и WoE

**Уровень:** общее ядро.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Split, cross-validation и утечки](<../01-reliability/df-split.md>), [Дисбаланс, метрики, порог и калибровка](<../01-reliability/df-metrics.md>)

**Используют:** [Нативные категории в бустинге](<feat-native.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>)

**Учебный вопрос:** Как использовать целевую статистику, не раскрывая собственную метку?





## Карта вычисления и границы этапов

![Target statistics, smoothing, cross-fitting и WoE: входы, преобразования и выходы](<../../оформление/course/feat-target.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:6a3c1e53161a5bb893 -->
<a id="u-6a3c1e53161a5bb893"></a>

## Target statistics и утечка

<!-- /unit:6a3c1e53161a5bb893 -->

<!-- unit:856c0227acc4f64390 -->
<a id="u-856c0227acc4f64390"></a>

Для category $c$ на train имеем $n_c$ labels и сумму $s_c$. Наивное encoding равно $s_c/n_c$. Для нового validation объекта это допустимая статистика, если она fit только по train. Но для самой training строки её label входит в statistic. При уникальном ID $n_c=1$ encoding буквально равно $y_i$; модель получает готовый ответ.

<!-- /unit:856c0227acc4f64390 -->

<!-- unit:a956de0444d49ae2f9 -->
<a id="u-a956de0444d49ae2f9"></a>

Train score такого pipeline может быть идеальным даже при случайных labels. На новых IDs mapping отсутствует, и quality исчезает. Проблема не в плохом downstream estimator: вход уже содержит утечку. Train и inference features должны иметь сопоставимый механизм получения.

<!-- /unit:a956de0444d49ae2f9 -->

<!-- unit:530c777f43326fa1cc -->
<a id="u-530c777f43326fa1cc"></a>

### Conditional mean — это выбор представления

<!-- /unit:530c777f43326fa1cc -->

<!-- unit:ef1ff5bc37ce6fed08 -->
<a id="u-ef1ff5bc37ce6fed08"></a>

В регрессии target encoding оценивает $m(c)=\mathbb E[Y\mid C=c]$. Для squared loss conditional mean является оптимальным предсказанием среди всех функций только этого field: разложение условного риска даёт $\mathbb E[(Y-a)^2\mid c]=\operatorname{Var}(Y\mid c)+(m(c)-a)^2$. Это объясняет, почему среднее может быть полезно, но не делает его достаточным признаком для любой задачи.

<!-- /unit:ef1ff5bc37ce6fed08 -->

<!-- unit:280a9fd621fe5c3f06 -->
<a id="u-280a9fd621fe5c3f06"></a>

Две категории с одинаковым средним могут иметь разные variance, tails и взаимодействия с другими fields. Пусть для A и B средняя покупка одинакова, но в A высокий спрос возникает только вечером, а в B — утром. Один scalar mean не сохраняет эту структуру. Счётчик наблюдений и исходная category, если модель умеет с ней работать, могут дополнять statistic. Добавлять дисперсию или quantiles нужно по гипотезе задачи и с тем же ограничением доступности labels.

<!-- /unit:280a9fd621fe5c3f06 -->

<!-- unit:ed6d0a3e9e9c3bda0e -->
<a id="u-ed6d0a3e9e9c3bda0e"></a>

Mean class ID для multiclass особенно опасен: classes «кошка», «собака», «птица» не находятся на числовой оси. При coding 0/1/2 distribution (0.5,0,0.5) и distribution (0,1,0) имеют одинаковое среднее 1, хотя предсказывают разные outcomes. Переименование class IDs меняет такой feature. Вектор class probabilities сохраняет нужные различия; порядок его columns становится частью сохранённого контракта.

<!-- /unit:ed6d0a3e9e9c3bda0e -->

<!-- unit:8e26f3ce888d4f3028 -->
<a id="u-8e26f3ce888d4f3028"></a>

### Откуда берётся формула smoothing

<!-- /unit:8e26f3ce888d4f3028 -->

<!-- unit:de5892ded04626f81c -->
<a id="u-de5892ded04626f81c"></a>

Для binary target предположим независимые Bernoulli labels внутри category с неизвестным параметром $\theta_c$. Зададим prior $\theta_c\sim\operatorname{Beta}(\alpha\mu,\alpha(1-\mu))$, где $0<\mu<1$ и $\alpha>0$. После $s_c$ единиц и $n_c-s_c$ нулей posterior имеет параметры $s_c+\alpha\mu$ и $n_c-s_c+\alpha(1-\mu)$. Его среднее точно совпадает с TE из раздела 2.

<!-- /unit:de5892ded04626f81c -->

<!-- unit:01a83fb6e4fc3f0f56 -->
<a id="u-01a83fb6e4fc3f0f56"></a>

При $\mu=0.5,\alpha=2$ prior равномерен Beta(1,1). Категория A с одним success из двух получает Beta(2,2), mean 0.5 и posterior variance 0.05. Категория B с двумя successes получает Beta(3,1), mean 0.75 и variance 0.0375. Это неопределённость параметра в данной вероятностной модели, а не variance самого будущего Bernoulli outcome и не автоматически калиброванный confidence interval реального pipeline.

<!-- /unit:01a83fb6e4fc3f0f56 -->

<!-- unit:3d2d89f0f0694f05fc -->
<a id="u-3d2d89f0f0694f05fc"></a>

При таком prior posterior variance равна

<!-- /unit:3d2d89f0f0694f05fc -->

<!-- unit:5ff4e339d11e12829d -->
<a id="u-5ff4e339d11e12829d"></a>

$$
\operatorname{Var}(\theta_c\mid D)=
\frac{m_c(1-m_c)}{n_c+\alpha+1},
\qquad m_c=\frac{s_c+\alpha\mu}{n_c+\alpha}.
$$

<!-- /unit:5ff4e339d11e12829d -->

<!-- unit:5e708ff759fb73e4ba -->
<a id="u-5e708ff759fb73e4ba"></a>

Если prior mean и strength оценены по данным, это empirical Bayes, а не заранее известная истина. В честном training feature эти оценки тоже должны получать только разрешённые labels. Точная формула автоматического smoothing конкретной библиотеки может отличаться от учебного Beta prior. В regression та же shrinkage-форма может возникать из Gaussian model с иными предположениями о variance; binary conjugacy не переносится автоматически на произвольный непрерывный target.

<!-- /unit:5e708ff759fb73e4ba -->

<!-- unit:49d98dd21ba9e41538 -->
<a id="u-49d98dd21ba9e41538"></a>

### Почему сглаживание иногда ухудшает оценку

<!-- /unit:49d98dd21ba9e41538 -->

<!-- unit:a573c13b7de12e9e8f -->
<a id="u-a573c13b7de12e9e8f"></a>

Зафиксируем истинное $\theta$, известный prior mean $\mu$ и count $n$. Для estimator $(S+\alpha\mu)/(n+\alpha)$, где $S\sim\operatorname{Binomial}(n,\theta)$:

<!-- /unit:a573c13b7de12e9e8f -->

<!-- unit:ad0f009c2d1758a26f -->
<a id="u-ad0f009c2d1758a26f"></a>

$$
\operatorname{Bias}=\frac{\alpha(\mu-\theta)}{n+\alpha},
\qquad
\operatorname{Var}=\frac{n\theta(1-\theta)}{(n+\alpha)^2}.
$$

<!-- /unit:ad0f009c2d1758a26f -->

<!-- unit:89fd608f6a0e093f3b -->
<a id="u-89fd608f6a0e093f3b"></a>

При n=2, alpha=2 и theta=mu=0.5 variance уменьшается с 0.125 для raw mean до 0.03125, bias отсутствует. Но при theta=0.9 и mu=0.5 bias становится −0.2, variance 0.01125, MSE 0.05125. У raw mean MSE 0.045: сильное притяжение к неверному prior здесь вредит. Это частотный MSE estimator при повторении datasets, отдельно от posterior variance выше. Следовательно, alpha регулирует bias–variance, а не задаёт обязательное улучшение.

<!-- /unit:89fd608f6a0e093f3b -->

<!-- unit:d4e3ae309e2b08a1f3 -->
<a id="u-d4e3ae309e2b08a1f3"></a>

## Shrinkage, smoothing и LOO

<!-- /unit:d4e3ae309e2b08a1f3 -->

<!-- unit:0f817d115962ca095d -->
<a id="u-0f817d115962ca095d"></a>

Сглаженный mapping притягивает mean редкой category к глобальному prior $\mu$:

<!-- /unit:0f817d115962ca095d -->

<!-- unit:27106acb7087438e58 -->
<a id="u-27106acb7087438e58"></a>

$$
\operatorname{TE}(c)=\frac{s_c+\alpha\mu}{n_c+\alpha}
=\frac{n_c}{n_c+\alpha}\overline y_c+
\frac\alpha{n_c+\alpha}\mu.
$$

<!-- /unit:27106acb7087438e58 -->

<!-- unit:452332e1430e192b05 -->
<a id="u-452332e1430e192b05"></a>

$\alpha$ действует как pseudo-count для prior. При $n_c=0$ получаем $\mu$; при большом count влияние category grows. Но smoothing само по себе не удаляет label leakage: собственный $y_i$ остаётся в $s_c$, если не изменить data protocol. Коэффициент выбирается на validation, не по test.

<!-- /unit:452332e1430e192b05 -->

<!-- unit:e3d1203f55d077217e -->
<a id="u-e3d1203f55d077217e"></a>

Для multiclass можно кодировать distribution по classes с согласованным prior и нормировкой. Один scalar mean class ID не имеет смысла для nominal multiclass targets. Regression использует conditional mean, но outliers и drift требуют отдельной проверки.

<!-- /unit:e3d1203f55d077217e -->

<!-- unit:a096895adc113588ec -->
<a id="u-a096895adc113588ec"></a>

### Собственная метка остаётся даже после сильного shrinkage

<!-- /unit:a096895adc113588ec -->

<!-- unit:ff8fb8fea9cba5ec05 -->
<a id="u-ff8fb8fea9cba5ec05"></a>

Для singleton category с фиксированным внешним prior mu=0.5 и alpha=10 encoding равен 6/11 при y=1 и 5/11 при y=0. Значения близки, но threshold 0.5 всё ещё идеально восстанавливает training label. Поэтому фраза «добавили большое сглаживание, leakage исчезла» неверна даже в простейшем примере. Никакой measured score здесь не заявлен: это точное следствие формулы.

<!-- /unit:ff8fb8fea9cba5ec05 -->

<!-- unit:885cedd2ecf9364098 -->
<a id="u-885cedd2ecf9364098"></a>

Если глобальный prior тоже посчитан по полному train из N строк, собственная метка входит двумя путями: через category sum и через global mean. При фиксированных memberships производная непрерывного продолжения statistic по $y_i$ равна

<!-- /unit:885cedd2ecf9364098 -->

<!-- unit:723193864f9a082874 -->
<a id="u-723193864f9a082874"></a>

$$
\frac{\partial z_i}{\partial y_i}
=\frac{1+\alpha/N}{n_{c_i}+\alpha}.
$$

<!-- /unit:723193864f9a082874 -->

<!-- unit:5d9edba1f2899b8ed9 -->
<a id="u-5d9edba1f2899b8ed9"></a>

Даже если из category sum удалить $y_i$, но оставить full-train prior, останется влияние $\alpha/[N(n_{c_i}-1+\alpha)]$. Для binary label это можно понимать как разницу encoding при переключении 0→1 с остальными labels неизменными. Cross-fitting должен исключать held-out fold и из category statistics, и из оценки prior, smoothing rule и любых supervised промежуточных преобразований.

<!-- /unit:5d9edba1f2899b8ed9 -->

<!-- unit:895660b0f180e139dc -->
<a id="u-895660b0f180e139dc"></a>

### Чужая строка может содержать почти тот же ответ

<!-- /unit:895660b0f180e139dc -->

<!-- unit:a90c437887913490b2 -->
<a id="u-a90c437887913490b2"></a>

Допустим, у одного пациента несколько почти одинаковых измерений с одним диагнозом. Random folds исключают конкретную строку, но оставляют записи того же пациента в обучении mapping. Если deployment требует предсказывать новых пациентов, такой split проверяет другую задачу. Для нового пациента нужны group-disjoint внешние и внутренние splits; для повторного визита известного пациента допустимая история задаётся временем и доступностью результата.

<!-- /unit:a90c437887913490b2 -->

<!-- unit:1f5b32aba7d7b2540e -->
<a id="u-1f5b32aba7d7b2540e"></a>

OOF устраняет включение held-out label в его собственную fitted statistic при выбранных folds. Это не утверждение независимости всех получившихся строк и не универсальное доказательство отсутствия leakage. Категории, дубли, batch effects, delayed labels и схема разбиения определяют, какие связи разрешены. LOO дополнительно оставляет почти полный overlap между reference sets разных строк, поэтому легко получает тесные зависимости даже без прямого собственного label.

<!-- /unit:1f5b32aba7d7b2540e -->

<!-- unit:47eb2728141ad62119 -->
<a id="u-47eb2728141ad62119"></a>

## Все строки ручного расчёта

<!-- /unit:47eb2728141ad62119 -->

<!-- unit:cc931bf53567f26b0d -->
<a id="u-cc931bf53567f26b0d"></a>

Пусть categories идут A,A,B,B,C,C,D,D,E,E, labels — $(1,0,1,1,0,0,1,0,1,0)$. Global mean $\mu=0.5$. Наивные category means: A=0.5, B=1, C=0, D=0.5, E=0.5. При $\alpha=2$ full-train smoothed mapping: A=0.5, B=0.75, C=0.25, D=0.5, E=0.5.

<!-- /unit:cc931bf53567f26b0d -->

<!-- unit:b3d6a13affb0c0ca9c -->
<a id="u-b3d6a13affb0c0ca9c"></a>

Для OOF разделим odd и even row indices. Features odd rows считаются только по even rows, где mean 0.2; features even — только по odd, где mean 0.8. В каждой category противоположный fold содержит один example. Поэтому row 1 (A,label 1) получает smoothed value $(0+2\cdot0.2)/3\approx0.1333$, а row 2 (A,label 0) — $(1+2\cdot0.8)/3\approx0.8667$.

<!-- /unit:b3d6a13affb0c0ca9c -->

<!-- unit:f92b3a747d05b23a28 -->
<a id="u-f92b3a747d05b23a28"></a>

Эти values отличаются от full mapping 0.5, и они не используют собственную label. Маленькие folds дали очень шумные оценки и даже неблагоприятный pattern для A; cross-fitting устраняет конкретное прямое подглядывание, а не гарантирует хороший predictive signal. Выбор folds и достаточный объём data важны. Все fold priors тоже считаются только по соответствующей training части.

<!-- /unit:f92b3a747d05b23a28 -->

<!-- unit:f344dae59afaf45f3f -->
<a id="u-f344dae59afaf45f3f"></a>

### Полный расчёт OOF

<!-- /unit:f344dae59afaf45f3f -->

<!-- unit:30a341d0ffd5991e35 -->
<a id="u-30a341d0ffd5991e35"></a>

Для набора раздела 3 odd rows используют even reference labels (0,1,0,0,0), prior 1/5; even rows используют odd reference labels (1,1,0,1,1), prior 4/5. В противоположной половине для каждой category есть ровно одна строка. При alpha=2 знаменатель всегда 3.

<!-- /unit:30a341d0ffd5991e35 -->

<!-- unit:2dbcbcd63e125a5cfd -->
<a id="u-2dbcbcd63e125a5cfd"></a>

По исходному порядку строк получаем

<!-- /unit:2dbcbcd63e125a5cfd -->

<!-- unit:d090a30dddc65cfae6 -->
<a id="u-d090a30dddc65cfae6"></a>

$$
z^{\mathrm{OOF}}=
\frac1{15}(2,13,7,13,2,8,2,13,2,13).
$$

<!-- /unit:d090a30dddc65cfae6 -->

<!-- unit:4384f4b98f0cb38ec9 -->
<a id="u-4384f4b98f0cb38ec9"></a>

Например, row 3 категории B получает $(1+2/5)/3=7/15$, а row 6 категории C — $(0+8/5)/3=8/15$. Full-train smoothed vector для тех же строк равен (0.5,0.5,0.75,0.75,0.25,0.25,0.5,0.5,0.5,0.5). Подменить им OOF matrix при обучении downstream model означает вернуть собственные labels в features.

<!-- /unit:4384f4b98f0cb38ec9 -->

<!-- unit:5703e813a82b02b5ec -->
<a id="u-5703e813a82b02b5ec"></a>

Нужно сохранять различие двух результатов: OOF matrix нужна для fit downstream model; mapping по полному разрешённому train нужен для будущих объектов. Повторный вызов transform на train может законно вернуть full mapping, а не прежние OOF values. Поэтому train metric, посчитанная после такого повторного transform, уже не описывает тот feature matrix, на котором обучалась модель. OOF matrix и fold assignment сохраняют отдельно, если они нужны для аудита.

<!-- /unit:5703e813a82b02b5ec -->

<!-- unit:274b708729296ab437 -->
<a id="u-274b708729296ab437"></a>

### Почему OOF по всему dataset перед внешней CV недостаточно

<!-- /unit:274b708729296ab437 -->

<!-- unit:8518a65cbe94b72374 -->
<a id="u-8518a65cbe94b72374"></a>

Пусть для evaluation внешняя training часть — строки 1–8, validation — 9–10. Если заранее построить OOF на всех десяти строках с odd/even folds, row 1 использует even prior, в который входит label row 10 из внешнего validation. Собственная label row 1 исключена, но внешний validation уже влияет на вход training algorithm. Переключение y10 с 0 на 1 меняет prior с 1/5 до 2/5 и z1 с 2/15 до 4/15. Через эту связь validation участвует в выборе fitted model.

<!-- /unit:8518a65cbe94b72374 -->

<!-- unit:a76e145503fd689414 -->
<a id="u-a76e145503fd689414"></a>

Правильный порядок: выбрать внешний train; внутри него заново получить OOF features; fit downstream model; fit serving mapping по всему этому внешнему train; только затем transform внешнего validation. При tuning alpha, grouping thresholds или model capacity вся конструкция повторяется в разрешённом evaluation protocol. Финальный test не становится ещё одним validation для выбора удобного encoding.

<!-- /unit:a76e145503fd689414 -->

<!-- unit:1cb6f624c53c908b2f -->
<a id="u-1cb6f624c53c908b2f"></a>

### Несовпадение reference size — ожидаемая часть метода

<!-- /unit:1cb6f624c53c908b2f -->

<!-- unit:9c2d22b85062564195 -->
<a id="u-9c2d22b85062564195"></a>

При K folds OOF statistics обычно обучены примерно на $(K-1)/K$ train, а serving mapping — на всём train. Для rare categories это заметно меняет counts, variance и число fallback cases. Увеличение K уменьшает различие размеров reference sets, но увеличивает вычисления и overlap; бесконечного обещанного выигрыша нет. Усреднение моделей по folds — другой алгоритм: у каждой модели свой reference mapping и меньше training data. Его сравнивают отдельно, не называя тождественным full refit.

<!-- /unit:9c2d22b85062564195 -->

<!-- unit:f0d85d126be5297ce1 -->
<a id="u-f0d85d126be5297ce1"></a>

Повторный cross-fitting с несколькими seeds может уменьшать чувствительность к одному partition. Однако downstream validation остаётся внешней; averaging OOF values само по себе не доказывает качество, а повторные seeds не являются независимыми новыми datasets. Для времени начальные строки могут не иметь доступной истории: допустимы внешний prior, warm-up или исключение из fit с честным указанием population. Будущие labels нельзя использовать ради заполнения всех ячеек.

<!-- /unit:f0d85d126be5297ce1 -->

<!-- unit:e03c0adee96b897285 -->
<a id="u-e03c0adee96b897285"></a>

## Вложенный protocol

<!-- /unit:e03c0adee96b897285 -->

<!-- unit:318e497887fc7e265d -->
<a id="u-318e497887fc7e265d"></a>

Для внешнего train $T$ строим внутренние folds $T_1,\ldots,T_K$. Каждая training строка из $T_k$ кодируется mapping, fit по $T\setminus T_k$. Downstream model обучается на этих OOF features. Для внешнего validation $V$ mapping fit по всему $T$ и применяется без использования labels из $V$.

<!-- /unit:318e497887fc7e265d -->

<!-- unit:f63393cf9f6910bc5a -->
<a id="u-f63393cf9f6910bc5a"></a>

$$
z_i=\operatorname{TE}_{T\setminus T_k}(c_i),\ i\in T_k;
\qquad
z_j=\operatorname{TE}_{T}(c_j),\ j\in V.
$$

<!-- /unit:f63393cf9f6910bc5a -->

<!-- unit:3b526f412dcf769bed -->
<a id="u-3b526f412dcf769bed"></a>

Нельзя сначала получить OOF feature по всему dataset, а потом выполнить внешнюю CV: labels внешнего validation уже могли попасть в features внешнего train и повлиять на fit модели. Каждая внешняя итерация должна повторять всю процедуру подготовки.

<!-- /unit:3b526f412dcf769bed -->

<!-- unit:b66ff547ec81e5efb6 -->
<a id="u-b66ff547ec81e5efb6"></a>

Group/time constraints действуют и на внутренний split. Random internal KFold не подходит для causal features в исторической задаче. Текущий sklearn `TargetEncoder` различает `fit_transform` и `fit(...).transform(...)` на train из-за cross-fitting; особенности принимаемых splitters зависят от версии. [Документация TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html). Не предполагайте, что любой encoder с похожим названием делает это автоматически.

<!-- /unit:b66ff547ec81e5efb6 -->

<!-- unit:a07b1cfce5b65f3a2c -->
<a id="u-a07b1cfce5b65f3a2c"></a>

## Ordered statistics

<!-- /unit:a07b1cfce5b65f3a2c -->

<!-- unit:21dfeea6b8f2ca1e20 -->
<a id="u-21dfeea6b8f2ca1e20"></a>

LOO исключает собственный label из category sum:

<!-- /unit:21dfeea6b8f2ca1e20 -->

<!-- unit:c69d737163fd9afebc -->
<a id="u-c69d737163fd9afebc"></a>

$$
\operatorname{LOO}_i=
\frac{s_{c_i}-y_i+\alpha\mu_{-i}}
{n_{c_i}-1+\alpha}.
$$

<!-- /unit:c69d737163fd9afebc -->

<!-- unit:22c6982e1979fc0964 -->
<a id="u-22c6982e1979fc0964"></a>

Здесь показан и prior без собственной строки. Для category size 2 без smoothing feature одной строки равно label второй. Это уменьшает прямое включение собственной label, но сохраняет сильные зависимости training examples и train/inference mismatch. LOO не автоматически безопаснее правильно устроенного cross-fitting.

<!-- /unit:22c6982e1979fc0964 -->

<!-- unit:888f220a2b9e3ad3db -->
<a id="u-888f220a2b9e3ad3db"></a>

Ordered statistics CatBoost используют precedents в permutation: для строки учитываются ранее расположенные примеры и prior. Это отличается от full mean и уменьшает target leakage/prediction shift в соответствующем алгоритме. Random permutation не является chronological history: timestamp availability в time task надо контролировать дополнительно. Ordered statistics и ordered boosting — связанные, но разные компоненты алгоритма. [CatBoost](https://arxiv.org/abs/1706.09516).

<!-- /unit:888f220a2b9e3ad3db -->

<!-- unit:f23a3fa49a30be6540 -->
<a id="u-f23a3fa49a30be6540"></a>

В учебном варианте ordered statistics с prior mu=0.5, alpha=2 и последовательностью A:1, A:0, B:1, A:1 значения перед очередным update равны 0.5, 2/3, 0.5, 0.5. Сначала вычисляется feature из prefix, затем добавляется текущая label. Обратный порядок снова включает ответ. Это демонстрация prefix mechanism, а не точное воспроизведение всех разновидностей CTR в CatBoost.

<!-- /unit:f23a3fa49a30be6540 -->

<!-- unit:37b013287cfd21d0bb -->
<a id="u-37b013287cfd21d0bb"></a>

Случайная permutation обеспечивает алгоритмический порядок в пределах training dataset. Временная доступность — отдельный контракт реального процесса; permutation сама его не создаёт. Ordered boosting дополнительно решает проблему зависимостей при построении последовательности boosting residuals, поэтому его нельзя свести к одному заранее рассчитанному столбцу TE. Механизм и различие компонентов описаны в [работе CatBoost](https://arxiv.org/abs/1706.09516).

<!-- /unit:37b013287cfd21d0bb -->

<!-- unit:e93c43ba60365bcbec -->
<a id="u-e93c43ba60365bcbec"></a>

Для собственного pipeline сохраняют field schema, category normalization, fold/reference definition, sums/counts, prior, smoothing, class column order, fallback и data cutoff. На inference update labels не происходит, если явно не определён online protocol. Round-trip mapping должен давать те же features на известных, rare, missing и unseen inputs; только равенства shape недостаточно.

<!-- /unit:e93c43ba60365bcbec -->

<!-- unit:feb0dc262640e3efd7 -->
<a id="u-feb0dc262640e3efd7"></a>

В [документации TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) различаются training fit_transform с cross-fitting и transform с сохранённым mapping. Поддержка передаваемых splitters зависит от установленной версии. Temporal splitter, не покрывающий каждую строку ровно один раз, может не соответствовать контракту API; тогда нужен явный causal feature builder, а не предположение, что название splitter автоматически делает estimator корректным.

<!-- /unit:feb0dc262640e3efd7 -->

<!-- unit:fbe5cf3a4effdc87c0 -->
<a id="u-fbe5cf3a4effdc87c0"></a>

**Самопроверка:** singleton со сглаживанием всё ещё выдаёт label через threshold; global prior тоже может протекать; OOF до внешнего split недостаточно; group/time restrictions проверяются на обоих уровнях; full mapping и OOF matrix — разные артефакты. Практику из раздела 7 теперь можно воспроизвести по всем десяти значениям, а не по двум первым строкам.

<!-- /unit:fbe5cf3a4effdc87c0 -->

<!-- unit:fbe396e11b492b177c -->
<a id="u-fbe396e11b492b177c"></a>

## WoE и multiclass

<!-- /unit:fbe396e11b492b177c -->

<!-- unit:ed6a57dd989eedea5a -->
<a id="u-ed6a57dd989eedea5a"></a>

При выбранном positive class определим:

<!-- /unit:ed6a57dd989eedea5a -->

<!-- unit:bb0bee2695c68279eb -->
<a id="u-bb0bee2695c68279eb"></a>

$$
\operatorname{WoE}(c)=
\log\frac{P(C=c\mid Y=1)}{P(C=c\mid Y=0)}.
$$

<!-- /unit:bb0bee2695c68279eb -->

<!-- unit:9618837726c1e6a8ee -->
<a id="u-9618837726c1e6a8ee"></a>

По Bayes:

<!-- /unit:9618837726c1e6a8ee -->

<!-- unit:4cb65da6aae2035221 -->
<a id="u-4cb65da6aae2035221"></a>

$$
\log\frac{P(Y=1\mid c)}{P(Y=0\mid c)}
=\operatorname{WoE}(c)+\log\frac{P(Y=1)}{P(Y=0)}.
$$

<!-- /unit:4cb65da6aae2035221 -->

<!-- unit:faa727885f0c17e0ad -->
<a id="u-faa727885f0c17e0ad"></a>

WoE отличается от posterior log-odds на prior log-odds; знак меняется, если поменять positive/negative convention. Нулевые counts дают бесконечность без smoothing. Binning, smoothing и supervised mapping fit только на train, а для downstream train features тоже требуется честный protocol. WoE не является free leakage-proof alternative TE.

<!-- /unit:faa727885f0c17e0ad -->

<!-- unit:7479c0d86358c128d6 -->
<a id="u-7479c0d86358c128d6"></a>

### Distribution вместо среднего номера класса

<!-- /unit:7479c0d86358c128d6 -->

<!-- unit:61950f77f8e1bcfcb1 -->
<a id="u-61950f77f8e1bcfcb1"></a>

Для L классов зададим prior probabilities $\pi_\ell$, сумма которых 1. Тогда

<!-- /unit:61950f77f8e1bcfcb1 -->

<!-- unit:982f488758d61ed6e5 -->
<a id="u-982f488758d61ed6e5"></a>

$$
z_{c,\ell}=\frac{n_{c,\ell}+\alpha\pi_\ell}{n_c+\alpha}.
$$

<!-- /unit:982f488758d61ed6e5 -->

<!-- unit:337367089b6bd6dde5 -->
<a id="u-337367089b6bd6dde5"></a>

При counts (2,1,0), prior (0.2,0.3,0.5), alpha=2 получаем (0.48,0.32,0.20), сумма равна 1. Отсутствие третьего класса среди трёх наблюдений не стало заявлением о нулевой вероятности. Если обучающая часть fold не содержит один global class, схема columns всё равно должна сохранять объявленный class space и предусмотренный prior. Иначе concatenate незаметно смешает значения разных classes.

<!-- /unit:337367089b6bd6dde5 -->

<!-- unit:486cc8065669391489 -->
<a id="u-486cc8065669391489"></a>

Sample weights меняют оцениваемую population. Взвешенный statistic использует $s_c=\sum w_i y_i$, $n_c=\sum w_i$ и согласованный weighted prior. Умножение всех весов на 10 меняет относительную силу фиксированного alpha: pseudo-count имеет единицы выбранной массы. Для observations с весами 1 и 9 эффективный размер по формуле $(\sum w)^2/\sum w^2$ равен 100/82≈1.22, а не 10 независимым наблюдениям. Это ориентир для концентрации весов при соответствующих предположениях, не универсальная замена count во всех posterior models.

<!-- /unit:486cc8065669391489 -->

<!-- unit:c0a166c4354adf6272 -->
<a id="u-c0a166c4354adf6272"></a>

### Числовое различие WoE и posterior odds

<!-- /unit:c0a166c4354adf6272 -->

<!-- unit:fbc3790867a318b02d -->
<a id="u-fbc3790867a318b02d"></a>

Пусть всего 40 positives и 60 negatives; у category c — 12 positives и 6 negatives. Тогда WoE=log((12/40)/(6/60))=log 3≈1.0986. Posterior positive probability равна 12/18=2/3, posterior log-odds — log 2≈0.6931. Разница объясняется prior log-odds log(40/60)≈−0.4055: log 3+log(2/3)=log 2.

<!-- /unit:fbc3790867a318b02d -->

<!-- unit:0d7faf5428ae9fc0b5 -->
<a id="u-0d7faf5428ae9fc0b5"></a>

При smoothing нужно объявить, какое распределение сглаживается. Например, для конечного словаря из трёх categories добавление единицы к каждому class-conditional category count даёт вероятности 13/43 и 7/63 для c; соответствующий smoothed WoE равен log(117/43)≈1.0010. Это отличается от добавления единицы к двум outcomes внутри c и вычисления log(13/7). Первое сглаживает $P(C\mid Y)$, второе — conditional outcome odds; нельзя произвольно смешать их нормировочные constants и затем ссылаться на точную Bayes identity для одного общего распределения.

<!-- /unit:0d7faf5428ae9fc0b5 -->

<!-- unit:1e78cb4aaeb49b054a -->
<a id="u-1e78cb4aaeb49b054a"></a>

Supervised binning числового field до WoE тоже обучается: границы, monotonic constraints и объединение rare bins должны выбирать только на разрешённом train. Бесконечность от нулевого count говорит об ограниченной оценке, а не о доказанном бесконечном risk ratio. Clipping задаёт ещё один hyperparameter, а последующая logistic regression не снимает leakage из созданных features.

<!-- /unit:1e78cb4aaeb49b054a -->

<!-- unit:006dc88498fb1d0be4 -->
<a id="u-006dc88498fb1d0be4"></a>

## Контроль и практика

<!-- /unit:006dc88498fb1d0be4 -->

<!-- unit:d6d88ef8d379013508 -->
<a id="u-d6d88ef8d379013508"></a>

1. Убирает ли smoothing собственный label? **Ответ:** нет; нужна смена способа вычисления train feature.
2. Почему нельзя один OOF feature использовать до внешней CV? **Ответ:** внешний validation может уже участвовать в features, по которым обучается модель.
3. Чем WoE отличается от posterior log-odds? **Ответ:** prior odds constant при указанной convention.

<!-- /unit:d6d88ef8d379013508 -->

<!-- unit:e19d56d5cb8d2c6385 -->
<a id="u-e19d56d5cb8d2c6385"></a>

**Практика:** досчитайте OOF values всех десяти строк при $\alpha=2$. Добавьте случайный unique ID и сравните naive, smoothed и OOF representations. Нарисуйте два уровня split с точным указанием labels, доступных каждому mapping. Новые measured scores сообщайте только после реального запуска.

<!-- /unit:e19d56d5cb8d2c6385 -->

<!-- unit:cc24e74763aa99c545 -->
<a id="u-cc24e74763aa99c545"></a>

<!-- source-depth-categorical-В -->

<!-- /unit:cc24e74763aa99c545 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как использовать целевую статистику, не раскрывая собственную метку?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Нативные категории в бустинге](<feat-native.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
