# Target statistics, smoothing, cross-fitting и WoE

**Уровень:** общее ядро.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Split, cross-validation и утечки](<../01-reliability/df-split.md>), [Дисбаланс, метрики, порог и калибровка](<../01-reliability/df-metrics.md>)

**Используют:** [Нативные категории в бустинге](<feat-native.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>)

**Учебный вопрос:** Как использовать целевую статистику, не раскрывая собственную метку?

## Карта вычисления и границы этапов

![Target statistics, smoothing, cross-fitting и WoE: входы, преобразования и выходы](<../../оформление/course/feat-target.svg>)

Признаки обучающих строк строятся вне собственного фолда. Для новых объектов используется отдельная карта по всей разрешённой обучающей части; эти две карты выполняют разные роли.

## Target statistics и утечка

Для category $c$ на train имеем $n_c$ labels (метки) и сумму $s_c$. Наивное encoding равно $s_c/n_c$. Для нового validation (проверочная часть для выбора решения) объекта это допустимая статистика, если она fit только по train. Но для самой training строки её label (метка) входит в statistic. При уникальном ID $n_c=1$ encoding буквально равно $y_i$; модель получает готовый ответ.

Train score (оценка) такого pipeline может быть идеальным даже при случайных labels. На новых IDs mapping отсутствует, и quality исчезает. Проблема не в плохом downstream estimator: вход уже содержит утечку. Train и inference (применение обученной модели) features (признаки) должны иметь сопоставимый механизм получения.

### Conditional mean — это выбор представления

В регрессии target encoding оценивает $m(c)=\mathbb E[Y\mid C=c]$. Для squared loss (функция потерь) conditional mean является оптимальным предсказанием среди всех функций только этого field: разложение условного риска даёт $\mathbb E[(Y-a)^2\mid c]=\operatorname{Var}(Y\mid c)+(m(c)-a)^2$. Это объясняет, почему среднее может быть полезно, но не делает его достаточным признаком для любой задачи.

Две категории с одинаковым средним могут иметь разные variance, tails и взаимодействия с другими fields. Пусть для A и B средняя покупка одинакова, но в A высокий спрос возникает только вечером, а в B — утром. Один scalar mean не сохраняет эту структуру. Счётчик наблюдений и исходная category, если модель умеет с ней работать, могут дополнять statistic. Добавлять дисперсию или quantiles нужно по гипотезе задачи и с тем же ограничением доступности labels.

Mean class ID для multiclass особенно опасен: classes «кошка», «собака», «птица» не находятся на числовой оси. При coding 0/1/2 distribution (0.5,0,0.5) и distribution (0,1,0) имеют одинаковое среднее 1, хотя предсказывают разные outcomes. Переименование class IDs меняет такой feature (признак). Вектор class probabilities сохраняет нужные различия; порядок его columns становится частью сохранённого контракта.

### Откуда берётся формула smoothing

Для binary target предположим независимые Bernoulli labels внутри category с неизвестным параметром $\theta_c$. Зададим prior $\theta_c\sim\operatorname{Beta}(\alpha\mu,\alpha(1-\mu))$, где $0<\mu<1$ и $\alpha>0$. После $s_c$ единиц и $n_c-s_c$ нулей posterior имеет параметры $s_c+\alpha\mu$ и $n_c-s_c+\alpha(1-\mu)$. Его среднее точно совпадает с TE из раздела 2.

При $\mu=0.5,\alpha=2$ prior равномерен Beta(1,1). Категория A с одним success из двух получает Beta(2,2), mean 0.5 и posterior variance 0.05. Категория B с двумя successes получает Beta(3,1), mean 0.75 и variance 0.0375. Это неопределённость параметра в данной вероятностной модели, а не variance самого будущего Bernoulli outcome и не автоматически калиброванный confidence (уверенность по принятой оценке) interval реального pipeline.

При таком prior posterior variance равна

$$
\operatorname{Var}(\theta_c\mid D)=
\frac{m_c(1-m_c)}{n_c+\alpha+1},
\qquad m_c=\frac{s_c+\alpha\mu}{n_c+\alpha}.
$$

Если prior mean и strength оценены по данным, это empirical Bayes, а не заранее известная истина. В честном training feature эти оценки тоже должны получать только разрешённые labels. Точная формула автоматического smoothing конкретной библиотеки может отличаться от учебного Beta prior. В regression та же shrinkage-форма может возникать из Gaussian model с иными предположениями о variance; binary conjugacy не переносится автоматически на произвольный непрерывный target.

### Почему сглаживание иногда ухудшает оценку

Зафиксируем истинное $\theta$, известный prior mean $\mu$ и count $n$. Для estimator $(S+\alpha\mu)/(n+\alpha)$, где $S\sim\operatorname{Binomial}(n,\theta)$:

$$
\operatorname{Bias}=\frac{\alpha(\mu-\theta)}{n+\alpha},
\qquad
\operatorname{Var}=\frac{n\theta(1-\theta)}{(n+\alpha)^2}.
$$

При n=2, alpha=2 и theta=mu=0.5 variance уменьшается с 0.125 для raw mean до 0.03125, bias отсутствует. Но при theta=0.9 и mu=0.5 bias становится −0.2, variance 0.01125, MSE 0.05125. У raw mean MSE 0.045: сильное притяжение к неверному prior здесь вредит. Это частотный MSE estimator при повторении datasets, отдельно от posterior variance выше. Следовательно, alpha регулирует bias–variance, а не задаёт обязательное улучшение.

## Shrinkage, smoothing и LOO

Сглаженный mapping притягивает mean редкой category к глобальному prior $\mu$:

$$
\operatorname{TE}(c)=\frac{s_c+\alpha\mu}{n_c+\alpha}
=\frac{n_c}{n_c+\alpha}\overline y_c+
\frac\alpha{n_c+\alpha}\mu.
$$

$\alpha$ действует как pseudo-count для prior. При $n_c=0$ получаем $\mu$; при большом count влияние category grows. Но smoothing само по себе не удаляет label leakage (утечка недоступной информации): собственный $y_i$ остаётся в $s_c$, если не изменить data protocol. Коэффициент выбирается на validation, не по test.

Для multiclass можно кодировать distribution по classes с согласованным prior и нормировкой. Один scalar mean class ID не имеет смысла для nominal multiclass targets (целевые значения). Regression использует conditional mean, но outliers и drift (изменение распределения или зависимости со временем) требуют отдельной проверки.

### Собственная метка остаётся даже после сильного shrinkage

Для singleton category с фиксированным внешним prior mu=0.5 и alpha=10 encoding равен 6/11 при y=1 и 5/11 при y=0. Значения близки, но threshold 0.5 всё ещё идеально восстанавливает training label. Поэтому фраза «добавили большое сглаживание, leakage исчезла» неверна даже в простейшем примере. Никакой measured score здесь не заявлен: это точное следствие формулы.

Если глобальный prior тоже посчитан по полному train из N строк, собственная метка входит двумя путями: через category sum и через global mean. При фиксированных memberships производная непрерывного продолжения statistic по $y_i$ равна

$$
\frac{\partial z_i}{\partial y_i}
=\frac{1+\alpha/N}{n_{c_i}+\alpha}.
$$

Даже если из category sum удалить $y_i$, но оставить full-train prior, останется влияние $\alpha/[N(n_{c_i}-1+\alpha)]$. Для binary label это можно понимать как разницу encoding при переключении 0→1 с остальными labels неизменными. Cross-fitting должен исключать held-out fold и из category statistics, и из оценки prior, smoothing rule и любых supervised промежуточных преобразований.

### Чужая строка может содержать почти тот же ответ

Допустим, у одного пациента несколько почти одинаковых измерений с одним диагнозом. Random folds исключают конкретную строку, но оставляют записи того же пациента в обучении mapping. Если deployment требует предсказывать новых пациентов, такой split (разбиение данных) проверяет другую задачу. Для нового пациента нужны group-disjoint внешние и внутренние splits; для повторного визита известного пациента допустимая история задаётся временем и доступностью результата.

OOF устраняет включение held-out label в его собственную fitted statistic при выбранных folds. Это не утверждение независимости всех получившихся строк и не универсальное доказательство отсутствия leakage. Категории, дубли, batch (пакет примеров) effects, delayed labels и схема разбиения определяют, какие связи разрешены. LOO дополнительно оставляет почти полный overlap между reference sets разных строк, поэтому легко получает тесные зависимости даже без прямого собственного label.

## Все строки ручного расчёта

Пусть categories идут A,A,B,B,C,C,D,D,E,E, labels — $(1,0,1,1,0,0,1,0,1,0)$. Global mean $\mu=0.5$. Наивные category means: A=0.5, B=1, C=0, D=0.5, E=0.5. При $\alpha=2$ full-train smoothed mapping: A=0.5, B=0.75, C=0.25, D=0.5, E=0.5.

Для OOF разделим odd и even row indices. Features odd rows считаются только по even rows, где mean 0.2; features even — только по odd, где mean 0.8. В каждой category противоположный fold содержит один example. Поэтому row 1 (A,label 1) получает smoothed value $(0+2\cdot0.2)/3\approx0.1333$, а row 2 (A,label 0) — $(1+2\cdot0.8)/3\approx0.8667$.

Эти values отличаются от full mapping 0.5, и они не используют собственную label. Маленькие folds дали очень шумные оценки и даже неблагоприятный pattern для A; cross-fitting устраняет конкретное прямое подглядывание, а не гарантирует хороший predictive signal. Выбор folds и достаточный объём data важны. Все fold priors тоже считаются только по соответствующей training части.

### Полный расчёт OOF

Для набора раздела 3 odd rows используют even reference labels (0,1,0,0,0), prior 1/5; even rows используют odd reference labels (1,1,0,1,1), prior 4/5. В противоположной половине для каждой category есть ровно одна строка. При alpha=2 знаменатель всегда 3.

По исходному порядку строк получаем

$$
z^{\mathrm{OOF}}=
\frac1{15}(2,13,7,13,2,8,2,13,2,13).
$$

Например, row 3 категории B получает $(1+2/5)/3=7/15$, а row 6 категории C — $(0+8/5)/3=8/15$. Full-train smoothed vector для тех же строк равен (0.5,0.5,0.75,0.75,0.25,0.25,0.5,0.5,0.5,0.5). Подменить им OOF matrix при обучении downstream model означает вернуть собственные labels в features.

Нужно сохранять различие двух результатов: OOF matrix нужна для fit downstream model; mapping по полному разрешённому train нужен для будущих объектов. Повторный вызов transform на train может законно вернуть full mapping, а не прежние OOF values. Поэтому train metric, посчитанная после такого повторного transform, уже не описывает тот feature matrix, на котором обучалась модель. OOF matrix и fold assignment сохраняют отдельно, если они нужны для аудита.

### Почему OOF по всему dataset перед внешней CV недостаточно

Пусть для evaluation внешняя training часть — строки 1–8, validation — 9–10. Если заранее построить OOF на всех десяти строках с odd/even folds, row 1 использует even prior, в который входит label row 10 из внешнего validation. Собственная label row 1 исключена, но внешний validation уже влияет на вход training algorithm. Переключение y10 с 0 на 1 меняет prior с 1/5 до 2/5 и z1 с 2/15 до 4/15. Через эту связь validation участвует в выборе fitted model.

Правильный порядок: выбрать внешний train; внутри него заново получить OOF features; fit downstream model; fit serving (обслуживание запросов к модели) mapping по всему этому внешнему train; только затем transform внешнего validation. При tuning alpha, grouping thresholds или model capacity вся конструкция повторяется в разрешённом evaluation protocol. Финальный test не становится ещё одним validation для выбора удобного encoding.

### Несовпадение reference size — ожидаемая часть метода

При K folds OOF statistics обычно обучены примерно на $(K-1)/K$ train, а serving mapping — на всём train. Для rare categories это заметно меняет counts, variance и число fallback cases. Увеличение K уменьшает различие размеров reference sets, но увеличивает вычисления и overlap; бесконечного обещанного выигрыша нет. Усреднение моделей по folds — другой алгоритм: у каждой модели свой reference mapping и меньше training data. Его сравнивают отдельно, не называя тождественным full refit.

Повторный cross-fitting с несколькими seeds может уменьшать чувствительность к одному partition. Однако downstream validation остаётся внешней; averaging OOF values само по себе не доказывает качество, а повторные seeds не являются независимыми новыми datasets. Для времени начальные строки могут не иметь доступной истории: допустимы внешний prior, warm-up или исключение из fit с честным указанием population. Будущие labels нельзя использовать ради заполнения всех ячеек.

## Вложенный protocol

Для внешнего train $T$ строим внутренние folds $T_1,\ldots,T_K$. Каждая training строка из $T_k$ кодируется mapping, fit по $T\setminus T_k$. Downstream model обучается на этих OOF features. Для внешнего validation $V$ mapping fit по всему $T$ и применяется без использования labels из $V$.

$$
z_i=\operatorname{TE}_{T\setminus T_k}(c_i),\ i\in T_k;
\qquad
z_j=\operatorname{TE}_{T}(c_j),\ j\in V.
$$

Нельзя сначала получить OOF feature по всему dataset, а потом выполнить внешнюю CV: labels внешнего validation уже могли попасть в features внешнего train и повлиять на fit модели. Каждая внешняя итерация должна повторять всю процедуру подготовки.

Group/time constraints действуют и на внутренний split. Random internal KFold не подходит для causal features в исторической задаче. Текущий sklearn `TargetEncoder` различает `fit_transform` и `fit(...).transform(...)` на train из-за cross-fitting; особенности принимаемых splitters зависят от версии. [Документация TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html). Не предполагайте, что любой encoder (кодировщик входа) с похожим названием делает это автоматически.

## Ordered statistics

LOO исключает собственный label из category sum:

$$
\operatorname{LOO}_i=
\frac{s_{c_i}-y_i+\alpha\mu_{-i}}
{n_{c_i}-1+\alpha}.
$$

Здесь показан и prior без собственной строки. Для category size 2 без smoothing feature одной строки равно label второй. Это уменьшает прямое включение собственной label, но сохраняет сильные зависимости training examples и train/inference mismatch. LOO не автоматически безопаснее правильно устроенного cross-fitting.

Ordered statistics CatBoost используют precedents в permutation: для строки учитываются ранее расположенные примеры и prior. Это отличается от full mean и уменьшает target leakage/prediction shift в соответствующем алгоритме. Random permutation не является chronological history: timestamp availability в time task надо контролировать дополнительно. Ordered statistics и ordered boosting — связанные, но разные компоненты алгоритма. [CatBoost](https://arxiv.org/abs/1706.09516).

В учебном варианте ordered statistics с prior mu=0.5, alpha=2 и последовательностью A:1, A:0, B:1, A:1 значения перед очередным update равны 0.5, 2/3, 0.5, 0.5. Сначала вычисляется feature из prefix (уже доступный префикс), затем добавляется текущая label. Обратный порядок снова включает ответ. Это демонстрация prefix mechanism, а не точное воспроизведение всех разновидностей CTR в CatBoost.

Случайная permutation обеспечивает алгоритмический порядок в пределах training dataset. Временная доступность — отдельный контракт реального процесса; permutation сама его не создаёт. Ordered boosting дополнительно решает проблему зависимостей при построении последовательности boosting residuals, поэтому его нельзя свести к одному заранее рассчитанному столбцу TE. Механизм и различие компонентов описаны в [работе CatBoost](https://arxiv.org/abs/1706.09516).

Для собственного pipeline сохраняют field schema (схема допустимых данных), category normalization (нормировка по выбранному правилу), fold/reference definition, sums/counts, prior, smoothing, class column order, fallback и data cutoff. На inference update labels не происходит, если явно не определён online protocol. Round-trip mapping должен давать те же features на известных, rare, missing и unseen inputs; только равенства shape недостаточно.

В [документации TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) различаются training fit_transform с cross-fitting и transform с сохранённым mapping. Поддержка передаваемых splitters зависит от установленной версии. Temporal splitter, не покрывающий каждую строку ровно один раз, может не соответствовать контракту API; тогда нужен явный causal feature builder, а не предположение, что название splitter автоматически делает estimator корректным.

**Самопроверка:** singleton со сглаживанием всё ещё выдаёт label через threshold; global prior тоже может протекать; OOF до внешнего split недостаточно; group/time restrictions проверяются на обоих уровнях; full mapping и OOF matrix — разные артефакты. Практику из раздела 7 теперь можно воспроизвести по всем десяти значениям, а не по двум первым строкам.

## WoE и multiclass

При выбранном positive class определим:

$$
\operatorname{WoE}(c)=
\log\frac{P(C=c\mid Y=1)}{P(C=c\mid Y=0)}.
$$

По Bayes:

$$
\log\frac{P(Y=1\mid c)}{P(Y=0\mid c)}
=\operatorname{WoE}(c)+\log\frac{P(Y=1)}{P(Y=0)}.
$$

WoE отличается от posterior log-odds на prior log-odds; знак меняется, если поменять positive/negative convention. Нулевые counts дают бесконечность без smoothing. Binning, smoothing и supervised mapping fit только на train, а для downstream train features тоже требуется честный protocol. WoE не является free leakage-proof alternative TE.

### Distribution вместо среднего номера класса

Для L классов зададим prior probabilities $\pi_\ell$, сумма которых 1. Тогда

$$
z_{c,\ell}=\frac{n_{c,\ell}+\alpha\pi_\ell}{n_c+\alpha}.
$$

При counts (2,1,0), prior (0.2,0.3,0.5), alpha=2 получаем (0.48,0.32,0.20), сумма равна 1. Отсутствие третьего класса среди трёх наблюдений не стало заявлением о нулевой вероятности. Если обучающая часть fold не содержит один global class, схема columns всё равно должна сохранять объявленный class space и предусмотренный prior. Иначе concatenate незаметно смешает значения разных classes.

Sample weights меняют оцениваемую population. Взвешенный statistic использует $s_c=\sum w_i y_i$, $n_c=\sum w_i$ и согласованный weighted prior. Умножение всех весов на 10 меняет относительную силу фиксированного alpha: pseudo-count имеет единицы выбранной массы. Для observations с весами 1 и 9 эффективный размер по формуле $(\sum w)^2/\sum w^2$ равен 100/82≈1.22, а не 10 независимым наблюдениям. Это ориентир для концентрации весов при соответствующих предположениях, не универсальная замена count во всех posterior models.

### Числовое различие WoE и posterior odds

Пусть всего 40 positives и 60 negatives; у category c — 12 positives и 6 negatives. Тогда WoE=log((12/40)/(6/60))=log 3≈1.0986. Posterior positive probability равна 12/18=2/3, posterior log-odds — log 2≈0.6931. Разница объясняется prior log-odds log(40/60)≈−0.4055: log 3+log(2/3)=log 2.

При smoothing нужно объявить, какое распределение сглаживается. Например, для конечного словаря из трёх categories добавление единицы к каждому class-conditional category count даёт вероятности 13/43 и 7/63 для c; соответствующий smoothed WoE равен log(117/43)≈1.0010. Это отличается от добавления единицы к двум outcomes внутри c и вычисления log(13/7). Первое сглаживает $P(C\mid Y)$, второе — conditional outcome odds; нельзя произвольно смешать их нормировочные constants и затем ссылаться на точную Bayes identity для одного общего распределения.

Supervised binning числового field до WoE тоже обучается: границы, monotonic constraints и объединение rare bins должны выбирать только на разрешённом train. Бесконечность от нулевого count говорит об ограниченной оценке, а не о доказанном бесконечном risk ratio. Clipping задаёт ещё один hyperparameter, а последующая logistic regression не снимает leakage из созданных features.

## Контроль и практика

1. Убирает ли smoothing собственный label? **Ответ:** нет; нужна смена способа вычисления train feature.
2. Почему нельзя один OOF feature использовать до внешней CV? **Ответ:** внешний validation может уже участвовать в features, по которым обучается модель.
3. Чем WoE отличается от posterior log-odds? **Ответ:** prior odds constant при указанной convention.

**Практика:** досчитайте OOF values всех десяти строк при $\alpha=2$. Добавьте случайный unique ID и сравните naive, smoothed и OOF representations. Нарисуйте два уровня split с точным указанием labels, доступных каждому mapping. Новые measured scores (оценки) сообщайте только после реального запуска.



## Результат главы и переход дальше

Следующие темы: [Нативные категории в бустинге](<feat-native.md>), [Единый pipeline и применение к новым объектам](<feat-pipeline.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
