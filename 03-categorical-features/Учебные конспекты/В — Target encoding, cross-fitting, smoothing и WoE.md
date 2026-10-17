# В. Как использовать target statistics, не подглядывая в ответ

[Оглавление](README.md). Пункт 3 и практика вложенной валидации. **Происхождение:** подробное дополнение по плану. Это центральная глава курса: неправильная подготовка features делает последующее сравнение недостоверным.

## 1. Наивное среднее и уникальный ID

Для category $c$ на train имеем $n_c$ labels и сумму $s_c$. Наивное encoding равно $s_c/n_c$. Для нового validation объекта это допустимая статистика, если она fit только по train. Но для самой training строки её label входит в statistic. При уникальном ID $n_c=1$ encoding буквально равно $y_i$; модель получает готовый ответ.

Train score такого pipeline может быть идеальным даже при случайных labels. На новых IDs mapping отсутствует, и quality исчезает. Проблема не в плохом downstream estimator: вход уже содержит утечку. Train и inference features должны иметь сопоставимый механизм получения.

## 2. Smoothing как shrinkage

Сглаженный mapping притягивает mean редкой category к глобальному prior $\mu$:

$$
\operatorname{TE}(c)=\frac{s_c+\alpha\mu}{n_c+\alpha}
=\frac{n_c}{n_c+\alpha}\overline y_c+
\frac\alpha{n_c+\alpha}\mu.
$$

$\alpha$ действует как pseudo-count для prior. При $n_c=0$ получаем $\mu$; при большом count влияние category grows. Но smoothing само по себе не удаляет label leakage: собственный $y_i$ остаётся в $s_c$, если не изменить data protocol. Коэффициент выбирается на validation, не по test.

Для multiclass можно кодировать distribution по classes с согласованным prior и нормировкой. Один scalar mean class ID не имеет смысла для nominal multiclass targets. Regression использует conditional mean, но outliers и drift требуют отдельной проверки.

## 3. Числовой набор из десяти строк

Пусть categories идут A,A,B,B,C,C,D,D,E,E, labels — $(1,0,1,1,0,0,1,0,1,0)$. Global mean $\mu=0.5$. Наивные category means: A=0.5, B=1, C=0, D=0.5, E=0.5. При $\alpha=2$ full-train smoothed mapping: A=0.5, B=0.75, C=0.25, D=0.5, E=0.5.

Для OOF разделим odd и even row indices. Features odd rows считаются только по even rows, где mean 0.2; features even — только по odd, где mean 0.8. В каждой category противоположный fold содержит один example. Поэтому row 1 (A,label 1) получает smoothed value $(0+2\cdot0.2)/3\approx0.1333$, а row 2 (A,label 0) — $(1+2\cdot0.8)/3\approx0.8667$.

Эти values отличаются от full mapping 0.5, и они не используют собственную label. Маленькие folds дали очень шумные оценки и даже неблагоприятный pattern для A; cross-fitting устраняет конкретное прямое подглядывание, а не гарантирует хороший predictive signal. Выбор folds и достаточный объём data важны. Все fold priors тоже считаются только по соответствующей training части.

## 4. Cross-fitting внутри внешнего split

Для внешнего train $T$ строим внутренние folds $T_1,\ldots,T_K$. Каждая training строка из $T_k$ кодируется mapping, fit по $T\setminus T_k$. Downstream model обучается на этих OOF features. Для внешнего validation $V$ mapping fit по всему $T$ и применяется без использования labels из $V$.

$$
z_i=\operatorname{TE}_{T\setminus T_k}(c_i),\ i\in T_k;
\qquad
z_j=\operatorname{TE}_{T}(c_j),\ j\in V.
$$

Нельзя сначала получить OOF feature по всему dataset, а потом выполнить внешнюю CV: labels внешнего validation уже могли попасть в features внешнего train и повлиять на fit модели. Каждая внешняя итерация должна повторять всю процедуру подготовки.

Group/time constraints действуют и на внутренний split. Random internal KFold не подходит для causal features в исторической задаче. Текущий sklearn `TargetEncoder` различает `fit_transform` и `fit(...).transform(...)` на train из-за cross-fitting; особенности принимаемых splitters зависят от версии. [Документация TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html). Не предполагайте, что любой encoder с похожим названием делает это автоматически.

## 5. Leave-one-out и ordered statistics

LOO исключает собственный label из category sum:

$$
\operatorname{LOO}_i=
\frac{s_{c_i}-y_i+\alpha\mu_{-i}}
{n_{c_i}-1+\alpha}.
$$

Здесь показан и prior без собственной строки. Для category size 2 без smoothing feature одной строки равно label второй. Это уменьшает прямое включение собственной label, но сохраняет сильные зависимости training examples и train/inference mismatch. LOO не автоматически безопаснее правильно устроенного cross-fitting.

Ordered statistics CatBoost используют precedents в permutation: для строки учитываются ранее расположенные примеры и prior. Это отличается от full mean и уменьшает target leakage/prediction shift в соответствующем алгоритме. Random permutation не является chronological history: timestamp availability в time task надо контролировать дополнительно. Ordered statistics и ordered boosting — связанные, но разные компоненты алгоритма. [CatBoost](https://arxiv.org/abs/1706.09516).

## 6. Weight of Evidence

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

## 7. Контроль и практика

1. Убирает ли smoothing собственный label? **Ответ:** нет; нужна смена способа вычисления train feature.
2. Почему нельзя один OOF feature использовать до внешней CV? **Ответ:** внешний validation может уже участвовать в features, по которым обучается модель.
3. Чем WoE отличается от posterior log-odds? **Ответ:** prior odds constant при указанной convention.

**Практика:** досчитайте OOF values всех десяти строк при $\alpha=2$. Добавьте случайный unique ID и сравните naive, smoothed и OOF representations. Нарисуйте два уровня split с точным указанием labels, доступных каждому mapping. Новые measured scores сообщайте только после реального запуска.
