# А. Зачем ограничивать модель, которая уже хорошо обучается

[Оглавление](README.md). Пункт 1 плана. **Происхождение:** дополнительный материал по плану; отдельного полного чата по регуляризации нет. В LLM-чатах встречаются KL, PEFT и generalization, но не этот теоретический курс.

## 1. Training error и риск

Модель минимизирует empirical risk на конечном training set:

$$
\widehat R_D(f)=\frac1n\sum_{i=1}^n\ell(f(x_i),y_i).
$$

Цель — низкий expected risk на будущих независимых объектах, а не только на запомненных строках. Регуляризация меняет предпочтения между решениями: штрафует веса, ограничивает доступные функции, вводит шум или останавливает оптимизацию. Иногда training error растёт, а validation error уменьшается. Но усиление ограничения способно ухудшить оба показателя.

Большой train–validation gap согласуется с overfitting, однако не доказывает его единственную причину. Domain shift, неодинаковая очистка, leakage или неправильный eval mode тоже меняют кривые. Маленькая training error при плохом test требует сначала проверить data protocol.

## 2. Точный смысл bias–variance

Пусть $Y=f^*(x)+\varepsilon$, где $\mathbb E[\varepsilon\mid x]=0$, conditional variance равна $\sigma^2(x)$, а обучаемая по случайному dataset $D$ функция — $\widehat f_D$. Для squared loss в фиксированном $x$:

$$
\mathbb E_{D,\varepsilon}[(Y-\widehat f_D(x))^2]
=\underbrace{(\mathbb E_D[\widehat f_D(x)]-f^*(x))^2}_{\mathrm{bias}^2}
+\underbrace{\operatorname{Var}_D[\widehat f_D(x)]}_{\mathrm{variance}}
+\sigma^2(x).
$$

Bias — систематическое смещение среднего предсказателя; variance — чувствительность к выбору training data. Noise нельзя устранить улучшением предсказания условного среднего при данной доступной информации. Это разложение не переносится буквально на любой loss и не означает, что каждая ошибка конкретной модели помечается только одним из трёх слов.

Пример: истинный target 3, две равновероятные обученные модели предсказывают 2 и 6. Среднее 4, bias squared равен 1, variance — 4, expected squared error без observation noise — 5. Более ограниченная модель всегда предсказывает 4: bias тот же, variance ноль, ошибка 1. Но если ограничение заставляет всегда предсказывать 0, ошибка 9. Ограничение полезно только в конкретном компромиссе.

## 3. MAP: штраф как prior

MAP максимизирует posterior, или минимизирует negative log-likelihood плюс negative log-prior:

$$
\widehat w_{\mathrm{MAP}}=
\arg\min_w\left[-\log p(D\mid w)-\log p(w)\right].
$$

Независимый Gaussian prior $w_j\sim\mathcal N(0,\tau^2)$ даёт L2 term $\|w\|_2^2/(2\tau^2)$. Laplace prior с scale $b$ даёт $\|w\|_1/b$. Коэффициенты зависят от likelihood/noise и от того, делится ли data loss на $n$. Поэтому одна и та же численная lambda в двух библиотеках не обязательно означает один objective.

MAP — point estimate, а не полный Bayesian posterior. Prior выражает предпочтение параметров, но его effect зависит от parametrization и scaling features. «Меньше weights» не всегда означает «проще функция» при возможности rescale соседние слои.

## 4. Что дают VC, Rademacher и PAC-Bayes

VC dimension описывает способность бинарного hypothesis class реализовывать все labelings на некоторых наборах points. Rademacher complexity измеряет, насколько class способен коррелировать со случайными signs на sample. Общая идея bounds: empirical error дополняется ценой complexity и неопределённости. Эти понятия требуют определённых assumptions и сами по себе не гарантируют, что любая маленькая сеть лучше большой.

PAC-Bayes сравнивает data-dependent posterior $Q$ над predictors с prior $P$, выбранным независимо от текущей training выборки. Один распространённый вид для bounded loss и i.i.d. data:

$$
\operatorname{kl}(\widehat R(Q)\Vert R(Q))
\le\frac{D_{\mathrm{KL}}(Q\Vert P)+\log(2\sqrt n/\delta)}n
$$

с вероятностью не менее $1-\delta$ при условиях соответствующей теоремы. Здесь $\operatorname{kl}$ — binary relative entropy, а $R(Q)$ — риск стохастического Gibbs predictor. Это не автоматически tight guarantee обычного deterministic checkpoint. Prior, loss bounds и единица независимого sample принципиальны. См. [учебный разбор PAC-Bayes и kl-bound](https://www.pascalgermain.info/talks/pacbayes-tutorial-2023.pdf).

У современных overparameterized networks число parameters само по себе плохо описывает весь процесс обобщения. Implicit bias optimizer, data structure и norms могут быть важнее грубого parameter count. Рассуждение «простота помогает» полезно как гипотеза, но проверяется на validation и не заменяет корректный эксперимент.

## 5. Контроль и практика

1. Верно ли bias²+variance+noise для любого loss? **Ответ:** указанная формула относится к squared error с заданными assumptions.
2. Равен ли MAP полному Bayesian inference? **Ответ:** нет; MAP выбирает одну точку posterior.
3. Может ли сильная регуляризация дать underfit? **Ответ:** да; ограничение может исключить полезную функцию и повысить bias.

**Практика:** для нескольких случайных training samples мысленно сравните почти константную и интерполирующую модели. Вычислите bias/variance для предсказаний $(2,6)$ и $(4,4)$ при target 3. Затем выпишите likelihood и prior для ridge, явно отметив normalizer data loss.
