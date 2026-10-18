# Д. Нормализация: что усредняется и почему это важно

[Оглавление](README.md). Пункт 5 плана. **Основа:** блоки Transformer из S2 и backbones CV из S7; систематическое сравнение семейств — дополнение. Нормализация прежде всего меняет оптимизацию, а шум статистик может дополнительно регуляризовать.

## 1. Общая формула и разные множества

Для выбранного множества элементов $\mathcal S$:

$$
\mu=\frac1{|\mathcal S|}\sum_{j\in\mathcal S}x_j,
\quad
v=\frac1{|\mathcal S|}\sum_{j\in\mathcal S}(x_j-\mu)^2,
\quad
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+\beta_i.
$$

Главное различие — какие элементы принадлежат $\mathcal S$ и как разделяются affine parameters. $\epsilon>0$ стабилизирует деление, поэтому variance нормированного output не обязана быть точно 1. Affine layer может затем намеренно изменить mean и scale.

Для тензора $N,C,H,W$ **BatchNorm** считает statistics по $N,H,W$ отдельно для каждого channel. **InstanceNorm** — по $H,W$ для каждого example/channel. **GroupNorm** — по пространству и channels выбранной группы внутри одного example. **LayerNorm** — по объявленным normalized dimensions; в Transformer $N,T,D$ обычно только по $D$ отдельно для каждой пары example/token. Нельзя без config считать, что LN всегда нормирует всё изображение.

Пример с одним channel и двумя изображениями, каждый из двух pixels: $(1,3)$ и $(5,7)$. BN использует все четыре значения, mean 4 и variance 5. IN для первого использует mean 2, variance 1, для второго mean 6, variance 1. Без epsilon/affine IN даёт $(-1,1)$ в обоих examples, стирая различие общей яркости. BN сохраняет его в normalized values. Это показывает, почему выбор norm меняет доступную информацию.

## 2. BatchNorm на training и inference

На training BN использует mini-batch statistics и обновляет running estimates. На eval типичный режим использует накопленные estimates, а не statistics текущего входного batch. Конкретные variance estimators и momentum convention зависят от library; не переносите число momentum из optimizer в BN по сходству названий.

Малый или сильно correlated batch даёт шумные statistics. Gradient accumulation не эквивалентен большому BN batch: несколько forward по microbatches не объединяют их statistics автоматически. SyncBatchNorm может синхронизировать statistics между devices, но не решает все проблемы не-i.i.d. data и имеет communication cost.

Шум batch statistics способен давать regularizing effect, но историческое объяснение через internal covariate shift не является единственным установленным механизмом. Исследуется также сглаживание optimization landscape; см. [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604). Нельзя считать BN простым заменителем L2 или dropout.

После freezing backbone нужно отдельно решить режим BN. Замороженные weights не обязательно означают замороженные running statistics. При fine-tuning с маленьким domain-specific batch их неосторожное обновление может ухудшить pretrained representation.

## 3. Batch Renormalization

Batch Renorm вводит corrections между текущими batch statistics и running estimates. Упрощённо:

$$
\widehat x_i=\frac{x_i-\mu_B}{\sigma_B},\quad
r=\operatorname{clip}\left(\frac{\sigma_B}{\sigma},\frac1{r_{\max}},r_{\max}\right),
\quad
d=\operatorname{clip}\left(\frac{\mu_B-\mu}{\sigma},-d_{\max},d_{\max}\right),
$$
$$
y_i=\gamma(r\widehat x_i+d)+\beta.
$$

Здесь standard deviations включают stabilizer по реализации, а $r,d$ используются с остановленным gradient; clipping limits могут меняться по schedule. Это не просто замена batch mean running mean во всех derivatives. Идея — уменьшить mismatch training/inference при сохранении управляемой оптимизации. [Batch Renormalization](https://arxiv.org/abs/1702.03275).

## 4. RMSNorm и нормализация weights

RMSNorm не центрирует:

$$
\operatorname{RMS}(x)=\sqrt{\frac1D\sum_{j=1}^D x_j^2+\epsilon},
\qquad
y_i=\gamma_i\frac{x_i}{\operatorname{RMS}(x)}.
$$

Для $(1,3)$ без epsilon LN даёт $(-1,1)$, RMSNorm — $(1/\sqrt5,3/\sqrt5)\approx(0.4472,1.3416)$. Mean не удалена. Это изменение преобразования, а не лишь сокращённая запись LN. Применимость оценивается для конкретной architecture/training recipe.

Weight Normalization параметризует vector weights как $w=g\,v/\|v\|_2$: scale и direction оптимизируются отдельно. Здесь нормируются parameters, а не batch activations. Weight Standardization центрирует и масштабирует coefficients каждого filter, например по input-channel и kernel axes. Его часто сочетают с activation normalization; WS и GN решают разные части преобразования.

Нормирование weights не гарантирует, что вся network имеет Lipschitz constant 1. Для этого нужны bounds operator norms и учёт nonlinearities, residual sums и других layers. Spectral norm разобрана в следующей главе.

## 5. Контроль и практика

1. Почему accumulation не увеличивает BN batch автоматически? **Ответ:** statistics считаются при каждом forward microbatch отдельно.
2. Равна ли RMSNorm LayerNorm без bias? **Ответ:** нет; RMSNorm также не вычитает mean.
3. Что может меняться при frozen weights? **Ответ:** state вроде BN running statistics, если оставлен training mode.

**Практика:** на shape $2\times4\times2\times2$ выпишите axes для BN, IN, GN с двумя groups и LN с двумя разными `normalized_shape`. Вручную нормируйте приведённые pixels. Проверьте train/eval outputs и запишите, какие statistics сохраняются в checkpoint.

<!-- source-depth-regularization-Д -->

## 6. Оси, число статистик и affine-параметры

### 6.1. Один shape, четыре разных операции

Для shape `(N,C,H,W)=(2,4,2,2)` всего 32 элемента. BN создаёт четыре набора статистик, по одному на channel; в каждый входят N·H·W=8 значений. IN создаёт восемь наборов, по одному для каждой пары example/channel; в каждом H·W=4 значения. GN с двумя группами каналов создаёт четыре набора, по два на example; в каждом два канала и четыре spatial позиции, всего восемь значений.

LN с `normalized_shape=(4,2,2)` создаёт два набора по 16 значений, а LN с `(2,2)` — восемь наборов по четыре значения. Однако одинаковые множества статистик ещё не делают слои одинаковыми. Обычная IN с affine имеет scale/bias на channel, а LN по `(H,W)` — параметры по spatial positions, разделяемые между остальными ведущими осями. GN с одной группой совпадает с LN по `(C,H,W)` в нормирующей части, но стандартные affine-параметры у них обычно разной формы.

Для Transformer `(B,T,D)=(2,3,4)` обычная LN по D создаёт шесть независимых наборов статистик по четыре features. Scale и bias длины D применяются к каждому токену, то есть не требуют отдельного параметра на каждую позицию T. Поэтому увеличение длины контекста не должно само по себе менять число LN-параметров в такой архитектуре. Конкретные формы и axes проверяют в [LayerNorm API](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) и [GroupNorm API](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html).

### 6.2. Нормирование не сохраняет всю информацию входа

У IN пары pixels (1,3) и (5,7) превращаются в одинаковые (−1,1), если для арифметики убрать epsilon и affine. Общий уровень яркости больше нельзя восстановить из нормированного вектора. Это может быть полезной инвариантностью для одной задачи и потерей сигнала для другой. Affine-параметры возвращают обучаемую шкалу слоя, но не индивидуально потерянное среднее каждого входа.

При variance v и стабилизаторе epsilon variance нормированного вектора до affine равна $v/(v+\epsilon)$. Например, при v=5 и epsilon=1 получается 5/6, а не 1. После scale gamma она дополнительно умножается на gamma². Постоянный вектор после центрирования становится нулём до affine; epsilon предотвращает деление на ноль, но не создаёт различий между равными координатами.

### 6.3. Почему gradient зависит от соседей в нормируемом наборе

Обозначим $\widehat x_i=(x_i-\mu)/s$, $s=\sqrt{v+\epsilon}$ и $a_i=\partial L/\partial\widehat x_i$, уже с учётом affine. Для m элементов одного набора стандартная производная имеет вид

$$
\frac{\partial L}{\partial x_i}=
\frac1s\left[a_i-\frac1m\sum_j a_j
-\widehat x_i\frac1m\sum_j a_j\widehat x_j\right].
$$

Изменение одного x меняет mean и variance, поэтому влияет не только его собственную координату. В BN coupling проходит между examples одного batch и spatial positions канала; в token-wise LN — только между features этого токена. При inference с фиксированными running statistics BN становится обычным affine-преобразованием и имеет другой путь производной.

Размер «batch» для статистик определяется фактическим reduction set. Если N=1, но H·W много, BN ещё может иметь несколько элементов на channel. Если reduction set содержит один элемент, оценка variance вырождается и многие training-реализации не допускают такой вход. Большое число pixels при этом не равно большому числу независимых изображений: соседние pixels и кадры могут сильно коррелировать.

## 7. BatchNorm: state, режим и преобразование при inference

### 7.1. Два estimator variance в одном слое

В обычном PyTorch BN train forward использует variance с делением на m, а running variance обновляется с поправкой m/(m−1) для соответствующей оценки. Это конкретный библиотечный контракт, который следует проверять в своей версии; в названии «дисперсия batch» эти детали не видны. Running statistics с momentum rho обновляются как `(1−rho)·old + rho·new`, что отличается от распространённого смысла momentum optimizer.

Для четырёх значений (1,3,5,7) mean=4, сумма квадратов отклонений=20. Variance train forward равна 5, исправленная — 20/3. Если old running mean=0, variance=1 и rho=0.1, после обновления получаем mean=0.4 и variance=$0.9+0.1\cdot20/3\approx1.5667$. Это ещё не population statistics: один batch только частично изменил начальное состояние. Поэтому сразу переключённый eval не обязан воспроизводить train output. Детали явно описаны в [BatchNorm2d API](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).

Если tracking running statistics выключен, соответствующий BN может использовать текущие batch statistics и на eval. Следовательно, одного вызова eval недостаточно, чтобы утверждать независимость output от состава batch: нужна проверка config. И наоборот, запрет gradients у gamma/beta не отключает обновление running buffers, если слой остался в training mode.

### 7.2. Accumulation, SyncBN и frozen backbone

Два microbatches (1,3) и (5,7) по отдельности имеют means 2 и 6, variances 1 и 1. Если нормировать их отдельно, оба дадут (−1,1). Общий batch использовал бы mean=4, variance=5 и дал $(−3,−1,1,3)/\sqrt5$. Накопление gradients после отдельных forwards не возвращает общий forward и не превращает эти статистики в одно вычисление.

SyncBatchNorm объединяет статистики между выбранными ranks одного forward. Это может увеличить reduction set, но не объединяет автоматически разные моменты накопления и не делает связанные observations независимыми. Его communication cost и назначение группы синхронизации входят в протокол обучения.

У frozen backbone отдельно выбирают обучаемость affine-параметров и режим buffers. Иногда сохраняют pretrained running statistics, иногда адаптируют их на train нового домена. Оба решения могут иметь смысл, но validation/test не должны незаметно участвовать в адаптации. Рекурсивное переключение всей модели в train может переопределить ранее выставленный режим отдельных BN-слоёв; проверяют фактические flags и сохранённое состояние.

### 7.3. Folding BN в линейную операцию

При фиксированных running mean и variance обозначим $a=\gamma/\sqrt{v_{run}+\epsilon}$. Для channel линейного или convolutional слоя $z=Wx+b$ BN после него даёт

$$
y=a(Wx+b-\mu_{run})+\beta
=(aW)x+[a(b-\mu_{run})+\beta].
$$

Поэтому при подходящей структуре inference можно объединить операции, получив новые веса и bias. Учебный scalar пример: W=2, b=1, mean=3, variance=4, gamma=2, beta=0.5 и epsilon=0. Тогда a=1, folded W=2, bias=−1.5. При x=2 оба пути дают 2.5. Это алгебра фиксированного eval-преобразования, а не замена training BN с зависящими от batch статистиками.

## 8. Batch Renorm и Transformer-нормализации

### 8.1. Зачем в Batch Renorm нужны stop-gradient corrections

Без clipping corrections r и d из раздела 3 алгебраически переводят batch-normalized activation к running-normalized: $r(x-\mu_B)/\sigma_B+d=(x-\mu)/\sigma$. Например, batch mean=2, std=1, running mean=0, std=2 дают r=0.5, d=1. Для x=(1,3) результат равен (0.5,1.5).

При d_max=0.5 и r_max=2 r остаётся 0.5, а d ограничивается 0.5; результат станет (0,1). Это уже не точное running normalization. Clipping и schedule управляют переходом, а stop-gradient r/d сохраняет особую training-производную. Если дифференцировать через r/d как через обычные batch-функции, алгебраическое сокращение меняет gradient-механизм. Подробная постановка дана в [Batch Renormalization](https://arxiv.org/abs/1702.03275).

### 8.2. LN и RMSNorm: разные инвариантности

Для x=(1,3) LN до affine при epsilon=0 выдаёт (−1,1), RMSNorm — $(1,3)/\sqrt5$. Для x+10=(11,13) LN даёт те же (−1,1), а RMSNorm делит (11,13) на $\sqrt{145}$ и даёт другой вектор. Центрирование удаляет общий сдвиг, RMS-масштабирование его сохраняет в изменённом направлении нормированного вектора.

При положительном масштабировании x на c обе нормировки без epsilon имеют соответствующую scale invariance до affine. Ненулевой epsilon нарушает точное равенство при изменении масштаба, особенно около нуля. RMSNorm не следует описывать как LN только без bias: исчезло именно вычитание среднего. Его свойства и мотивация исследованы в [RMSNorm paper](https://arxiv.org/abs/1910.07467).

В S2 RMSNorm названа упрощённой и более быстрой версией LN. Это полезная мотивация, но конкретное ускорение зависит от kernel, устройства, shape и всего block. Сопоставимость качества тоже относится к training recipe; архитектурную замену нельзя считать автоматически безвредной только из-за меньшего числа операций.

### 8.3. Pre-Norm, Post-Norm и путь градиента

Для одного sublayer Pre-Norm имеет вид $y=x+F(N(x))$, Post-Norm — $y=N(x+F(x))$. Их Jacobians различаются:

$$
J_{pre}=I+J_F(N(x))J_N(x),\qquad
J_{post}=J_N(x+F(x))[I+J_F(x)].
$$

В Pre-Norm есть явный identity-term вокруг sublayer; в Post-Norm итоговый gradient проходит также через внешнюю нормализацию. Это помогает понять разные режимы оптимизации глубоких Transformer, но не доказывает, что один вариант лучше при всех размерах и инициализациях. Финальная norm, residual scales, warmup и параметры архитектуры тоже важны.

Token-wise LN по D не смешивает будущие временные позиции: attention остаётся отдельным механизмом обмена между токенами. Если ошибочно нормировать по `(T,D)` в causal LM, representation раннего токена начнёт зависеть от будущих значений через mean/variance. Causal mask последующего attention эту уже возникшую зависимость не устраняет. Для encoder-классификации полная последовательность может быть законным входом, но это другая задача и другая схема нормировки.

## 9. Нормировать weights — другая операция

Weight Norm с v=(3,4), g=2 даёт w=(1.2,1.6), norm=2. Параметры описывают scale и direction, но не принуждают activations иметь zero mean или unit variance. Для нескольких filters ограничение norm каждого row тоже не равно ограничению operator norm всей матрицы: две одинаковые единичные rows создают усиление по общему направлению.

Weight Standardization для filter (1,3) с mean=2 и std=1 даёт (−1,1), norm=$\sqrt2$. Нормирование распределения coefficients не означает unit spectral norm. Оно меняет параметризацию вычисления и может сочетаться с GN/другой activation norm. Предложенные режимы microbatch training разобраны в [Weight Standardization paper](https://arxiv.org/abs/1903.10520); качество и скорость проверяют для конкретного применения.

Нормализация activations также не является автоматическим robustness certificate. При малой variance derivative содержит $1/\sqrt{v+\epsilon}$, а learned affine scale может усиливать его. Даже после ограничений отдельных линейных слоёв надо учесть nonlinearities, residual sums и normalizations всей системы.

## 10. Контроль и практика

1. **Почему GN с одной группой и LN по C,H,W не всегда один слой?** Нормирующие множества могут совпасть, а формы affine-параметров различаться.
2. **Почему frozen weights ещё не означают frozen модель?** Running buffers могут изменяться без gradients параметров.
3. **Почему BN folding требует eval state?** Только фиксированные статистики дают постоянные новые W и bias.
4. **Почему неверные LN-оси опасны для causal Transformer?** Статистики по времени могут перенести информацию будущих токенов до causal attention.

Практика: для shape 2×4×2×2 перечислить reduction sets и формы affine. Проверить train/eval по одному channel, отдельно заморозить параметры и buffers. Для одного Transformer-block выписать normalized dimensions и residual order. Числа главы учебные; реальное ускорение или обучение пользовательской модели не измерялось.
