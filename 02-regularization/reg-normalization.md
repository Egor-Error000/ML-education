# Нормализации и состояния train/eval

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** По каким осям вычисляются статистики и какие из них сохраняются?





## Карта вычисления и границы этапов

![Нормализации и состояния train/eval: входы, преобразования и выходы](<../оформление/course/reg-normalization.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:41c1d64f5a26fd81a9 -->
<a id="u-41c1d64f5a26fd81a9"></a>

## Оси, статистики и affine-параметры

<!-- /unit:41c1d64f5a26fd81a9 -->

<!-- unit:c6f5d95a9bd6d0cb94 -->
<a id="u-c6f5d95a9bd6d0cb94"></a>

Для выбранного множества элементов $\mathcal S$:

<!-- /unit:c6f5d95a9bd6d0cb94 -->

<!-- unit:e6c8e4ecc9a3f42e58 -->
<a id="u-e6c8e4ecc9a3f42e58"></a>

$$
\mu=\frac1{|\mathcal S|}\sum_{j\in\mathcal S}x_j,
\quad
v=\frac1{|\mathcal S|}\sum_{j\in\mathcal S}(x_j-\mu)^2,
\quad
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+\beta_i.
$$

<!-- /unit:e6c8e4ecc9a3f42e58 -->

<!-- unit:9712f1d5e85aee20cf -->
<a id="u-9712f1d5e85aee20cf"></a>

Главное различие — какие элементы принадлежат $\mathcal S$ и как разделяются affine parameters. $\epsilon>0$ стабилизирует деление, поэтому variance нормированного output не обязана быть точно 1. Affine layer может затем намеренно изменить mean и scale.

<!-- /unit:9712f1d5e85aee20cf -->

<!-- unit:6e5c7136ecf3df9eb3 -->
<a id="u-6e5c7136ecf3df9eb3"></a>

Для тензора $N,C,H,W$ **BatchNorm** считает statistics по $N,H,W$ отдельно для каждого channel. **InstanceNorm** — по $H,W$ для каждого example/channel. **GroupNorm** — по пространству и channels выбранной группы внутри одного example. **LayerNorm** — по объявленным normalized dimensions; в Transformer $N,T,D$ обычно только по $D$ отдельно для каждой пары example/token. Нельзя без config считать, что LN всегда нормирует всё изображение.

<!-- /unit:6e5c7136ecf3df9eb3 -->

<!-- unit:35f24e272ec743d958 -->
<a id="u-35f24e272ec743d958"></a>

Пример с одним channel и двумя изображениями, каждый из двух pixels: $(1,3)$ и $(5,7)$. BN использует все четыре значения, mean 4 и variance 5. IN для первого использует mean 2, variance 1, для второго mean 6, variance 1. Без epsilon/affine IN даёт $(-1,1)$ в обоих examples, стирая различие общей яркости. BN сохраняет его в normalized values. Это показывает, почему выбор norm меняет доступную информацию.

<!-- /unit:35f24e272ec743d958 -->

<!-- unit:94863f275d28317c8f -->
<a id="u-94863f275d28317c8f"></a>

### Один shape, четыре разных операции

<!-- /unit:94863f275d28317c8f -->

<!-- unit:b3e42da51e4308900d -->
<a id="u-b3e42da51e4308900d"></a>

Для shape `(N,C,H,W)=(2,4,2,2)` всего 32 элемента. BN создаёт четыре набора статистик, по одному на channel; в каждый входят N·H·W=8 значений. IN создаёт восемь наборов, по одному для каждой пары example/channel; в каждом H·W=4 значения. GN с двумя группами каналов создаёт четыре набора, по два на example; в каждом два канала и четыре spatial позиции, всего восемь значений.

<!-- /unit:b3e42da51e4308900d -->

<!-- unit:6e585c0b7cbbb234ba -->
<a id="u-6e585c0b7cbbb234ba"></a>

LN с `normalized_shape=(4,2,2)` создаёт два набора по 16 значений, а LN с `(2,2)` — восемь наборов по четыре значения. Однако одинаковые множества статистик ещё не делают слои одинаковыми. Обычная IN с affine имеет scale/bias на channel, а LN по `(H,W)` — параметры по spatial positions, разделяемые между остальными ведущими осями. GN с одной группой совпадает с LN по `(C,H,W)` в нормирующей части, но стандартные affine-параметры у них обычно разной формы.

<!-- /unit:6e585c0b7cbbb234ba -->

<!-- unit:0b92369b9a9f16fad9 -->
<a id="u-0b92369b9a9f16fad9"></a>

Для Transformer `(B,T,D)=(2,3,4)` обычная LN по D создаёт шесть независимых наборов статистик по четыре features. Scale и bias длины D применяются к каждому токену, то есть не требуют отдельного параметра на каждую позицию T. Поэтому увеличение длины контекста не должно само по себе менять число LN-параметров в такой архитектуре. Конкретные формы и axes проверяют в [LayerNorm API](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) и [GroupNorm API](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html).

<!-- /unit:0b92369b9a9f16fad9 -->

<!-- unit:ab861ca8c43a377e3f -->
<a id="u-ab861ca8c43a377e3f"></a>

### Нормирование не сохраняет всю информацию входа

<!-- /unit:ab861ca8c43a377e3f -->

<!-- unit:9e26129001dfb9130c -->
<a id="u-9e26129001dfb9130c"></a>

У IN пары pixels (1,3) и (5,7) превращаются в одинаковые (−1,1), если для арифметики убрать epsilon и affine. Общий уровень яркости больше нельзя восстановить из нормированного вектора. Это может быть полезной инвариантностью для одной задачи и потерей сигнала для другой. Affine-параметры возвращают обучаемую шкалу слоя, но не индивидуально потерянное среднее каждого входа.

<!-- /unit:9e26129001dfb9130c -->

<!-- unit:6844e5b79c5e34d7f9 -->
<a id="u-6844e5b79c5e34d7f9"></a>

При variance v и стабилизаторе epsilon variance нормированного вектора до affine равна $v/(v+\epsilon)$. Например, при v=5 и epsilon=1 получается 5/6, а не 1. После scale gamma она дополнительно умножается на gamma². Постоянный вектор после центрирования становится нулём до affine; epsilon предотвращает деление на ноль, но не создаёт различий между равными координатами.

<!-- /unit:6844e5b79c5e34d7f9 -->

<!-- unit:a00423a22124d92115 -->
<a id="u-a00423a22124d92115"></a>

### Почему gradient зависит от соседей в нормируемом наборе

<!-- /unit:a00423a22124d92115 -->

<!-- unit:3c65b08624ae0c84aa -->
<a id="u-3c65b08624ae0c84aa"></a>

Обозначим $\widehat x_i=(x_i-\mu)/s$, $s=\sqrt{v+\epsilon}$ и $a_i=\partial L/\partial\widehat x_i$, уже с учётом affine. Для m элементов одного набора стандартная производная имеет вид

<!-- /unit:3c65b08624ae0c84aa -->

<!-- unit:febed21aadcca51913 -->
<a id="u-febed21aadcca51913"></a>

$$
\frac{\partial L}{\partial x_i}=
\frac1s\left[a_i-\frac1m\sum_j a_j
-\widehat x_i\frac1m\sum_j a_j\widehat x_j\right].
$$

<!-- /unit:febed21aadcca51913 -->

<!-- unit:23dbe1893e58c4b028 -->
<a id="u-23dbe1893e58c4b028"></a>

Изменение одного x меняет mean и variance, поэтому влияет не только его собственную координату. В BN coupling проходит между examples одного batch и spatial positions канала; в token-wise LN — только между features этого токена. При inference с фиксированными running statistics BN становится обычным affine-преобразованием и имеет другой путь производной.

<!-- /unit:23dbe1893e58c4b028 -->

<!-- unit:7ac6faa56f1826e444 -->
<a id="u-7ac6faa56f1826e444"></a>

Размер «batch» для статистик определяется фактическим reduction set. Если N=1, но H·W много, BN ещё может иметь несколько элементов на channel. Если reduction set содержит один элемент, оценка variance вырождается и многие training-реализации не допускают такой вход. Большое число pixels при этом не равно большому числу независимых изображений: соседние pixels и кадры могут сильно коррелировать.

<!-- /unit:7ac6faa56f1826e444 -->

<!-- unit:1305abeec5b239b318 -->
<a id="u-1305abeec5b239b318"></a>

## BatchNorm: состояние и inference

<!-- /unit:1305abeec5b239b318 -->

<!-- unit:1ae3c79171d6ba176a -->
<a id="u-1ae3c79171d6ba176a"></a>

На training BN использует mini-batch statistics и обновляет running estimates. На eval типичный режим использует накопленные estimates, а не statistics текущего входного batch. Конкретные variance estimators и momentum convention зависят от library; не переносите число momentum из optimizer в BN по сходству названий.

<!-- /unit:1ae3c79171d6ba176a -->

<!-- unit:b688bc725007868f96 -->
<a id="u-b688bc725007868f96"></a>

Малый или сильно correlated batch даёт шумные statistics. Gradient accumulation не эквивалентен большому BN batch: несколько forward по microbatches не объединяют их statistics автоматически. SyncBatchNorm может синхронизировать statistics между devices, но не решает все проблемы не-i.i.d. data и имеет communication cost.

<!-- /unit:b688bc725007868f96 -->

<!-- unit:67e85a929f6e520565 -->
<a id="u-67e85a929f6e520565"></a>

Шум batch statistics способен давать regularizing effect, но историческое объяснение через internal covariate shift не является единственным установленным механизмом. Исследуется также сглаживание optimization landscape; см. [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604). Нельзя считать BN простым заменителем L2 или dropout.

<!-- /unit:67e85a929f6e520565 -->

<!-- unit:ab1b23767c1b64fdf0 -->
<a id="u-ab1b23767c1b64fdf0"></a>

После freezing backbone нужно отдельно решить режим BN. Замороженные weights не обязательно означают замороженные running statistics. При fine-tuning с маленьким domain-specific batch их неосторожное обновление может ухудшить pretrained representation.

<!-- /unit:ab1b23767c1b64fdf0 -->

<!-- unit:86ae28d6fd5754dc32 -->
<a id="u-86ae28d6fd5754dc32"></a>

### Два estimator variance в одном слое

<!-- /unit:86ae28d6fd5754dc32 -->

<!-- unit:13aa06d89ded94f198 -->
<a id="u-13aa06d89ded94f198"></a>

В обычном PyTorch BN train forward использует variance с делением на m, а running variance обновляется с поправкой m/(m−1) для соответствующей оценки. Это конкретный библиотечный контракт, который следует проверять в своей версии; в названии «дисперсия batch» эти детали не видны. Running statistics с momentum rho обновляются как `(1−rho)·old + rho·new`, что отличается от распространённого смысла momentum optimizer.

<!-- /unit:13aa06d89ded94f198 -->

<!-- unit:492e5df075cd975515 -->
<a id="u-492e5df075cd975515"></a>

Для четырёх значений (1,3,5,7) mean=4, сумма квадратов отклонений=20. Variance train forward равна 5, исправленная — 20/3. Если old running mean=0, variance=1 и rho=0.1, после обновления получаем mean=0.4 и variance=$0.9+0.1\cdot20/3\approx1.5667$. Это ещё не population statistics: один batch только частично изменил начальное состояние. Поэтому сразу переключённый eval не обязан воспроизводить train output. Детали явно описаны в [BatchNorm2d API](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).

<!-- /unit:492e5df075cd975515 -->

<!-- unit:90efe345b8b5d35c02 -->
<a id="u-90efe345b8b5d35c02"></a>

Если tracking running statistics выключен, соответствующий BN может использовать текущие batch statistics и на eval. Следовательно, одного вызова eval недостаточно, чтобы утверждать независимость output от состава batch: нужна проверка config. И наоборот, запрет gradients у gamma/beta не отключает обновление running buffers, если слой остался в training mode.

<!-- /unit:90efe345b8b5d35c02 -->

<!-- unit:240c899dde9e182fce -->
<a id="u-240c899dde9e182fce"></a>

### Accumulation, SyncBN и frozen backbone

<!-- /unit:240c899dde9e182fce -->

<!-- unit:26076451449a79cac3 -->
<a id="u-26076451449a79cac3"></a>

Два microbatches (1,3) и (5,7) по отдельности имеют means 2 и 6, variances 1 и 1. Если нормировать их отдельно, оба дадут (−1,1). Общий batch использовал бы mean=4, variance=5 и дал $(−3,−1,1,3)/\sqrt5$. Накопление gradients после отдельных forwards не возвращает общий forward и не превращает эти статистики в одно вычисление.

<!-- /unit:26076451449a79cac3 -->

<!-- unit:5e42344f88c4618aa5 -->
<a id="u-5e42344f88c4618aa5"></a>

SyncBatchNorm объединяет статистики между выбранными ranks одного forward. Это может увеличить reduction set, но не объединяет автоматически разные моменты накопления и не делает связанные observations независимыми. Его communication cost и назначение группы синхронизации входят в протокол обучения.

<!-- /unit:5e42344f88c4618aa5 -->

<!-- unit:96d21716c30a06927d -->
<a id="u-96d21716c30a06927d"></a>

У frozen backbone отдельно выбирают обучаемость affine-параметров и режим buffers. Иногда сохраняют pretrained running statistics, иногда адаптируют их на train нового домена. Оба решения могут иметь смысл, но validation/test не должны незаметно участвовать в адаптации. Рекурсивное переключение всей модели в train может переопределить ранее выставленный режим отдельных BN-слоёв; проверяют фактические flags и сохранённое состояние.

<!-- /unit:96d21716c30a06927d -->

<!-- unit:f31396f5d756c10107 -->
<a id="u-f31396f5d756c10107"></a>

### Folding BN в линейную операцию

<!-- /unit:f31396f5d756c10107 -->

<!-- unit:0f8d1adb44a4db2a35 -->
<a id="u-0f8d1adb44a4db2a35"></a>

При фиксированных running mean и variance обозначим $a=\gamma/\sqrt{v_{run}+\epsilon}$. Для channel линейного или convolutional слоя $z=Wx+b$ BN после него даёт

<!-- /unit:0f8d1adb44a4db2a35 -->

<!-- unit:8787a82ea9cdfd7153 -->
<a id="u-8787a82ea9cdfd7153"></a>

$$
y=a(Wx+b-\mu_{run})+\beta
=(aW)x+[a(b-\mu_{run})+\beta].
$$

<!-- /unit:8787a82ea9cdfd7153 -->

<!-- unit:ac8b6db42922a3955e -->
<a id="u-ac8b6db42922a3955e"></a>

Поэтому при подходящей структуре inference можно объединить операции, получив новые веса и bias. Учебный scalar пример: W=2, b=1, mean=3, variance=4, gamma=2, beta=0.5 и epsilon=0. Тогда a=1, folded W=2, bias=−1.5. При x=2 оба пути дают 2.5. Это алгебра фиксированного eval-преобразования, а не замена training BN с зависящими от batch статистиками.

<!-- /unit:ac8b6db42922a3955e -->

<!-- unit:468fff67c6b6fd0418 -->
<a id="u-468fff67c6b6fd0418"></a>

## Batch Renorm, RMSNorm и Transformer

<!-- /unit:468fff67c6b6fd0418 -->

<!-- unit:9ef9966bbb944de0f1 -->
<a id="u-9ef9966bbb944de0f1"></a>

Batch Renorm вводит corrections между текущими batch statistics и running estimates. Упрощённо:

<!-- /unit:9ef9966bbb944de0f1 -->

<!-- unit:0d0a227435a04c27f5 -->
<a id="u-0d0a227435a04c27f5"></a>

$$
\widehat x_i=\frac{x_i-\mu_B}{\sigma_B},\quad
r=\operatorname{clip}\left(\frac{\sigma_B}{\sigma},\frac1{r_{\max}},r_{\max}\right),
\quad
d=\operatorname{clip}\left(\frac{\mu_B-\mu}{\sigma},-d_{\max},d_{\max}\right),
$$
$$
y_i=\gamma(r\widehat x_i+d)+\beta.
$$

<!-- /unit:0d0a227435a04c27f5 -->

<!-- unit:fccdad09cc20a2ae3b -->
<a id="u-fccdad09cc20a2ae3b"></a>

Здесь standard deviations включают stabilizer по реализации, а $r,d$ используются с остановленным gradient; clipping limits могут меняться по schedule. Это не просто замена batch mean running mean во всех derivatives. Идея — уменьшить mismatch training/inference при сохранении управляемой оптимизации. [Batch Renormalization](https://arxiv.org/abs/1702.03275).

<!-- /unit:fccdad09cc20a2ae3b -->

<!-- unit:556112f65f5728b27d -->
<a id="u-556112f65f5728b27d"></a>

RMSNorm не центрирует:

<!-- /unit:556112f65f5728b27d -->

<!-- unit:dfc350227bf9c2640b -->
<a id="u-dfc350227bf9c2640b"></a>

$$
\operatorname{RMS}(x)=\sqrt{\frac1D\sum_{j=1}^D x_j^2+\epsilon},
\qquad
y_i=\gamma_i\frac{x_i}{\operatorname{RMS}(x)}.
$$

<!-- /unit:dfc350227bf9c2640b -->

<!-- unit:2dcb5239388de50f65 -->
<a id="u-2dcb5239388de50f65"></a>

Для $(1,3)$ без epsilon LN даёт $(-1,1)$, RMSNorm — $(1/\sqrt5,3/\sqrt5)\approx(0.4472,1.3416)$. Mean не удалена. Это изменение преобразования, а не лишь сокращённая запись LN. Применимость оценивается для конкретной architecture/training recipe.

<!-- /unit:2dcb5239388de50f65 -->

<!-- unit:8b00b066b62e182ed8 -->
<a id="u-8b00b066b62e182ed8"></a>

Weight Normalization параметризует vector weights как $w=g\,v/\|v\|_2$: scale и direction оптимизируются отдельно. Здесь нормируются parameters, а не batch activations. Weight Standardization центрирует и масштабирует coefficients каждого filter, например по input-channel и kernel axes. Его часто сочетают с activation normalization; WS и GN решают разные части преобразования.

<!-- /unit:8b00b066b62e182ed8 -->

<!-- unit:bda15885abdbcb3514 -->
<a id="u-bda15885abdbcb3514"></a>

Нормирование weights не гарантирует, что вся network имеет Lipschitz constant 1. Для этого нужны bounds operator norms и учёт nonlinearities, residual sums и других layers. Spectral norm разобрана в следующей главе.

<!-- /unit:bda15885abdbcb3514 -->

<!-- unit:8f11d67db8e0901ae9 -->
<a id="u-8f11d67db8e0901ae9"></a>

### Зачем в Batch Renorm нужны stop-gradient corrections

<!-- /unit:8f11d67db8e0901ae9 -->

<!-- unit:4cdaa42e8b4f51d88f -->
<a id="u-4cdaa42e8b4f51d88f"></a>

Без clipping corrections r и d из раздела 3 алгебраически переводят batch-normalized activation к running-normalized: $r(x-\mu_B)/\sigma_B+d=(x-\mu)/\sigma$. Например, batch mean=2, std=1, running mean=0, std=2 дают r=0.5, d=1. Для x=(1,3) результат равен (0.5,1.5).

<!-- /unit:4cdaa42e8b4f51d88f -->

<!-- unit:a7b0ceeeb3646102b9 -->
<a id="u-a7b0ceeeb3646102b9"></a>

При d_max=0.5 и r_max=2 r остаётся 0.5, а d ограничивается 0.5; результат станет (0,1). Это уже не точное running normalization. Clipping и schedule управляют переходом, а stop-gradient r/d сохраняет особую training-производную. Если дифференцировать через r/d как через обычные batch-функции, алгебраическое сокращение меняет gradient-механизм. Подробная постановка дана в [Batch Renormalization](https://arxiv.org/abs/1702.03275).

<!-- /unit:a7b0ceeeb3646102b9 -->

<!-- unit:0ff368e11b74f5a4b5 -->
<a id="u-0ff368e11b74f5a4b5"></a>

### LN и RMSNorm: разные инвариантности

<!-- /unit:0ff368e11b74f5a4b5 -->

<!-- unit:4f17dccfffa1257952 -->
<a id="u-4f17dccfffa1257952"></a>

Для x=(1,3) LN до affine при epsilon=0 выдаёт (−1,1), RMSNorm — $(1,3)/\sqrt5$. Для x+10=(11,13) LN даёт те же (−1,1), а RMSNorm делит (11,13) на $\sqrt{145}$ и даёт другой вектор. Центрирование удаляет общий сдвиг, RMS-масштабирование его сохраняет в изменённом направлении нормированного вектора.

<!-- /unit:4f17dccfffa1257952 -->

<!-- unit:2e1151e8809bae6417 -->
<a id="u-2e1151e8809bae6417"></a>

При положительном масштабировании x на c обе нормировки без epsilon имеют соответствующую scale invariance до affine. Ненулевой epsilon нарушает точное равенство при изменении масштаба, особенно около нуля. RMSNorm не следует описывать как LN только без bias: исчезло именно вычитание среднего. Его свойства и мотивация исследованы в [RMSNorm paper](https://arxiv.org/abs/1910.07467).

<!-- /unit:2e1151e8809bae6417 -->

<!-- unit:feaf856df07e51997d -->
<a id="u-feaf856df07e51997d"></a>

В S2 RMSNorm названа упрощённой и более быстрой версией LN. Это полезная мотивация, но конкретное ускорение зависит от kernel, устройства, shape и всего block. Сопоставимость качества тоже относится к training recipe; архитектурную замену нельзя считать автоматически безвредной только из-за меньшего числа операций.

<!-- /unit:feaf856df07e51997d -->

<!-- unit:0adbff806b04945228 -->
<a id="u-0adbff806b04945228"></a>

### Pre-Norm, Post-Norm и путь градиента

<!-- /unit:0adbff806b04945228 -->

<!-- unit:6da202ed5e73d52553 -->
<a id="u-6da202ed5e73d52553"></a>

Для одного sublayer Pre-Norm имеет вид $y=x+F(N(x))$, Post-Norm — $y=N(x+F(x))$. Их Jacobians различаются:

<!-- /unit:6da202ed5e73d52553 -->

<!-- unit:c54b75e67c317c6813 -->
<a id="u-c54b75e67c317c6813"></a>

$$
J_{pre}=I+J_F(N(x))J_N(x),\qquad
J_{post}=J_N(x+F(x))[I+J_F(x)].
$$

<!-- /unit:c54b75e67c317c6813 -->

<!-- unit:a9e776a8906b2dcd2e -->
<a id="u-a9e776a8906b2dcd2e"></a>

В Pre-Norm есть явный identity-term вокруг sublayer; в Post-Norm итоговый gradient проходит также через внешнюю нормализацию. Это помогает понять разные режимы оптимизации глубоких Transformer, но не доказывает, что один вариант лучше при всех размерах и инициализациях. Финальная norm, residual scales, warmup и параметры архитектуры тоже важны.

<!-- /unit:a9e776a8906b2dcd2e -->

<!-- unit:a96cb3d4abdc4444a5 -->
<a id="u-a96cb3d4abdc4444a5"></a>

Token-wise LN по D не смешивает будущие временные позиции: attention остаётся отдельным механизмом обмена между токенами. Если ошибочно нормировать по `(T,D)` в causal LM, representation раннего токена начнёт зависеть от будущих значений через mean/variance. Causal mask последующего attention эту уже возникшую зависимость не устраняет. Для encoder-классификации полная последовательность может быть законным входом, но это другая задача и другая схема нормировки.

<!-- /unit:a96cb3d4abdc4444a5 -->

<!-- unit:234b3b297c0fce2294 -->
<a id="u-234b3b297c0fce2294"></a>

## Нормализация параметров

<!-- /unit:234b3b297c0fce2294 -->

<!-- unit:2c2998e7c7efea45a5 -->
<a id="u-2c2998e7c7efea45a5"></a>

Weight Norm с v=(3,4), g=2 даёт w=(1.2,1.6), norm=2. Параметры описывают scale и direction, но не принуждают activations иметь zero mean или unit variance. Для нескольких filters ограничение norm каждого row тоже не равно ограничению operator norm всей матрицы: две одинаковые единичные rows создают усиление по общему направлению.

<!-- /unit:2c2998e7c7efea45a5 -->

<!-- unit:0993cd8a6b9ba2e851 -->
<a id="u-0993cd8a6b9ba2e851"></a>

Weight Standardization для filter (1,3) с mean=2 и std=1 даёт (−1,1), norm=$\sqrt2$. Нормирование распределения coefficients не означает unit spectral norm. Оно меняет параметризацию вычисления и может сочетаться с GN/другой activation norm. Предложенные режимы microbatch training разобраны в [Weight Standardization paper](https://arxiv.org/abs/1903.10520); качество и скорость проверяют для конкретного применения.

<!-- /unit:0993cd8a6b9ba2e851 -->

<!-- unit:595ceb1cda22edd74e -->
<a id="u-595ceb1cda22edd74e"></a>

Нормализация activations также не является автоматическим robustness certificate. При малой variance derivative содержит $1/\sqrt{v+\epsilon}$, а learned affine scale может усиливать его. Даже после ограничений отдельных линейных слоёв надо учесть nonlinearities, residual sums и normalizations всей системы.

<!-- /unit:595ceb1cda22edd74e -->

<!-- unit:802284160b552e28d1 -->
<a id="u-802284160b552e28d1"></a>

## Контроль и практика

<!-- /unit:802284160b552e28d1 -->

<!-- unit:c2dc085b0c239a643b -->
<a id="u-c2dc085b0c239a643b"></a>

1. Почему accumulation не увеличивает BN batch автоматически? **Ответ:** statistics считаются при каждом forward microbatch отдельно.
2. Равна ли RMSNorm LayerNorm без bias? **Ответ:** нет; RMSNorm также не вычитает mean.
3. Что может меняться при frozen weights? **Ответ:** state вроде BN running statistics, если оставлен training mode.

<!-- /unit:c2dc085b0c239a643b -->

<!-- unit:166942624412beb4a7 -->
<a id="u-166942624412beb4a7"></a>

**Практика:** на shape $2\times4\times2\times2$ выпишите axes для BN, IN, GN с двумя groups и LN с двумя разными `normalized_shape`. Вручную нормируйте приведённые pixels. Проверьте train/eval outputs и запишите, какие statistics сохраняются в checkpoint.

<!-- /unit:166942624412beb4a7 -->

<!-- unit:a9013e4adb026873e1 -->
<a id="u-a9013e4adb026873e1"></a>

<!-- source-depth-regularization-Д -->

<!-- /unit:a9013e4adb026873e1 -->

<!-- unit:e920fee7874f4c37ca -->
<a id="u-e920fee7874f4c37ca"></a>

1. **Почему GN с одной группой и LN по C,H,W не всегда один слой?** Нормирующие множества могут совпасть, а формы affine-параметров различаться.
2. **Почему frozen weights ещё не означают frozen модель?** Running buffers могут изменяться без gradients параметров.
3. **Почему BN folding требует eval state?** Только фиксированные статистики дают постоянные новые W и bias.
4. **Почему неверные LN-оси опасны для causal Transformer?** Статистики по времени могут перенести информацию будущих токенов до causal attention.

<!-- /unit:e920fee7874f4c37ca -->

<!-- unit:1c68813c73c9eb2d1f -->
<a id="u-1c68813c73c9eb2d1f"></a>

Практика: для shape 2×4×2×2 перечислить reduction sets и формы affine. Проверить train/eval по одному channel, отдельно заморозить параметры и buffers. Для одного Transformer-block выписать normalized dimensions и residual order. Числа главы учебные; реальное ускорение или обучение пользовательской модели не измерялось.

<!-- /unit:1c68813c73c9eb2d1f -->

## Результат главы и переход дальше

Ответьте на вопрос: **По каким осям вычисляются статистики и какие из них сохраняются?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
