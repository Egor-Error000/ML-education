# SAM, Lookahead и итоговый выбор метода

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>), [Штрафы, ограничения, L1/L2 и AdamW](<reg-penalties.md>), [Early stopping, learning rate, SWA и EMA](<reg-training.md>), [Dropout, DropConnect, DropBlock и случайные ветви](<reg-stochastic.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как проверить гипотезу о регуляризации при фиксированном бюджете настройки?





## Карта вычисления и границы этапов

![SAM, Lookahead и итоговый выбор метода: входы, преобразования и выходы](<../оформление/course/reg-selection.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:0a2d9643a03620f27e -->
<a id="u-0a2d9643a03620f27e"></a>

## Два шага SAM

<!-- /unit:0a2d9643a03620f27e -->

<!-- unit:bdcb4000bcf9a2bcc1 -->
<a id="u-bdcb4000bcf9a2bcc1"></a>

Sharpness-Aware Minimization ищет weights с низким худшим loss в выбранной local neighborhood:

<!-- /unit:bdcb4000bcf9a2bcc1 -->

<!-- unit:8947093da38451bfa3 -->
<a id="u-8947093da38451bfa3"></a>

$$
\min_w\max_{\|\varepsilon\|_2\le\rho}L(w+\varepsilon).
$$

<!-- /unit:8947093da38451bfa3 -->

<!-- unit:ca08e944f4b89d4299 -->
<a id="u-ca08e944f4b89d4299"></a>

Линеаризация inner objective даёт perturbation

<!-- /unit:ca08e944f4b89d4299 -->

<!-- unit:717af136f7455fdfc5 -->
<a id="u-717af136f7455fdfc5"></a>

$$
\widehat\varepsilon=
\rho\frac{\nabla L(w)}{\|\nabla L(w)\|_2},
\qquad
w^+=w-\eta\nabla L(w+\widehat\varepsilon).
$$

<!-- /unit:717af136f7455fdfc5 -->

<!-- unit:d6b3aedb626cb8c327 -->
<a id="u-d6b3aedb626cb8c327"></a>

В базовом приближённом update perturbation рассматривается как фиксированное смещение для второго gradient; это не полный differentiation через весь inner solver. При нулевой gradient norm нужно отдельное стабилизированное правило. Параметр $\rho$ задаёт radius, не learning rate. [SAM](https://arxiv.org/abs/2010.01412).

<!-- /unit:d6b3aedb626cb8c327 -->

<!-- unit:62e64d9af9310cdaa9 -->
<a id="u-62e64d9af9310cdaa9"></a>

Пример для $L(w)=w^2/2$, $w=2,\rho=0.1,\eta=0.1$: первый gradient 2, perturbation 0.1, второй gradient 2.1, update даёт $w^+=1.79$. Обычный GD дал бы 1.8. В этом scalar convex примере нет загадочной «магии flat basins»; видно, как меняется шаг. Для network эффект зависит от data и geometry.

<!-- /unit:62e64d9af9310cdaa9 -->

<!-- unit:01cf351bd93e43fcbd -->
<a id="u-01cf351bd93e43fcbd"></a>

SAM обычно требует дополнительного forward/backward и аккуратного управления stateful layers. Если BatchNorm statistics обновить дважды несогласованно, сравнение может смешать эффект optimizer и statistics. Конкретная реализация должна явно определить state handling. Сравнивать нужно и quality, и compute.

<!-- /unit:01cf351bd93e43fcbd -->

<!-- unit:658fa137da6f83f6a2 -->
<a id="u-658fa137da6f83f6a2"></a>

### Линеаризация внутреннего максимума

<!-- /unit:658fa137da6f83f6a2 -->

<!-- unit:2520a902be7ef54094 -->
<a id="u-2520a902be7ef54094"></a>

В базовом SAM ищется устойчивость loss к perturbation weights. Для малого epsilon $L(w+\epsilon)\approx L(w)+g^\top\epsilon$, где g=gradient L(w). На Euclidean ball radius rho максимизирующее направление совпадает с g и имеет длину rho. Это объясняет формулу раздела 1, но только для линейного приближения inner objective.

<!-- /unit:2520a902be7ef54094 -->

<!-- unit:ec8b0be5ce07432f8a -->
<a id="u-ec8b0be5ce07432f8a"></a>

При другом norm ball оптимальное линейное направление меняется; rho без norm не определяет задачу. В нелинейном loss curvature может сдвинуть настоящий worst-case point относительно normalized gradient. В частности, нулевой g не означает отсутствия роста loss вокруг w: линейное приближение просто не видит направление. Стабилизатор denominator нужен вычислительно, но не превращает approximate inner step в точный solver.

<!-- /unit:ec8b0be5ce07432f8a -->

<!-- unit:782327a3af9596fd0e -->
<a id="u-782327a3af9596fd0e"></a>

Adversarial training из главы [«Adversarial training, consistency и spectral norm»](<reg-robustness.md>) ищет perturbation входа x, SAM — параметров w. Оба используют min–max идею, однако пространства, размерности, units и смысл сохранения target различаются. Из успеха одного метода не следует robustness другого к тем же изменениям.

<!-- /unit:782327a3af9596fd0e -->

<!-- unit:42397bdea3a02d82d0 -->
<a id="u-42397bdea3a02d82d0"></a>

### Двумерный шаг с разной curvature

<!-- /unit:42397bdea3a02d82d0 -->

<!-- unit:9986a83b4409025841 -->
<a id="u-9986a83b4409025841"></a>

Возьмём $L(w)=\tfrac12(w_1^2+4w_2^2)$, w=(3,1), rho=0.1, eta=0.1. Первый gradient равен (3,4), norm=5. Approximate inner perturbation=(0.06,0.08), perturbed weights=(3.06,1.08). Второй gradient=(3.06,4.32), поэтому внешний update исходных weights даёт (2.694,0.568).

<!-- /unit:9986a83b4409025841 -->

<!-- unit:4c9ab5a470ed8db71f -->
<a id="u-4c9ab5a470ed8db71f"></a>

Обычный GD дал бы (2.7,0.6). Второе направление сильнее реагирует на perturbation из-за большего quadratic curvature. Этот учебный пример показывает конкретную механику; он не доказывает улучшение test quality. Исходный objective и приближение приведены в [SAM](https://arxiv.org/abs/2010.01412).

<!-- /unit:4c9ab5a470ed8db71f -->

<!-- unit:f12f3c7c65ae9a15b8 -->
<a id="u-f12f3c7c65ae9a15b8"></a>

### Правильный порядок операций и состояние optimizer

<!-- /unit:f12f3c7c65ae9a15b8 -->

<!-- unit:7645354993cb65e3ec -->
<a id="u-7645354993cb65e3ec"></a>

На исходных weights получают первый gradient, строят perturbation и временно прибавляют её. Первый gradient очищают из накопителей, если он не нужен для отдельно определённого правила. Затем получают второй gradient, восстанавливают исходные weights и один раз выполняют внешний optimizer update с этим gradient. В базовом приближении epsilon при втором differentiation рассматривается как фиксированная. Если смешать оба gradients без такого намерения, получится другой update.

<!-- /unit:7645354993cb65e3ec -->

<!-- unit:fe4ea68f0d0a2328db -->
<a id="u-fe4ea68f0d0a2328db"></a>

В scalar примере w=2, rho=0.1, eta=0.1 и L=w²/2 правильный результат 1.79. Если забыть восстановить w и шагнуть из 2.1 со вторым gradient 2.1, получится 1.89. Если к gradient второго pass случайно добавить первый, получится 1.59. Эти числа позволяют проверить порядок без большого обучения.

<!-- /unit:fe4ea68f0d0a2328db -->

<!-- unit:807adce04754571e52 -->
<a id="u-807adce04754571e52"></a>

Moments Adam или velocity SGD должны обновляться согласно внешнему алгоритму, а не незаметно дважды из-за двух backward passes. Нужно явно решить, входит ли выбранный weight penalty во внутренний loss или применяется отдельно во внешнем update. Смена этой политики меняет направление perturbation и смысл сравнения.

<!-- /unit:807adce04754571e52 -->

<!-- unit:e6d5472da3bbc14330 -->
<a id="u-e6d5472da3bbc14330"></a>

### Два passes могут менять больше, чем weights

<!-- /unit:e6d5472da3bbc14330 -->

<!-- unit:f380a54d74c25c2af5 -->
<a id="u-f380a54d74c25c2af5"></a>

BN running statistics, dropout masks и random transforms создают дополнительные изменения между passes. Если BN buffers обновились дважды, а baseline видел batch один раз, результат включает и другой state-update protocol. Реализация должна явно задать, какие statistics обновляются на каждом pass. Случайные views и masks также влияют на то, действительно ли сравнивается loss в окрестности weights при сопоставимом условии.

<!-- /unit:f380a54d74c25c2af5 -->

<!-- unit:d1d4ce10ed314afa84 -->
<a id="u-d1d4ce10ed314afa84"></a>

При gradient accumulation надо определить, perturbation строится по отдельному microbatch или по накопленному gradient. Это разные приближения. В distributed training порядок synchronization и возмущения тоже относится к алгоритму. Одного названия SAM в конфигурации недостаточно для воспроизводимого описания.

<!-- /unit:d1d4ce10ed314afa84 -->

<!-- unit:0ac9c72fe197bc077a -->
<a id="u-0ac9c72fe197bc077a"></a>

Два gradient evaluations на update обычно повышают compute, но не гарантируют ровно двукратного wall-clock времени: есть data transfer, kernels, synchronization и other overhead. При бюджете 1000 gradient evaluations baseline может сделать 1000 обычных updates, а двухградиентный SAM — 500 внешних updates. При фиксированных 1000 updates SAM потребляет больше evaluations. Оба сравнения отвечают разным вопросам; budget указывают явно.

<!-- /unit:0ac9c72fe197bc077a -->

<!-- unit:82553e93f7a08d8b9a -->
<a id="u-82553e93f7a08d8b9a"></a>

## Flatness и параметризация

<!-- /unit:82553e93f7a08d8b9a -->

<!-- unit:3f4321552c93a06dba -->
<a id="u-3f4321552c93a06dba"></a>

В homogeneous networks weights соседних layers можно rescale так, чтобы функция не изменилась, но local parameter-space sharpness изменилась. Поэтому «более плоский минимум всегда лучше обобщается» без указания norm и parametrization неверно. [Исследование reparametrization](https://arxiv.org/abs/1703.04933) показывает ограничения наивной flatness-интуиции.

<!-- /unit:3f4321552c93a06dba -->

<!-- unit:980ad3a295c297fd3a -->
<a id="u-980ad3a295c297fd3a"></a>

Это не делает SAM бессмысленным; это ограничивает теоретические выводы из одного geometry metric. Обоснованный итог звучит как measured improvement в заданном protocol, а не universal theorem о всех плоских weights.

<!-- /unit:980ad3a295c297fd3a -->

<!-- unit:5ed321229a301c6928 -->
<a id="u-5ed321229a301c6928"></a>

### Одна функция и разная Hessian sharpness

<!-- /unit:5ed321229a301c6928 -->

<!-- unit:e0b085727aed78b6db -->
<a id="u-e0b085727aed78b6db"></a>

Рассмотрим двухслойную scalar модель с output uv для unit input и $L(u,v)=\tfrac12(uv-1)^2$. В любой точке uv=1 loss нулевой. Hessian в такой точке равна

<!-- /unit:e0b085727aed78b6db -->

<!-- unit:dafb22fd103f7d9251 -->
<a id="u-dafb22fd103f7d9251"></a>

$$
H=\begin{pmatrix}v^2&uv\\uv&u^2\end{pmatrix}.
$$

<!-- /unit:dafb22fd103f7d9251 -->

<!-- unit:38a10d2e75f0631830 -->
<a id="u-38a10d2e75f0631830"></a>

Она имеет один ненулевой eigenvalue $u^2+v^2$. У решения (1,1) он равен 2; у (10,0.1) — 100.01. Обе модели реализуют одну и ту же функцию f(x)=x, но sharpness в исходных Euclidean coordinates существенно различается. Баланс масштаба слоёв изменил geometry, не изменив predictions.

<!-- /unit:38a10d2e75f0631830 -->

<!-- unit:56134c1ce3b7af3c6a -->
<a id="u-56134c1ce3b7af3c6a"></a>

Поэтому утверждение «меньшая Hessian norm всегда означает лучшее обобщение» требует дополнительных условий. В SAM выбранная parameter-space neighborhood является частью определения метода. Нормировка или scale-aware варианты меняют эту neighborhood; одно число rho нельзя считать инвариантным ко всем reparametrizations. Ограничения naive flatness-интуиции исследуются в [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933).

<!-- /unit:56134c1ce3b7af3c6a -->

<!-- unit:dbd26c4446bb996acf -->
<a id="u-dbd26c4446bb996acf"></a>

### Что можно заключить из успешного эксперимента

<!-- /unit:dbd26c4446bb996acf -->

<!-- unit:656d64a46c9cdcd4a8 -->
<a id="u-656d64a46c9cdcd4a8"></a>

Если SAM улучшил independent test при объявленном budget и protocol, это свидетельство полезности выбранной процедуры в этом режиме. Если дополнительно измерили flatness, можно исследовать связь этих наблюдений при фиксированной parametrization. Это не доказывает, что любой более плоский checkpoint будет лучше или что единственной причиной выигрыша была одна геометрическая характеристика.

<!-- /unit:656d64a46c9cdcd4a8 -->

<!-- unit:6f1b68897a75e9c9e9 -->
<a id="u-6f1b68897a75e9c9e9"></a>

Оценивают также clean loss, calibration, редкие срезы и variability. Подбор rho вместе с LR/decay относится к selection по development. Лучший test после перебора rho не является независимым подтверждением выбранной гипотезы.

<!-- /unit:6f1b68897a75e9c9e9 -->

<!-- unit:fc418543ed007e9428 -->
<a id="u-fc418543ed007e9428"></a>

## Lookahead: fast и slow weights

<!-- /unit:fc418543ed007e9428 -->

<!-- unit:f72eb7ea866770fc3d -->
<a id="u-f72eb7ea866770fc3d"></a>

Fast optimizer делает $k$ steps от slow weights $\phi$. Затем slow weights интерполируются:

<!-- /unit:f72eb7ea866770fc3d -->

<!-- unit:cd7af1a37649d14312 -->
<a id="u-cd7af1a37649d14312"></a>

$$
\phi^+=\phi+\alpha(w_k-\phi),
\qquad w\leftarrow\phi^+.
$$

<!-- /unit:cd7af1a37649d14312 -->

<!-- unit:8e068706e85a72e1d3 -->
<a id="u-8e068706e85a72e1d3"></a>

Если $\phi=1,w_k=3,\alpha=0.5$, новые slow и reset fast weights равны 2. Механизм сглаживает trajectory выбранного inner optimizer. Это не тот же min–max objective, что SAM, и не average всех checkpoints как SWA. [Lookahead](https://arxiv.org/abs/1907.08610).

<!-- /unit:8e068706e85a72e1d3 -->

<!-- unit:6179d243730795d884 -->
<a id="u-6179d243730795d884"></a>

У fast и slow states есть свои checkpoint requirements. При возобновлении training недостаточно сохранить только текущие fast parameters, если хотим продолжить тот же алгоритм. Наблюдаемая стабильность не гарантирует лучшую held-out quality на любой задаче.

<!-- /unit:6179d243730795d884 -->

<!-- unit:2e0743e8356cbc6090 -->
<a id="u-2e0743e8356cbc6090"></a>

### Один полный цикл

<!-- /unit:2e0743e8356cbc6090 -->

<!-- unit:93fafaba8f1fe082b8 -->
<a id="u-93fafaba8f1fe082b8"></a>

Fast optimizer стартует из текущих slow weights phi и делает k обычных шагов. Затем slow weights перемещаются на долю alpha к fast result, а fast parameters копируются из нового slow state. Это feedback в дальнейшую траекторию, а не только отдельный evaluation average.

<!-- /unit:93fafaba8f1fe082b8 -->

<!-- unit:4ce0f7109d2c5e28a9 -->
<a id="u-4ce0f7109d2c5e28a9"></a>

Пусть $L(w)=w^2/2$, phi=4, fast SGD eta=0.5, k=2, alpha=0.5. Fast weights проходят 4→2→1. Новый slow равен $4+0.5(1-4)=2.5$, и fast reset тоже даёт 2.5. Следующий цикл проходит 2.5→1.25→0.625, затем slow становится $2.5+0.5(0.625-2.5)=1.5625$.

<!-- /unit:4ce0f7109d2c5e28a9 -->

<!-- unit:0a3f793cd995081b62 -->
<a id="u-0a3f793cd995081b62"></a>

SWA только усреднял бы выбранные states для отдельной модели, не обязательно возвращая average в fast trajectory. EMA тоже не предполагает автоматически такой reset. В [Lookahead](https://arxiv.org/abs/1907.08610) именно взаимодействие fast/slow является существенной частью алгоритма.

<!-- /unit:0a3f793cd995081b62 -->

<!-- unit:a2f7f93f9726ba065d -->
<a id="u-a2f7f93f9726ba065d"></a>

### Формула для quadratic и предел обещаний

<!-- /unit:a2f7f93f9726ba065d -->

<!-- unit:b8497ea5053203ec3b -->
<a id="u-b8497ea5053203ec3b"></a>

Для $L(w)=\tfrac12(w-a)^2$ fast SGD даёт $w_k=a+(1-\eta)^k(\phi-a)$. После slow interpolation:

<!-- /unit:b8497ea5053203ec3b -->

<!-- unit:13e144ccaa9777af1c -->
<a id="u-13e144ccaa9777af1c"></a>

$$
\phi^+-a=[1-\alpha+\alpha(1-\eta)^k](\phi-a).
$$

<!-- /unit:13e144ccaa9777af1c -->

<!-- unit:c483ebc6b16e2e16c2 -->
<a id="u-c483ebc6b16e2e16c2"></a>

В предыдущем примере multiplier равен $0.5+0.5\cdot0.5^2=0.625$ на cycle. Но при eta=2.5 и тех же k=2, alpha=0.5 multiplier=1.625, и расстояние до a растёт. Lookahead не исправляет любой чрезмерный шаг inner optimizer автоматически.

<!-- /unit:c483ebc6b16e2e16c2 -->

<!-- unit:3559bf0b8458f4e6a7 -->
<a id="u-3559bf0b8458f4e6a7"></a>

При alpha=1 slow weights полностью принимают fast result; при сохранении остальных состояний параметрическая траектория совпадает с продолжением fast optimizer. При иных alpha и k меняются частота обратной связи и scale продвижения. Для momentum/adaptive inner optimizer простая замена learning rate не воспроизводит автоматически весь алгоритм, поскольку есть дополнительное состояние.

<!-- /unit:3559bf0b8458f4e6a7 -->

<!-- unit:cb1b4af1a94c30825e -->
<a id="u-cb1b4af1a94c30825e"></a>

### Checkpoint содержит два уровня

<!-- /unit:cb1b4af1a94c30825e -->

<!-- unit:92531de2a9eaaa2754 -->
<a id="u-92531de2a9eaaa2754"></a>

Для resume нужны fast parameters, slow parameters, optimizer state, положение внутри k-step cycle, scheduler и обычное data/RNG state. Сохранять только slow checkpoint удобно для inference, но недостаточно для точного продолжения того же обучения. Политика сохранения либо переноса fast moments после synchronization должна соответствовать выбранной реализации.

<!-- /unit:92531de2a9eaaa2754 -->

<!-- unit:deadeef4045f1ec484 -->
<a id="u-deadeef4045f1ec484"></a>

Сравнение учитывает все k inner updates, а не считает один slow synchronization равным одному обычному optimizer step. Возможное сглаживание trajectory и улучшение test являются разными наблюдаемыми свойствами: второе требует отдельной оценки.

<!-- /unit:deadeef4045f1ec484 -->

<!-- unit:f5303ca3c7839275d4 -->
<a id="u-f5303ca3c7839275d4"></a>

## Диагностика, sweep и окончательный выбор

<!-- /unit:f5303ca3c7839275d4 -->

<!-- unit:cc3516ff95923bc642 -->
<a id="u-cc3516ff95923bc642"></a>

Сначала проверяем learning loop на маленьком batch: верные labels, shapes, masks, gradient flow и возможность уменьшить training loss. Это диагностический опыт, а не proof generalization. Затем baseline на полном train, корректный split и learning curves. Если модель underfit из-за отсутствия нужного feature, увеличение dropout не создаст информацию.

<!-- /unit:cc3516ff95923bc642 -->

<!-- unit:a3c60ecdfdc1585a45 -->
<a id="u-a3c60ecdfdc1585a45"></a>

Выбираем один механизм, соответствующий гипотезе: L2 при нестабильных coefficients, data augmentation при обоснованной invariance, early stopping при degradation поздних checkpoints. Отсутствие training progress сначала требует проверить optimizer/LR/data. Distribution shift требует анализа срезов и sources.

<!-- /unit:a3c60ecdfdc1585a45 -->

<!-- unit:f59fcb6f5bd0f094a2 -->
<a id="u-f59fcb6f5bd0f094a2"></a>

Начальный логарифмический sweep lambda, например от $10^{-5}$ до $10^{-1}$, исследует порядки magnitude. Диапазон зависит от normalization loss, model и optimizer; эти числа не универсальная рекомендация. Обязательно есть baseline без выбранного penalty и несколько seeds для finalist configurations.

<!-- /unit:f59fcb6f5bd0f094a2 -->

<!-- unit:928d1d01c9671ba719 -->
<a id="u-928d1d01c9671ba719"></a>

Настройки взаимодействуют: dropout, augmentation, weight decay, LR и epochs меняют effective training difficulty. Сначала полезна ablation по одному фактору, затем совместный подбор наиболее перспективных. Bayesian optimization/Optuna — инструменты поиска, которые не устраняют validation overfitting и не делают test пригодным для tuning.

<!-- /unit:928d1d01c9671ba719 -->

<!-- unit:d63a49faaada1cca21 -->
<a id="u-d63a49faaada1cca21"></a>

Универсального закона $\lambda\propto1/n$ нет. Он зависит от постановки objective, prior interpretation, noise и class модели. При удвоении числа examples нужно также понять, оставили ли steps фиксированными или удвоили compute.

<!-- /unit:d63a49faaada1cca21 -->

<!-- unit:a22f02ebc89037dfd9 -->
<a id="u-a22f02ebc89037dfd9"></a>

Итоговый журнал включает split/version, objective convention, training budget, curves, validation metric, calibration, seeds, runtime и final independent test. Не следует объявлять winner по одному случайному seed или по лучшей метрике из множества просмотренных tests.

<!-- /unit:a22f02ebc89037dfd9 -->

<!-- unit:af7af3975eb2f111ff -->
<a id="u-af7af3975eb2f111ff"></a>

### Диагностика задаёт следующий эксперимент

<!-- /unit:af7af3975eb2f111ff -->

<!-- unit:653c0dcd42b6e6eca7 -->
<a id="u-653c0dcd42b6e6eca7"></a>

Фиксируют цель и independent unit: новый документ, пользователь, изображение или будущий период. Проверяют targets, splits, transforms, masks, data loss, режим evaluation и доступность данных. Если обучение не идёт из-за неверных labels или LR, добавление ещё одного штрафа может скрыть причину. Если различается source distribution, одна regularization-strength не заменяет анализа переноса.

<!-- /unit:653c0dcd42b6e6eca7 -->

<!-- unit:29191bf91292f250dc -->
<a id="u-29191bf91292f250dc"></a>

Baseline означает отсутствие проверяемого вмешательства, а не буквальное отсутствие всех priors. В нём уже могут быть pretrained weights, weight sharing, normalization и существующие аугментации. Поэтому запись «без регуляризации» лучше уточнить списком включённых механизмов. Иначе сравнение не воспроизводится.

<!-- /unit:29191bf91292f250dc -->

<!-- unit:100f8636c7437e06b0 -->
<a id="u-100f8636c7437e06b0"></a>

Для выбранной гипотезы заранее задают ожидаемый наблюдаемый эффект и критерий опровержения. Например, ridge должен уменьшить нестабильность coefficients при коррелированных features, а augmentation — помочь определённым допустимым изменениям входа. Одного уменьшения training objective недостаточно: penalty сама меняет этот objective.

<!-- /unit:100f8636c7437e06b0 -->

<!-- unit:724e419d6e39d2e1eb -->
<a id="u-724e419d6e39d2e1eb"></a>

### Абляции и взаимодействия

<!-- /unit:724e419d6e39d2e1eb -->

<!-- unit:eabdbf89a975774766 -->
<a id="u-eabdbf89a975774766"></a>

Учебные validation losses четырёх configurations: baseline=0.40, только decay=0.38, только augmentation=0.37, вместе=0.43. Оба метода по отдельности помогли, вместе ухудшили. Additive interaction contrast равен $0.43-0.38-0.37+0.40=0.08$. Это пример, почему нельзя автоматически складывать все успешные приёмы.

<!-- /unit:eabdbf89a975774766 -->

<!-- unit:a905f2f9a975c77449 -->
<a id="u-a905f2f9a975c77449"></a>

Причиной могут быть чрезмерное ограничение, другой оптимальный LR, нехватка steps или изменение информации в inputs. Сначала исследуют эффекты отдельно, затем совместно подбирают осмысленные combinations. Такие четыре числа иллюстрируют дизайн, но без реальных повторений и sampling uncertainty не являются измеренным доказательством взаимодействия конкретной модели.

<!-- /unit:a905f2f9a975c77449 -->

<!-- unit:d412e2556c975887ae -->
<a id="u-d412e2556c975887ae"></a>

### Sweep и финальный выбор

<!-- /unit:d412e2556c975887ae -->

<!-- unit:13c4fa0d72654ea4c9 -->
<a id="u-13c4fa0d72654ea4c9"></a>

Начальный logarithmic sweep исследует несколько порядков strength и содержит baseline coefficient=0. Если все ненулевые значения ухудшают результат, вывод не обязан быть «увеличивать дальше»: диапазон мог быть слишком сильным либо гипотеза неверна. Если лучший находится у границы, можно уточнить диапазон, сохраняя историю selection. Нормировку loss, параметры groups и LR schedule фиксируют рядом с lambda.

<!-- /unit:13c4fa0d72654ea4c9 -->

<!-- unit:c00764ff30d287343c -->
<a id="u-c00764ff30d287343c"></a>

Grid, random search и Bayesian optimization различаются организацией поиска, но все используют development evidence и могут переобучить выбор. Finalists оценивают по заранее выбранным repeats/seeds и важным срезам. Test открывают для оценки итоговой процедуры; последующая настройка по его ошибкам означает новый development cycle, которому потребуется новое независимое подтверждение.

<!-- /unit:c00764ff30d287343c -->

<!-- unit:8111dfbdeaa3536d11 -->
<a id="u-8111dfbdeaa3536d11"></a>

Итог сохраняет версии данных/кода, конфигурацию всех активных методов, число реальных updates и processed tokens, выбранный checkpoint, stateful buffers, runtime, метрики и uncertainty procedure. Для Transformer отдельно важны active token masks, pretrained reference, adapter modules и параметры decoder: изменение decoding может изменить quality без изменения training regularization.

<!-- /unit:8111dfbdeaa3536d11 -->

<!-- unit:889551d01a1fe03b87 -->
<a id="u-889551d01a1fe03b87"></a>

## Контроль и практика

<!-- /unit:889551d01a1fe03b87 -->

<!-- unit:c63a5bce31d581649f -->
<a id="u-c63a5bce31d581649f"></a>

1. Отличаются ли radius SAM и LR? **Ответ:** radius определяет inner perturbation neighborhood, LR — внешний update step.
2. Эквивалентен ли Lookahead SWA? **Ответ:** нет; slow weights периодически перемещаются к fast weights, после чего fast reset.
3. Когда больше regularization ухудшает обе curves? **Ответ:** при слишком сильном ограничении или уже существующем underfit; также нужно исключить optimization/data ошибки.

<!-- /unit:c63a5bce31d581649f -->

<!-- unit:8d44e9080d73f6fefc -->
<a id="u-8d44e9080d73f6fefc"></a>

**Практика:** повторите scalar SAM/Lookahead examples. Затем спроектируйте журнал из baseline и шести configurations одной регуляризации: hypothesis, strength, одинаковый budget, validation result и seeds. Заранее запишите правило выбора finalist и закрытый test; отсутствие improvement тоже является содержательным результатом.

<!-- /unit:8d44e9080d73f6fefc -->

<!-- unit:831acd01430a301387 -->
<a id="u-831acd01430a301387"></a>

**Схема курса:** baseline checkpoint → perturb weights and compute SAM direction or synchronize Lookahead slow/fast weights → optimizer update → matched-budget validation comparison

<!-- /unit:831acd01430a301387 -->

<!-- unit:8087fddcd3d6df1e28 -->
<a id="u-8087fddcd3d6df1e28"></a>

<!-- source-depth-regularization-К -->

<!-- /unit:8087fddcd3d6df1e28 -->

<!-- unit:79070538e504f1f56d -->
<a id="u-79070538e504f1f56d"></a>

1. **Почему SAM должен восстановить исходные weights перед внешним update?** Возмущение служит для вычисления второго gradient, а не является самостоятельным постоянным шагом обучения.
2. **Почему большой eigenvalue Hessian не доказывает плохую функцию?** Он зависит от параметризации; rescaling слоёв может оставить функцию прежней.
3. **Чем Lookahead отличается от evaluation averaging?** Slow weights возвращаются в дальнейший fast training через reset.
4. **Почему два полезных метода могут ухудшить результат вместе?** Их effects взаимодействуют через objective, информацию данных и траекторию оптимизации.

<!-- /unit:79070538e504f1f56d -->

<!-- unit:743937a3ebc3d5555e -->
<a id="u-743937a3ebc3d5555e"></a>

Практика: воспроизвести двумерный SAM, два Lookahead cycles и Hessian rescaling. Затем записать протокол одного сравнения с baseline, budget, finalists и закрытым test. Отсутствие выигрыша также сохраняют как результат. Все числа главы учебные; SAM/Lookahead обучение пользовательской модели здесь не запускалось.

<!-- /unit:743937a3ebc3d5555e -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как проверить гипотезу о регуляризации при фиксированном бюджете настройки?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
