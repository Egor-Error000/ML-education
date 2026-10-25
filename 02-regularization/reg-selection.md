# SAM, Lookahead и итоговый выбор метода

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>), [Штрафы, ограничения, L1/L2 и AdamW](<reg-penalties.md>), [Early stopping, learning rate, SWA и EMA](<reg-training.md>), [Dropout, DropConnect, DropBlock и случайные ветви](<reg-stochastic.md>)

**Используют:** Завершение выбранной ветки: самостоятельный эксперимент по практике этой главы.

**Учебный вопрос:** Как проверить гипотезу о регуляризации при фиксированном бюджете настройки?

## Карта вычисления и границы этапов

![SAM, Lookahead и итоговый выбор метода: входы, преобразования и выходы](<../оформление/course/reg-selection.svg>)

Протокол выбора связывает вмешательство с проверяемой причиной. Итоговый вариант фиксируется до теста, а отрицательные результаты сохраняются вместе с затратами поиска.

## Два шага SAM

Sharpness-Aware Minimization ищет weights с низким худшим loss (функция потерь) в выбранной local neighborhood:

$$
\min_w\max_{\|\varepsilon\|_2\le\rho}L(w+\varepsilon).
$$

Линеаризация inner objective (оптимизируемая целевая функция) даёт perturbation

$$
\widehat\varepsilon=
\rho\frac{\nabla L(w)}{\|\nabla L(w)\|_2},
\qquad
w^+=w-\eta\nabla L(w+\widehat\varepsilon).
$$

В базовом приближённом update perturbation рассматривается как фиксированное смещение для второго gradient (градиент); это не полный differentiation через весь inner solver. При нулевой gradient norm нужно отдельное стабилизированное правило. Параметр $\rho$ задаёт radius, не learning rate (шаг обучения). [SAM](https://arxiv.org/abs/2010.01412).

Пример для $L(w)=w^2/2$, $w=2,\rho=0.1,\eta=0.1$: первый gradient 2, perturbation 0.1, второй gradient 2.1, update даёт $w^+=1.79$. Обычный GD дал бы 1.8. В этом scalar convex примере нет загадочной «магии flat basins»; видно, как меняется шаг. Для network эффект зависит от data и geometry.

SAM обычно требует дополнительного forward (прямой проход вычисления)/backward (обратный проход вычисления производных) и аккуратного управления stateful layers. Если BatchNorm statistics обновить дважды несогласованно, сравнение может смешать эффект optimizer (оптимизатор обновления параметров) и statistics. Конкретная реализация должна явно определить state (состояние системы) handling. Сравнивать нужно и quality, и compute.

### Линеаризация внутреннего максимума

В базовом SAM ищется устойчивость loss к perturbation weights. Для малого epsilon $L(w+\epsilon)\approx L(w)+g^\top\epsilon$, где g=gradient L(w). На Euclidean ball radius rho максимизирующее направление совпадает с g и имеет длину rho. Это объясняет формулу раздела 1, но только для линейного приближения inner objective.

При другом norm ball оптимальное линейное направление меняется; rho без norm не определяет задачу. В нелинейном loss curvature может сдвинуть настоящий worst-case point относительно normalized gradient. В частности, нулевой g не означает отсутствия роста loss вокруг w: линейное приближение просто не видит направление. Стабилизатор denominator нужен вычислительно, но не превращает approximate inner step в точный solver.

Adversarial training из главы [«Adversarial training, consistency и spectral norm»](<reg-robustness.md>) ищет perturbation входа x, SAM — параметров w. Оба используют min–max идею, однако пространства, размерности, units и смысл сохранения target различаются. Из успеха одного метода не следует robustness другого к тем же изменениям.

### Двумерный шаг с разной curvature

Возьмём $L(w)=\tfrac12(w_1^2+4w_2^2)$, w=(3,1), rho=0.1, eta=0.1. Первый gradient равен (3,4), norm=5. Approximate inner perturbation=(0.06,0.08), perturbed weights=(3.06,1.08). Второй gradient=(3.06,4.32), поэтому внешний update исходных weights даёт (2.694,0.568).

Обычный GD дал бы (2.7,0.6). Второе направление сильнее реагирует на perturbation из-за большего quadratic curvature. Этот учебный пример показывает конкретную механику; он не доказывает улучшение test quality. Исходный objective и приближение приведены в [SAM](https://arxiv.org/abs/2010.01412).

### Правильный порядок операций и состояние optimizer

На исходных weights получают первый gradient, строят perturbation и временно прибавляют её. Первый gradient очищают из накопителей, если он не нужен для отдельно определённого правила. Затем получают второй gradient, восстанавливают исходные weights и один раз выполняют внешний optimizer update с этим gradient. В базовом приближении epsilon при втором differentiation рассматривается как фиксированная. Если смешать оба gradients без такого намерения, получится другой update.

В scalar примере w=2, rho=0.1, eta=0.1 и L=w²/2 правильный результат 1.79. Если забыть восстановить w и шагнуть из 2.1 со вторым gradient 2.1, получится 1.89. Если к gradient второго pass случайно добавить первый, получится 1.59. Эти числа позволяют проверить порядок без большого обучения.

Moments Adam или velocity SGD должны обновляться согласно внешнему алгоритму, а не незаметно дважды из-за двух backward passes. Нужно явно решить, входит ли выбранный weight penalty во внутренний loss или применяется отдельно во внешнем update. Смена этой политики меняет направление perturbation и смысл сравнения.

### Два passes могут менять больше, чем weights

BN running statistics, dropout (случайное выключение выбранных активаций при обучении) masks и random transforms создают дополнительные изменения между passes. Если BN buffers обновились дважды, а baseline (базовый вариант сравнения) видел batch (пакет примеров) один раз, результат включает и другой state-update protocol. Реализация должна явно задать, какие statistics обновляются на каждом pass. Случайные views и masks также влияют на то, действительно ли сравнивается loss в окрестности weights при сопоставимом условии.

При gradient accumulation надо определить, perturbation строится по отдельному microbatch или по накопленному gradient. Это разные приближения. В distributed training порядок synchronization и возмущения тоже относится к алгоритму. Одного названия SAM в конфигурации недостаточно для воспроизводимого описания.

Два gradient evaluations на update обычно повышают compute, но не гарантируют ровно двукратного wall-clock времени: есть data transfer, kernels, synchronization и other overhead. При бюджете 1000 gradient evaluations baseline может сделать 1000 обычных updates, а двухградиентный SAM — 500 внешних updates. При фиксированных 1000 updates SAM потребляет больше evaluations. Оба сравнения отвечают разным вопросам; budget указывают явно.

## Flatness и параметризация

В homogeneous networks weights соседних layers можно rescale так, чтобы функция не изменилась, но local parameter-space sharpness изменилась. Поэтому «более плоский минимум всегда лучше обобщается» без указания norm и parametrization неверно. [Исследование reparametrization](https://arxiv.org/abs/1703.04933) показывает ограничения наивной flatness-интуиции.

Это не делает SAM бессмысленным; это ограничивает теоретические выводы из одного geometry metric. Обоснованный итог звучит как measured improvement в заданном protocol, а не universal theorem о всех плоских weights.

### Одна функция и разная Hessian sharpness

Рассмотрим двухслойную scalar модель с output uv для unit input и $L(u,v)=\tfrac12(uv-1)^2$. В любой точке uv=1 loss нулевой. Hessian в такой точке равна

$$
H=\begin{pmatrix}v^2&uv\\uv&u^2\end{pmatrix}.
$$

Она имеет один ненулевой eigenvalue $u^2+v^2$. У решения (1,1) он равен 2; у (10,0.1) — 100.01. Обе модели реализуют одну и ту же функцию f(x)=x, но sharpness в исходных Euclidean coordinates существенно различается. Баланс масштаба слоёв изменил geometry, не изменив predictions.

Поэтому утверждение «меньшая Hessian norm всегда означает лучшее обобщение» требует дополнительных условий. В SAM выбранная parameter-space neighborhood является частью определения метода. Нормировка или scale-aware варианты меняют эту neighborhood; одно число rho нельзя считать инвариантным ко всем reparametrizations. Ограничения naive flatness-интуиции исследуются в [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933).

### Что можно заключить из успешного эксперимента

Если SAM улучшил independent test при объявленном budget и protocol, это свидетельство полезности выбранной процедуры в этом режиме. Если дополнительно измерили flatness, можно исследовать связь этих наблюдений при фиксированной parametrization. Это не доказывает, что любой более плоский checkpoint (сохранённое состояние модели) будет лучше или что единственной причиной выигрыша была одна геометрическая характеристика.

Оценивают также clean loss, calibration, редкие срезы и variability. Подбор rho вместе с LR/decay относится к selection по development. Лучший test после перебора rho не является независимым подтверждением выбранной гипотезы.

## Lookahead: fast и slow weights

Fast optimizer делает $k$ steps от slow weights $\phi$. Затем slow weights интерполируются:

$$
\phi^+=\phi+\alpha(w_k-\phi),
\qquad w\leftarrow\phi^+.
$$

Если $\phi=1,w_k=3,\alpha=0.5$, новые slow и reset fast weights равны 2. Механизм сглаживает trajectory (траектория действий и наблюдений) выбранного inner optimizer. Это не тот же min–max objective, что SAM, и не average всех checkpoints как SWA. [Lookahead](https://arxiv.org/abs/1907.08610).

У fast и slow states есть свои checkpoint requirements. При возобновлении training недостаточно сохранить только текущие fast parameters (параметры), если хотим продолжить тот же алгоритм. Наблюдаемая стабильность не гарантирует лучшую held-out quality на любой задаче.

### Один полный цикл

Fast optimizer стартует из текущих slow weights phi и делает k обычных шагов. Затем slow weights перемещаются на долю alpha к fast result, а fast parameters копируются из нового slow state. Это feedback в дальнейшую траекторию, а не только отдельный evaluation average.

Пусть $L(w)=w^2/2$, phi=4, fast SGD eta=0.5, k=2, alpha=0.5. Fast weights проходят 4→2→1. Новый slow равен $4+0.5(1-4)=2.5$, и fast reset тоже даёт 2.5. Следующий цикл проходит 2.5→1.25→0.625, затем slow становится $2.5+0.5(0.625-2.5)=1.5625$.

SWA только усреднял бы выбранные states для отдельной модели, не обязательно возвращая average в fast trajectory. EMA тоже не предполагает автоматически такой reset. В [Lookahead](https://arxiv.org/abs/1907.08610) именно взаимодействие fast/slow является существенной частью алгоритма.

### Формула для quadratic и предел обещаний

Для $L(w)=\tfrac12(w-a)^2$ fast SGD даёт $w_k=a+(1-\eta)^k(\phi-a)$. После slow interpolation:

$$
\phi^+-a=[1-\alpha+\alpha(1-\eta)^k](\phi-a).
$$

В предыдущем примере multiplier равен $0.5+0.5\cdot0.5^2=0.625$ на cycle. Но при eta=2.5 и тех же k=2, alpha=0.5 multiplier=1.625, и расстояние до a растёт. Lookahead не исправляет любой чрезмерный шаг inner optimizer автоматически.

При alpha=1 slow weights полностью принимают fast result; при сохранении остальных состояний параметрическая траектория совпадает с продолжением fast optimizer. При иных alpha и k меняются частота обратной связи и scale продвижения. Для momentum/adaptive inner optimizer простая замена learning rate не воспроизводит автоматически весь алгоритм, поскольку есть дополнительное состояние.

### Checkpoint содержит два уровня

Для resume нужны fast parameters, slow parameters, optimizer state, положение внутри k-step cycle, scheduler и обычное data/RNG state. Сохранять только slow checkpoint удобно для inference (применение обученной модели), но недостаточно для точного продолжения того же обучения. Политика сохранения либо переноса fast moments после synchronization должна соответствовать выбранной реализации.

Сравнение учитывает все k inner updates, а не считает один slow synchronization равным одному обычному optimizer step. Возможное сглаживание trajectory и улучшение test являются разными наблюдаемыми свойствами: второе требует отдельной оценки.

## Диагностика, sweep и окончательный выбор

Сначала проверяем learning loop на маленьком batch: верные labels (метки), shapes, masks, gradient flow и возможность уменьшить training loss. Это диагностический опыт, а не proof generalization. Затем baseline на полном train, корректный split (разбиение данных) и learning curves. Если модель underfit из-за отсутствия нужного feature (признак), увеличение dropout не создаст информацию.

Выбираем один механизм, соответствующий гипотезе: L2 при нестабильных coefficients, data augmentation (преобразование обучающего примера с согласованной целью) при обоснованной invariance, early stopping при degradation поздних checkpoints. Отсутствие training progress сначала требует проверить optimizer/LR/data. Distribution shift требует анализа срезов и sources.

Начальный логарифмический sweep lambda, например от $10^{-5}$ до $10^{-1}$, исследует порядки magnitude. Диапазон зависит от normalization (нормировка по выбранному правилу) loss, model и optimizer; эти числа не универсальная рекомендация. Обязательно есть baseline без выбранного penalty и несколько seeds для finalist configurations.

Настройки взаимодействуют: dropout, augmentation, weight decay (затухание весов при обновлении), LR и epochs меняют effective training difficulty. Сначала полезна ablation (сравнение с удалённой или изменённой частью метода) по одному фактору, затем совместный подбор наиболее перспективных. Bayesian optimization/Optuna — инструменты поиска, которые не устраняют validation (проверочная часть для выбора решения) overfitting и не делают test пригодным для tuning.

Универсального закона $\lambda\propto1/n$ нет. Он зависит от постановки objective, prior interpretation, noise и class модели. При удвоении числа examples нужно также понять, оставили ли steps фиксированными или удвоили compute.

Итоговый журнал включает split/version, objective convention, training budget, curves, validation metric, calibration, seeds, runtime (среда исполнения модели) и final independent test. Не следует объявлять winner по одному случайному seed (начальное состояние генератора случайных чисел) или по лучшей метрике из множества просмотренных tests.

### Диагностика задаёт следующий эксперимент

Фиксируют цель и independent unit: новый документ, пользователь, изображение или будущий период. Проверяют targets (целевые значения), splits, transforms, masks, data loss, режим evaluation и доступность данных. Если обучение не идёт из-за неверных labels или LR, добавление ещё одного штрафа может скрыть причину. Если различается source distribution, одна regularization-strength не заменяет анализа переноса.

Baseline означает отсутствие проверяемого вмешательства, а не буквальное отсутствие всех priors. В нём уже могут быть pretrained weights, weight sharing, normalization и существующие аугментации. Поэтому запись «без регуляризации» лучше уточнить списком включённых механизмов. Иначе сравнение не воспроизводится.

Для выбранной гипотезы заранее задают ожидаемый наблюдаемый эффект и критерий опровержения. Например, ridge должен уменьшить нестабильность coefficients при коррелированных features (признаки), а augmentation — помочь определённым допустимым изменениям входа. Одного уменьшения training objective недостаточно: penalty сама меняет этот objective.

### Абляции и взаимодействия

Учебные validation losses четырёх configurations: baseline=0.40, только decay=0.38, только augmentation=0.37, вместе=0.43. Оба метода по отдельности помогли, вместе ухудшили. Additive interaction contrast равен $0.43-0.38-0.37+0.40=0.08$. Это пример, почему нельзя автоматически складывать все успешные приёмы.

Причиной могут быть чрезмерное ограничение, другой оптимальный LR, нехватка steps или изменение информации в inputs. Сначала исследуют эффекты отдельно, затем совместно подбирают осмысленные combinations. Такие четыре числа иллюстрируют дизайн, но без реальных повторений и sampling (случайный выбор по распределению) uncertainty не являются измеренным доказательством взаимодействия конкретной модели.

### Sweep и финальный выбор

Начальный logarithmic sweep исследует несколько порядков strength и содержит baseline coefficient=0. Если все ненулевые значения ухудшают результат, вывод не обязан быть «увеличивать дальше»: диапазон мог быть слишком сильным либо гипотеза неверна. Если лучший находится у границы, можно уточнить диапазон, сохраняя историю selection. Нормировку loss, параметры groups и LR schedule фиксируют рядом с lambda.

Grid, random search и Bayesian optimization различаются организацией поиска, но все используют development evidence и могут переобучить выбор. Finalists оценивают по заранее выбранным repeats/seeds и важным срезам. Test открывают для оценки итоговой процедуры; последующая настройка по его ошибкам означает новый development cycle, которому потребуется новое независимое подтверждение.

Итог сохраняет версии данных/кода, конфигурацию всех активных методов, число реальных updates и processed tokens (токены), выбранный checkpoint, stateful buffers, runtime, метрики и uncertainty procedure. Для Transformer отдельно важны active token (токен) masks, pretrained reference, adapter modules и параметры decoder (декодер): изменение decoding может изменить quality без изменения training regularization.

## Контроль и практика

1. Отличаются ли radius SAM и LR? **Ответ:** radius определяет inner perturbation neighborhood, LR — внешний update step.
2. Эквивалентен ли Lookahead SWA? **Ответ:** нет; slow weights периодически перемещаются к fast weights, после чего fast reset.
3. Когда больше regularization ухудшает обе curves? **Ответ:** при слишком сильном ограничении или уже существующем underfit; также нужно исключить optimization/data ошибки.

**Практика:** повторите scalar SAM/Lookahead examples. Затем спроектируйте журнал из baseline и шести configurations одной регуляризации: hypothesis, strength, одинаковый budget, validation result и seeds. Заранее запишите правило выбора finalist и закрытый test; отсутствие improvement тоже является содержательным результатом.

**Схема курса:** baseline checkpoint → perturb weights and compute SAM direction or synchronize Lookahead slow/fast weights → optimizer update → matched-budget validation comparison



1. **Почему SAM должен восстановить исходные weights перед внешним update?** Возмущение служит для вычисления второго gradient, а не является самостоятельным постоянным шагом обучения.
2. **Почему большой eigenvalue Hessian не доказывает плохую функцию?** Он зависит от параметризации; rescaling слоёв может оставить функцию прежней.
3. **Чем Lookahead отличается от evaluation averaging?** Slow weights возвращаются в дальнейший fast training через reset.
4. **Почему два полезных метода могут ухудшить результат вместе?** Их effects взаимодействуют через objective, информацию данных и траекторию оптимизации.

Практика: воспроизвести двумерный SAM, два Lookahead cycles и Hessian rescaling. Затем записать протокол одного сравнения с baseline, budget, finalists и закрытым test. Отсутствие выигрыша также сохраняют как результат. Все числа главы учебные; SAM/Lookahead обучение пользовательской модели здесь не запускалось.

## Результат главы и переход дальше

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
