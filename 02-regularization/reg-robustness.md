# Adversarial training, consistency и spectral norm

**Уровень:** общее ядро.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>)

**Используют:** Завершение выбранной ветки: самостоятельный эксперимент по практике этой главы.

**Учебный вопрос:** Относительно каких допустимых изменений требуется устойчивость?

## Карта вычисления и границы этапов

![Adversarial training, consistency и spectral norm: входы, преобразования и выходы](<../оформление/course/reg-robustness.svg>)

Штраф согласованности сравнивает ответы на связанные входы. Устойчивость относится к объявленному множеству возмущений и не доказывает надёжность вне него.

## Threat model и задача устойчивости

Малое расстояние во входном пространстве не всегда означает неизменную метку. Если $\delta$ может изменить число на изображении, слово отрицания в тексте или физически важный параметр, forcing consistency даст неверный prior. Robustness всегда относится к выбранному множеству perturbations и threat model.

Supervised adversarial training решает приближённую min–max задачу:

$$
\min_\theta\mathbb E_{(x,y)}
\left[\max_{\delta\in\Delta}\ell(f_\theta(x+\delta),y)\right].
$$

Внутри ищется harmful perturbation при фиксированной модели, снаружи weights обновляются для устойчивого prediction. Для $L_\infty$ ball FGSM использует linear approximation:

$$
\delta=\epsilon\operatorname{sign}(\nabla_x\ell(f_\theta(x),y)).
$$

PGD выполняет несколько gradient-ascent steps с projection обратно в $\Delta$. Область допустимых pixels тоже учитывается. Один слабый attack не доказывает robustness ко всем attacks; gradient (градиент) masking может создавать ложное впечатление защиты.

### Зафиксировать пространство, радиус и допустимые изменения

Сначала описывают, что изменяется: pixels до нормализации, embeddings (векторные представления), физические параметры, части текста или результат конкретного преобразования. Затем задают norm, radius, диапазон значений и предположение о сохранности target. Радиус 0.1 без указания единиц может означать долю цветового диапазона, standardized feature (признак) или совершенно другое изменение.

Для изображения x в [0,1] после preprocessing $z_j=(x_j-\mu_j)/s_j$ ограничение $|\delta x_j|\le\epsilon$ соответствует $|\delta z_j|\le\epsilon/s_j$. Если s=(0.5,0.25), raw radius 0.1 превращается в coordinate bounds (0.2,0.4). Одинаковая normalized epsilon для всех channels задаёт уже другую raw threat model. В report сохраняют, на каком этапе pipeline выполнена атака.

Не всякое малое изменение сохраняет истинный ответ. Удаление одного символа из математического знака или отрицания может быть малым в одном representation и большим по смыслу. Поэтому robustness к norm ball, к допустимым перефразировкам и к смене камеры — разные требования. Ни одно автоматически не включает остальные.

### Внутренняя и внешняя оптимизация

Во внутренней задаче параметры theta фиксированы, а perturbation выбирается для увеличения loss (функция потерь). Во внешней задаче theta обновляют по найденным adversarial inputs. Если на каждом inner step вызвать optimizer (оптимизатор обновления параметров) модели, получится другая процедура: меняется и объект, против которого ищется perturbation.

Обычно найденную perturbation для внешнего шага рассматривают как зафиксированную и заново вычисляют model loss; не требуется автоматически дифференцировать через всю историю PGD. Но detach perturbation не должен случайно удалить gradient внешнего loss по theta. Режим dropout, BN state (состояние системы) и preprocessing во время поиска тоже задают явно: stochastic forward (прямой проход вычисления) или меняющиеся buffers меняют inner objective (оптимизируемая целевая функция).

## FGSM и PGD

Линеаризуем loss: $\ell(x+\delta)\approx\ell(x)+g^\top\delta$. При $|\delta_j|\le\epsilon$ каждый summand максимален при $\delta_j=\epsilon\operatorname{sign}(g_j)$. Для $g=(2,-1),\epsilon=0.1$ берём $(0.1,-0.1)$ и predicted increase равен $0.3$. В $L_2$ ball направление другое: $\delta=\epsilon g/\|g\|_2$. Radius без указания norm не определяет один и тот же attack.

Для tokens (токены) нельзя непосредственно прибавить дробное число к token (токен) ID. Возможны perturbations embeddings или discrete substitutions, но они задают разные threat models и constraints. Даже малый embedding (векторное представление) noise не обязательно соответствует допустимому человеческому перефразированию.

### Линейная задача и двойственная норма

Для локального gradient g inner objective приближённо увеличивается на $g^\top\delta$. При norm ball $\|\delta\|_p\le\epsilon$ максимум линейного слагаемого равен $\epsilon\|g\|_q$, где $1/p+1/q=1$. Для $L_\infty$ это sign gradient, для $L_2$ — нормированный gradient. Для $L_1$ бюджета линейный максимум можно направить в coordinate с наибольшим абсолютным gradient.

При g=(2,−1) и epsilon=0.1 $L_\infty$ perturbation (0.1,−0.1) даёт прирост 0.3; $L_2$ perturbation $(0.2,−0.1)/\sqrt5$ — примерно 0.2236; $L_1$ perturbation (0.1,0) — 0.2. Это разные допустимые множества при одном численном radius. Линеаризация не гарантирует такой же прирост настоящего nonlinear loss. Мотивация одношагового метода дана в [FGSM paper](https://arxiv.org/abs/1412.6572).

Описанная sign-атака увеличивает loss истинной метки и является untargeted. Если цель — добиться конкретного другого класса, objective меняется: можно уменьшать loss целевого класса при тех же constraints. Направление шага надо выводить из выбранной цели, а не механически копировать знак.

### PGD с точным учётом границ pixels

Для $L_\infty$ случая шаг с размером alpha записывается как ascent с последующей projection на пересечение исходного norm ball и допустимого диапазона:

$$
x^{k+1}=\operatorname{clip}_{[\max(0,x-\epsilon),\min(1,x+\epsilon)]}
\left(x^k+\alpha\operatorname{sign}(\nabla_{x^k}\ell)\right).
$$

Исходный x в границах остаётся центром ball на всех шагах. Если каждый раз строить ball вокруг текущего xk, perturbation постепенно выйдет за исходный radius. Random start внутри разрешённой области и несколько restarts исследуют разные траектории; они повышают силу поиска, но не доказывают нахождение глобального inner maximum.

Учебный пример: x=(0.95,0.20), epsilon=0.1, alpha=0.08, signs gradient постоянно (+,−). Разрешённые coordinate intervals: `[0.85,1]` и `[0.10,0.30]`. Первый шаг из x после clipping даёт (1,0.12), второй — (1,0.10). Итоговая delta=(0.05,−0.10) удовлетворяет norm budget. Для линейного g=(2,−1) прирост 0.2, а не 0.3: первая coordinate ограничена верхней границей pixels. Многошаговая robust optimization рассматривается в [работе Madry и соавторов](https://arxiv.org/abs/1706.06083).

### Что измеряет атака

Успешная допустимая perturbation является свидетельством ошибки на конкретном объекте. Неуспех одного алгоритма не доказывает отсутствия других perturbations в том же ball. Если оценивать, осталась ли модель верной после всех найденных атак, можно переоценить настоящую robust accuracy: часть вредных perturbations могла остаться ненайденной.

Сохраняют clean accuracy, attacked accuracy, norm/radius, число steps, step size, restarts, preprocessing и mode модели. Дополнительно проверяют фактическую norm delta и диапазон pixels. Резкое «улучшение защиты» при исчезновении gradients может быть gradient masking, а не реальной устойчивостью; для stochastic или nondifferentiable компонентов может потребоваться адаптированный способ оценки. Это всё ещё эмпирическая проверка, которую отделяют от формального сертификата.

Adversarial training увеличивает вычисления и меняет objective. Число inner forwards/backwards и outer updates входит в budget. Улучшение attacked quality может сопровождаться изменением clean quality; сравнивают обе стороны. Нельзя приписывать эффект только методу, если одной модели дали существенно больше обучения без соответствующего контроля.

## VAT без истинной метки

Virtual Adversarial Training сравнивает собственные distributions модели:

$$
r_{\mathrm{vadv}}\approx
\arg\max_{\|r\|\le\epsilon}
D_{\mathrm{KL}}\bigl(p_{\bar\theta}(\cdot\mid x)
\Vert p_\theta(\cdot\mid x+r)\bigr).
$$

В первой distribution используется остановленный gradient, обозначенный $\bar\theta$. Затем штрафуется KL в найденном направлении. Labels (метки) не нужны для этой части objective; она может дополнять supervised loss на размеченном subset. Direction находят приближённо, поскольку точный inner maximum дорог. [VAT](https://arxiv.org/abs/1704.03976).

Если исходный prediction ошибочен, consistency может укреплять ошибку. Поэтому smoothness prior не заменяет supervised signal и не гарантирует правильность на любой unlabeled population. Domain mismatch или возмущения через истинную границу classes способны вредить.

### Почему нельзя начать с обычного sign gradient в нуле

Пусть $D(r)=KL(p_{\bar\theta}(\cdot\mid x)\Vert p_\theta(\cdot\mid x+r))$, reference distribution зафиксирована. При r=0 обе distributions совпадают, divergence=0, и при гладкости gradient по r в этой точке равен нулю. Поэтому прямой FGSM от нулевого r не найдёт полезного направления.

Локально $D(r)\approx\tfrac12r^\top H r$, где H — Hessian divergence по perturbation. На $L_2$ ball максимальное quadratic изменение задаётся ведущим eigenvector H. VAT приближает его: выбирает маленькое случайное направление d, оценивает input gradient divergence при r=xi·d, нормирует полученный vector и использует его для adversarial radius epsilon. Несколько power-like шагов могут уточнять направление. xi задаёт масштаб локального поиска, epsilon — итогового perturbation; это разные параметры.

### Учебная геометрия Hessian

Для $H=\operatorname{diag}(9,1)$ и стартового единичного направления $(1,1)/\sqrt2$ quadratic coefficient $d^\top Hd$ равен 5. Умножение на H и нормировка даёт $(9,1)/\sqrt{82}$, коэффициент становится $730/82\approx8.9024$, ближе к максимуму 9. При epsilon=0.1 quadratic divergence увеличивается с 0.025 до примерно 0.04451; истинный максимум этой квадратичной модели равен 0.045.

Это объяснение локального приближения, а не обещание точного eigenvector для произвольного nonlinear classifier. Во внешнем VAT-loss направление обычно фиксируют, а gradient оптимизирует prediction возмущённого входа относительно detached reference. Механизм подробно изложен в [VAT paper](https://arxiv.org/abs/1704.03976).

### Smoothness и правильность различаются

Если модель уверенно ошибается, можно сделать её ошибку устойчивой в окрестности. Unlabeled consistency не сообщает автоматически правильную метку. Риск особенно велик при несовпадении unlabeled domain, нарушении label-preserving assumptions или раннем случайном prediction. Supervised loss, качество views, расписание коэффициента и отбор данных остаются существенными.

Для p=(0.8,0.2), q=(0.6,0.4) $KL(p\Vert q)\approx0.09152$, а $KL(q\Vert p)\approx0.10465$. Divergence направлена; менять местами reference и student (модель-ученик) без пересмотра objective нельзя. Сумма квадратов разности этих же vectors равна 0.08 и имеет другую шкалу. Коэффициент consistency нельзя механически переносить между KL и MSE либо между разными reductions.

## Consistency и согласование views

Consistency objective для двух допустимых stochastic views:

$$
\mathcal L=L_{\mathrm{sup}}+
\lambda\,d\bigl(p_\theta(\cdot\mid T_1x),
p_{\mathrm{teacher}}(\cdot\mid T_2x)\bigr).
$$

В Π-model сравниваются два stochastic прохода; у Mean Teacher (модель-учитель) teacher weights — EMA student weights. Teacher target обычно detached. Это отличается от отдельного постоянно обучаемого teacher через тот же gradient и от uniform SWA. Ранняя ramp-up consistency может быть полезна, чтобы не усиливать случайные initial predictions. [Mean Teacher](https://arxiv.org/abs/1703.01780).

Для dense predictions нужно выровнять spatial coordinates двух views. Если один image flipped, а другой нет, нельзя сравнить probability maps pixelwise без обратного mapping. Consistency должна относиться к одному semantic location.

### Откуда берётся target

Π-model использует два stochastic прохода одной модели, например с разными dropout masks и допустимыми аугментациями. В симметричной squared consistency оба прохода могут получать gradient; наличие двух outputs ещё не означает frozen teacher. Mean Teacher задаёт отдельный teacher state, обновляемый EMA весов student, и использует его prediction как target без обычного gradient update teacher этим же loss.

Если teacher вес равен 1, новый student вес 3, alpha=0.9, после EMA teacher станет 1.2. Uniform average двух состояний дал бы 2. Усреднение weights не равно усреднению probability outputs нелинейной модели. Кроме EMA-параметра задают начало обновлений, инициализацию и buffers. Исходная схема и мотивация представлены в [Mean Teacher](https://arxiv.org/abs/1703.01780).

Слишком сильная consistency в начале может закрепить неинформативные outputs, поэтому иногда используют ramp-up. Confidence (уверенность по принятой оценке) gating оставляет только выбранные teacher-targets, но меняет состав обучаемых объектов и может исключить редкие классы. Если ни один объект не прошёл gate, знаменатель среднего loss требует отдельной обработки. Unlabeled часть, число предъявлений и стоимость teacher forward входят в экспериментальный протокол.

### Dense prediction требует общей системы координат

У segmentation probability map относится к spatial locations. Если teacher-view имеет foreground probabilities (0.9,0.2) слева направо, а student видит horizontal flip и правильно выдаёт (0.2,0.9), прямой MSE между ними равен 0.49. После обратного flip student-map совпадает с teacher-map, consistency error равна нулю. Без alignment loss штрафовал бы правильную equivariance.

При разных crop сравнивают только общую область, для которой есть соответствие; padding (дополнение до общей длины) и исчезнувшие области исключают объявленной mask. Для keypoints дополнительно переставляют left/right IDs. Photometric views могут не требовать пространственного mapping, но всё равно должны сохранять информацию о target. Эти требования относятся к геометрии данных независимо от того, использует модель CNN или Transformer-backbone.

## Spectral norm и ограничения вывода

Для linear layer $Wx$ по Euclidean norm:

$$
\|Wx-Wx'\|_2\le\|W\|_2\|x-x'\|_2,
\qquad
\widetilde W=\frac{W}{\sigma_{\max}(W)}.
$$

Spectral norm — максимальное singular value, а не Frobenius norm всех coefficients. Для diagonal $W=\operatorname{diag}(3,4)$ spectral norm 4, Frobenius norm 5. Нормировка делением на 4 ограничивает linear operator norm единицей.

В реализации singular value часто приближается power iteration; для convolution матрица flattened kernel и полный spatial operator могут иметь разные norms. Bound всей network требует учесть composition: для последовательности constants перемножаются, для residual sum складываются. Spectral normalization (нормировка по выбранному правилу) стабилизирует некоторые training regimes, но не даёт universal robustness certificate всей системы автоматически.

### Матрица и её сильнейшее направление

Spectral norm $\|W\|_2=\max_{\|v\|=1}\|Wv\|$ равна наибольшему singular value. Для diag(3,4) деление на 4 даёт diag(0.75,1), максимальное усиление 1 достигается на направлении (0,1). Frobenius norm равна 5; деление на неё дало бы максимальное усиление 0.8 и другой слой. Обе нормы информативны, но не взаимозаменяемы.

Power iteration приближает главное направление через чередование W и W transpose либо через W transpose W. Для $W=\operatorname{diag}(3,4)$ и стартового $v=(1,1)^\top/\sqrt2$ один шаг $W^\top W$ с нормировкой даёт $v=(9,16)^\top/\sqrt{337}$. Величина $\|Wv\|$ примерно 3.784, ещё меньше истинной norm 4. Если делить W на такую недооценку, operator norm результата будет больше 1. Конечная итерация не является автоматически строгой верхней границей. Если старт не имеет компоненты в главном направлении, простой power method может вообще его не увидеть.

На практике сохраняют приближённые singular vectors между updates; нулевая norm требует обработки. Постановка и эффективная параметризация описаны в [Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957). Стабилизация конкретного training regime и доказанная устойчивость всей модели — разные результаты.

### Свёрточное ядро и полный spatial operator

Для одномерной valid convolution с kernel (1,1) и input длины 3 flatten kernel имеет norm $\sqrt2$. Но полный linear operator равен

$$
W_{conv}=\begin{pmatrix}1&1&0\\0&1&1\end{pmatrix}.
$$

Его spectral norm равна $\sqrt3$. Если нормировать только flattened kernel делением на $\sqrt2$, полный operator получит norm $\sqrt{3/2}\approx1.2247$, а не 1. Повторное использование coefficients на overlapping позициях меняет усиление. Stride (шаг пространственного перемещения операции), padding и spatial shape также влияют на полный operator, поэтому гарантию надо связывать с действительно нормируемым объектом.

### Почему ограничения отдельных слоёв недостаточно для любого сертификата

Для последовательной композиции Lipschitz constants перемножаются. Два linear layers с constants 2 и 3 и промежуточной 1-Lipschitz ReLU дают верхнюю границу 6. Для residual $x+F(x)$ bound равен $1+L_F$; если F(x)=2x, реальный constant равен 3. Поэтому residual connection не сохраняет автоматически constant 1 нормированной ветви.

Нужно также учитывать nonlinearities, learned normalization scales, epsilon и зависимости attention (механизм внимания) от самого input. Не каждая activation имеет constant 1 в выбранной norm, а normalization может сильно менять локальную чувствительность около малой variance. Чтобы получить robustness certificate решения по классам, кроме bound чувствительности необходима связь с margin и областью допустимых perturbations. Одного включённого spectral-norm wrapper для такого вывода недостаточно.

## Контроль и практика

1. Чем adversarial training отличается от VAT? **Ответ:** supervised вариант увеличивает loss относительно true label (метка); VAT ищет изменение собственной distribution и может использовать unlabeled inputs.
2. Почему consistency может вредить? **Ответ:** неправильный prior/view или ошибочный teacher закрепляет неверные predictions.
3. Равны ли spectral и Frobenius norms? **Ответ:** нет; первая — maximum singular value, вторая — корень суммы их квадратов.

**Практика:** выведите FGSM для двух coordinates и сравните $L_2$/$L_\infty$ perturbations. Для flipped segmentation view опишите mapping перед consistency loss. Составьте clean/robust evaluation с фиксированным radius, norm, attack budget и train-only tuning.



1. **Почему FGSM и VAT начинают поиск по-разному?** Supervised loss обычно имеет ненулевой input gradient; self-KL в точке совпадения distributions имеет нулевой gradient и требует поиска направления локальной кривизны.
2. **Почему teacher EMA не является доказанной правильной меткой?** Это сглаженное состояние модели, которое может сохранять её ошибки.
3. **Почему PGD failure не является сертификатом?** Алгоритм мог не найти существующее допустимое adversarial изменение.
4. **Почему kernel spectral norm не всегда ограничивает всю convolution?** Полный spatial operator повторно использует kernel на пересекающихся позициях.

Практика: вручную воспроизвести constrained PGD и проверить raw/normalized radius. Для двух dense views восстановить координатное соответствие. Сравнить matrix и convolution norms, затем записать clean/robust evaluation protocol с фиксированными budgets. Все attacks, distributions и matrices главы учебные; пользовательская модель не атаковалась и её robust accuracy не измерялась.

## Результат главы и переход дальше

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
