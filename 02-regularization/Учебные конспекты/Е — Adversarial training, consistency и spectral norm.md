# Е. Устойчивость predictions к допустимым возмущениям

[Оглавление](README.md). Пункт 6 плана. **Происхождение:** новое углубление; оно не приписывается исходным CV/LLM-чатам как пройденное занятие.

## 1. Сначала определить допустимое изменение

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

PGD выполняет несколько gradient-ascent steps с projection обратно в $\Delta$. Область допустимых pixels тоже учитывается. Один слабый attack не доказывает robustness ко всем attacks; gradient masking может создавать ложное впечатление защиты.

## 2. Почему FGSM имеет такой вид

Линеаризуем loss: $\ell(x+\delta)\approx\ell(x)+g^\top\delta$. При $|\delta_j|\le\epsilon$ каждый summand максимален при $\delta_j=\epsilon\operatorname{sign}(g_j)$. Для $g=(2,-1),\epsilon=0.1$ берём $(0.1,-0.1)$ и predicted increase равен $0.3$. В $L_2$ ball направление другое: $\delta=\epsilon g/\|g\|_2$. Radius без указания norm не определяет один и тот же attack.

Для tokens нельзя непосредственно прибавить дробное число к token ID. Возможны perturbations embeddings или discrete substitutions, но они задают разные threat models и constraints. Даже малый embedding noise не обязательно соответствует допустимому человеческому перефразированию.

## 3. VAT: метка не нужна для local consistency

Virtual Adversarial Training сравнивает собственные distributions модели:

$$
r_{\mathrm{vadv}}\approx
\arg\max_{\|r\|\le\epsilon}
D_{\mathrm{KL}}\bigl(p_{\bar\theta}(\cdot\mid x)
\Vert p_\theta(\cdot\mid x+r)\bigr).
$$

В первой distribution используется остановленный gradient, обозначенный $\bar\theta$. Затем штрафуется KL в найденном направлении. Labels не нужны для этой части objective; она может дополнять supervised loss на размеченном subset. Direction находят приближённо, поскольку точный inner maximum дорог. [VAT](https://arxiv.org/abs/1704.03976).

Если исходный prediction ошибочен, consistency может укреплять ошибку. Поэтому smoothness prior не заменяет supervised signal и не гарантирует правильность на любой unlabeled population. Domain mismatch или возмущения через истинную границу classes способны вредить.

## 4. Π-model и Mean Teacher

Consistency objective для двух допустимых stochastic views:

$$
\mathcal L=L_{\mathrm{sup}}+
\lambda\,d\bigl(p_\theta(\cdot\mid T_1x),
p_{\mathrm{teacher}}(\cdot\mid T_2x)\bigr).
$$

В Π-model сравниваются два stochastic прохода; у Mean Teacher teacher weights — EMA student weights. Teacher target обычно detached. Это отличается от отдельного постоянно обучаемого teacher через тот же gradient и от uniform SWA. Ранняя ramp-up consistency может быть полезна, чтобы не усиливать случайные initial predictions. [Mean Teacher](https://arxiv.org/abs/1703.01780).

Для dense predictions нужно выровнять spatial coordinates двух views. Если один image flipped, а другой нет, нельзя сравнить probability maps pixelwise без обратного mapping. Consistency должна относиться к одному semantic location.

## 5. Spectral normalization

Для linear layer $Wx$ по Euclidean norm:

$$
\|Wx-Wx'\|_2\le\|W\|_2\|x-x'\|_2,
\qquad
\widetilde W=\frac{W}{\sigma_{\max}(W)}.
$$

Spectral norm — максимальное singular value, а не Frobenius norm всех coefficients. Для diagonal $W=\operatorname{diag}(3,4)$ spectral norm 4, Frobenius norm 5. Нормировка делением на 4 ограничивает linear operator norm единицей.

В реализации singular value часто приближается power iteration; для convolution матрица flattened kernel и полный spatial operator могут иметь разные norms. Bound всей network требует учесть composition: для последовательности constants перемножаются, для residual sum складываются. Spectral normalization стабилизирует некоторые training regimes, но не даёт universal robustness certificate всей системы автоматически.

## 6. Контроль и практика

1. Чем adversarial training отличается от VAT? **Ответ:** supervised вариант увеличивает loss относительно true label; VAT ищет изменение собственной distribution и может использовать unlabeled inputs.
2. Почему consistency может вредить? **Ответ:** неправильный prior/view или ошибочный teacher закрепляет неверные predictions.
3. Равны ли spectral и Frobenius norms? **Ответ:** нет; первая — maximum singular value, вторая — корень суммы их квадратов.

**Практика:** выведите FGSM для двух coordinates и сравните $L_2$/$L_\infty$ perturbations. Для flipped segmentation view опишите mapping перед consistency loss. Составьте clean/robust evaluation с фиксированным radius, norm, attack budget и train-only tuning.
