# К. Как выбрать следующий эксперимент и не переобучиться на validation

[Оглавление](README.md). Объединены пункты 12 и 14 и итоговая практика плана. **Происхождение:** дополнительный материал; названия «передовые/SOTA» в плане рассматриваются как исследовательский обзор, а не обещание лучшего текущего метода для любой задачи.

## 1. SAM: низкий loss в окрестности

Sharpness-Aware Minimization ищет weights с низким худшим loss в выбранной local neighborhood:

$$
\min_w\max_{\|\varepsilon\|_2\le\rho}L(w+\varepsilon).
$$

Линеаризация inner objective даёт perturbation

$$
\widehat\varepsilon=
\rho\frac{\nabla L(w)}{\|\nabla L(w)\|_2},
\qquad
w^+=w-\eta\nabla L(w+\widehat\varepsilon).
$$

В базовом приближённом update perturbation рассматривается как фиксированное смещение для второго gradient; это не полный differentiation через весь inner solver. При нулевой gradient norm нужно отдельное стабилизированное правило. Параметр $\rho$ задаёт radius, не learning rate. [SAM](https://arxiv.org/abs/2010.01412).

Пример для $L(w)=w^2/2$, $w=2,\rho=0.1,\eta=0.1$: первый gradient 2, perturbation 0.1, второй gradient 2.1, update даёт $w^+=1.79$. Обычный GD дал бы 1.8. В этом scalar convex примере нет загадочной «магии flat basins»; видно, как меняется шаг. Для network эффект зависит от data и geometry.

SAM обычно требует дополнительного forward/backward и аккуратного управления stateful layers. Если BatchNorm statistics обновить дважды несогласованно, сравнение может смешать эффект optimizer и statistics. Конкретная реализация должна явно определить state handling. Сравнивать нужно и quality, и compute.

## 2. Почему flatness не абсолютна

В homogeneous networks weights соседних layers можно rescale так, чтобы функция не изменилась, но local parameter-space sharpness изменилась. Поэтому «более плоский минимум всегда лучше обобщается» без указания norm и parametrization неверно. [Исследование reparametrization](https://arxiv.org/abs/1703.04933) показывает ограничения наивной flatness-интуиции.

Это не делает SAM бессмысленным; это ограничивает теоретические выводы из одного geometry metric. Обоснованный итог звучит как measured improvement в заданном protocol, а не universal theorem о всех плоских weights.

## 3. Lookahead: быстрые и медленные weights

Fast optimizer делает $k$ steps от slow weights $\phi$. Затем slow weights интерполируются:

$$
\phi^+=\phi+\alpha(w_k-\phi),
\qquad w\leftarrow\phi^+.
$$

Если $\phi=1,w_k=3,\alpha=0.5$, новые slow и reset fast weights равны 2. Механизм сглаживает trajectory выбранного inner optimizer. Это не тот же min–max objective, что SAM, и не average всех checkpoints как SWA. [Lookahead](https://arxiv.org/abs/1907.08610).

У fast и slow states есть свои checkpoint requirements. При возобновлении training недостаточно сохранить только текущие fast parameters, если хотим продолжить тот же алгоритм. Наблюдаемая стабильность не гарантирует лучшую held-out quality на любой задаче.

## 4. Диагностика перед sweep

Сначала проверяем learning loop на маленьком batch: верные labels, shapes, masks, gradient flow и возможность уменьшить training loss. Это диагностический опыт, а не proof generalization. Затем baseline на полном train, корректный split и learning curves. Если модель underfit из-за отсутствия нужного feature, увеличение dropout не создаст информацию.

Выбираем один механизм, соответствующий гипотезе: L2 при нестабильных coefficients, data augmentation при обоснованной invariance, early stopping при degradation поздних checkpoints. Отсутствие training progress сначала требует проверить optimizer/LR/data. Distribution shift требует анализа срезов и sources.

## 5. Как подбирать strength

Начальный логарифмический sweep lambda, например от $10^{-5}$ до $10^{-1}$, исследует порядки magnitude. Диапазон зависит от normalization loss, model и optimizer; эти числа не универсальная рекомендация. Обязательно есть baseline без выбранного penalty и несколько seeds для finalist configurations.

Настройки взаимодействуют: dropout, augmentation, weight decay, LR и epochs меняют effective training difficulty. Сначала полезна ablation по одному фактору, затем совместный подбор наиболее перспективных. Bayesian optimization/Optuna — инструменты поиска, которые не устраняют validation overfitting и не делают test пригодным для tuning.

Универсального закона $\lambda\propto1/n$ нет. Он зависит от постановки objective, prior interpretation, noise и class модели. При удвоении числа examples нужно также понять, оставили ли steps фиксированными или удвоили compute.

Итоговый журнал включает split/version, objective convention, training budget, curves, validation metric, calibration, seeds, runtime и final independent test. Не следует объявлять winner по одному случайному seed или по лучшей метрике из множества просмотренных tests.

## 6. Контроль и практика

1. Отличаются ли radius SAM и LR? **Ответ:** radius определяет inner perturbation neighborhood, LR — внешний update step.
2. Эквивалентен ли Lookahead SWA? **Ответ:** нет; slow weights периодически перемещаются к fast weights, после чего fast reset.
3. Когда больше regularization ухудшает обе curves? **Ответ:** при слишком сильном ограничении или уже существующем underfit; также нужно исключить optimization/data ошибки.

**Практика:** повторите scalar SAM/Lookahead examples. Затем спроектируйте журнал из baseline и шести configurations одной регуляризации: hypothesis, strength, одинаковый budget, validation result и seeds. Заранее запишите правило выбора finalist и закрытый test; отсутствие improvement тоже является содержательным результатом.
