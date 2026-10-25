# NeRF: лучи, radiance field и volume rendering

**Уровень:** углубление.

**Предпосылки:** [Камера и геометрия: проекция, координаты и калибровка](<cv-camera.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** [3D Gaussian Splatting: проекция, compositing и densification](<cv-gaussians.md>)

**Учебный вопрос:** Как обучить представление сцены по фотографиям и отрисовать новый вид?

## Схема вычислений

![NeRF: лучи, radiance field и volume rendering: вычислительные этапы](<../оформление/course/expanded-cv-nerf.svg>)

NeRF field: Position x + direction d → Positional encoding → MLP → Density σ(x), colour c(x,d) → Обучаемое поле одной сцены. Rendering / training: Camera pose → pixel ray → Coarse / fine ray samples → Transmittance → weighted RGB → Image MSE → field gradients.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Scene representation вместо classifier

NeRF описывает одну сцену функцией, которая по 3D position $x$ и viewing direction $d$ выдаёт density $\sigma(x)$ и цвет $c(x,d)$. В исходной модели это MLP с positional encoding координат. Density не зависит от direction, цвет может зависеть. [Mildenhall et al.](https://arxiv.org/abs/2003.08934) обучают scene-specific field по calibrated views; это не готовый universal generator произвольных сцен.

## Лучи и volume rendering

Для pixel камера задаёт ray $r(t)=o+td$, где $o$ — camera center. Выбирают samples $t_i$ и интервалы $\delta_i$. Для каждого sample field даёт density и RGB. Вероятность поглощения на интервале $\alpha_i=1-\exp(-\sigma_i\delta_i)$, transmittance до него $T_i=\prod_{j<i}(1-\alpha_j)$. Итоговый цвет $\hat C=\sum_iT_i\alpha_i c_i$, плюс background contribution при неполной opacity.

Density измеряется в обратных единицах длины, чтобы $\sigma_i\delta_i$ было безразмерным. Без этого scale scene и шагов sampling меняет rendering непредсказуемо. Positional encoding добавляет sinusoidal features разных частот, помогая представить высокочастотные детали; оно не создаёт новые наблюдения.

## Обучение и применение

Выбирают rays из train images, render colours и минимизируют pixel MSE с observed RGB. Gradient проходит через volume integration в field parameters. Исходный NeRF использует coarse и fine networks с hierarchical sampling: fine samples чаще располагаются в важных частях ray. На inference задают новую camera pose и render каждый pixel; GT нового вида не используется.

Known poses, static scene и достаточное coverage входят в assumptions. Незнакомый ракурс за пределами наблюдений может дать artifacts. Успешное восстановление RGB не гарантирует точную surface geometry: разные fields способны объяснять ограниченные observations.

## Разобранный пример: compositing двух samples

Два samples имеют $\alpha_1=0.5$, $\alpha_2=0.25$, scalar colours 0.8 и 0.2. $T_1=1$, $T_2=1-0.5=0.5$. Weights: 0.5 и 0.125. Foreground contribution $0.5\cdot0.8+0.125\cdot0.2=0.425$. Remaining transmittance $0.5\cdot0.75=0.375$. При белом background полный colour равен 0.8, при чёрном — 0.425.

Если $\delta=0.1$ m и $\alpha=0.5$, density $\sigma=-\ln(1-0.5)/0.1\approx6.93147$ m$^{-1}$. Для $800\times800$ pixels и 128 samples на ray требуется 81920000 field evaluations до дополнительных coarse passes. Это объясняет цену naive rendering.

## Нюансы оценки

PSNR измеряет pixel error, но не полностью perceptual fidelity и geometry. Train views и held-out views оценивают раздельно. Нельзя подбирать reconstruction по test views и затем называть их независимыми. Для dynamic scenes требуется отдельная модель времени или deformation.

## Самопроверка с ответами

**Практика:** поменяйте порядок samples. Transmittance weights изменятся; volume rendering учитывает occlusion по направлению луча.

**Почему сумма weights может быть меньше единицы?** Часть света проходит сквозь все интервалы к background.

**Получает ли NeRF depth labels обязательно?** Исходный pipeline может обучаться по RGB и poses; это не делает любую полученную depth карту метрически достоверной.

## Иллюстрации из первоисточников

![nerf: Figure 2 из авторской статьи](<../оформление/course/papers/nerf.png>)

**Источник:** [nerf, Figure 2](https://arxiv.org/abs/2003.08934), PDF, с. 5. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [NeRF — Mildenhall et al., ECCV 2020](https://arxiv.org/abs/2003.08934)

[Следующие углубления: cv-gaussians](<README.md>) · [Сценарии подраздела](<06-geometry/План.md>)
