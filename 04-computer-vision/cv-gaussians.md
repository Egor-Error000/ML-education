# 3D Gaussian Splatting: проекция, compositing и densification

**Уровень:** углубление.

**Предпосылки:** [NeRF: лучи, radiance field и volume rendering](<cv-nerf.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как явные 3D Gaussian primitives заменяют многократные запросы neural field?

## Схема вычислений

![3D Gaussian Splatting: проекция, compositing и densification: вычислительные этапы](<../оформление/course/expanded-cv-gaussians.svg>)

3DGS optimization: SfM points + camera poses → Means / scales / rotations / SH → Differentiable splat renderer → Photo loss → densify / prune. Rendering: 3D Gaussians + new camera → Project covariance → 2D ellipse → Tile binning + depth ordering → Alpha compositing → RGB.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Явное представление сцены

3D Gaussian Splatting, 3DGS, хранит набор anisotropic Gaussian primitives. Каждая имеет mean $\mu\in\mathbb R^3$, covariance $\Sigma$, opacity и view-dependent colour, обычно spherical harmonics coefficients. Это оптимизируемая сцена, а не один pretrained classifier. [Kerbl et al.](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) начинают с sparse points SfM и camera poses, затем оптимизируют representation по photographs.

## Covariance и проекция

Covariance параметризуют через scale и rotation: $\Sigma=RSS^TR^T$, что позволяет сохранять positive semidefiniteness. Projected 2D covariance приближают с Jacobian перспективной проекции: $\Sigma_{2D}=JW\Sigma W^TJ^T$, где $W$ — linear world-to-camera transform. Из 3D ellipsoid получается screen-space ellipse. Это local approximation perspective; near-plane и numerical issues требуют обработки.

Renderer определяет screen-space footprint, распределяет primitives по tiles и упорядочивает для compositing. Вклад Gaussian зависит от расстояния pixel до projected mean и opacity. Цвет собирают front-to-back через transmittance, как идею compositing, но samples здесь явные projected primitives, а не произвольные точки вдоль ray в NeRF.

## Оптимизация и densification

Photometric objective сочетает pixel reconstruction и structural similarity component. Gradient меняет positions, scale, rotations, opacity и colour parameters. Density control добавляет, клонирует или разделяет primitives там, где текущая модель недостаточна; pruning удаляет ненужные primitives. Количество элементов меняется, поэтому память не описывается только размером начального cloud.

На inference новые camera poses проецируют уже обученную scene. Результат зависит от view coverage и camera calibration. Прозрачные, отражающие и тонкие structures могут давать floaters и неверную geometry; real-time results статьи относятся к определённому hardware и scene workload.

## Разобранный пример: compositing и память

Пусть два projected primitives на pixel дают $\alpha_1=0.6$, $\alpha_2=0.5$ и scalar colours 1 и 0.2. Первый вклад 0.6. Transmittance перед вторым 0.4, второй вклад $0.4\cdot0.5\cdot0.2=0.04$. Black-background output 0.64. В обратном порядке получаем $0.5\cdot0.2+0.5\cdot0.6\cdot1=0.4$. Depth order существенно меняет изображение.

Для учебной записи с 3 mean, 3 scales, 4 quaternion rotation, 1 opacity и 48 colour coefficients нужно 59 float32 values на primitive, 236 bytes. Для миллиона primitives — 236000000 bytes, примерно 225.07 MiB только параметров. Optimizer states, gradients, sorting и rasterization buffers добавляются отдельно. Это объясняет, почему высокая скорость rendering не означает малую память обучения.

## Сравнение с NeRF

NeRF запрашивает continuous field вдоль rays; 3DGS rasterizes explicit splats. Различаются representation, renderer, density control и memory/compute tradeoffs. Сравнивают одинаковые scenes, poses, resolution и качество held-out views. Красивый render не гарантирует mesh без artifacts или правильный metric scale.

## Самопроверка с ответами

**Практика:** удвойте число primitives при том же формате. Память parameters удвоится, но rendering time зависит и от projected footprints и overlap.

**Почему нельзя произвольно менять covariance matrix?** Она должна задавать допустимую Gaussian shape; параметризация через scale и rotation помогает соблюдать это условие.

**Получаем ли universal генерацию новой сцены?** Нет, здесь оптимизируется representation наблюдаемой сцены для новых views.

## Иллюстрации из первоисточников

![gaussians: Figure 2 из авторской статьи](<../оформление/course/papers/gaussians.png>)

**Источник:** [gaussians, Figure 2](https://arxiv.org/abs/2308.04079), PDF, с. 5. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [3DGS — Kerbl et al., SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<06-geometry/План.md>)
