# Voxels, PointPillars и BEV: представления для 3D detection

**Уровень:** углубление.

**Предпосылки:** [PointNet и PointNet++: обучение на множествах точек](<cv-pointnet.md>), [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как преобразовать 3D measurements в удобное представление для detection?

## Схема вычислений

![Voxels, PointPillars и BEV: представления для 3D detection: вычислительные этапы](<../оформление/course/expanded-cv-bev.svg>)

PointPillars: LiDAR N×(xyz,intensity) → Pillars → point decoration → Shared encoder + max → scatter → 2D BEV backbone → 3D boxes. Sensor fusion: LiDAR points + camera image → Time sync + extrinsic transform → Project / lift → common frame → Fuse features → task head.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Выбор представления

Dense voxel grid делит пространство на 3D cells; стоимость растёт с объёмом, включая пустоту. Sparse convolution обрабатывает занятые cells, но требует индексов и подходящих kernels. Bird's-eye view, BEV, задаёт карту по ground plane $(x,y)$ с learned features; height information кодируется в channels или агрегируется. Это не фотография сверху, а координатное представление.

## PointPillars

[PointPillars](https://arxiv.org/abs/1812.05784) разбивает cloud на вертикальные columns, pillars. Для points добавляются признаки положения относительно среднего pillar и его центра. Shared PointNet-подобный encoder и max pooling дают один vector на pillar. Scatter размещает vectors в 2D pseudo-image; обычный 2D backbone и detection head предсказывают 3D boxes.

Pillar не делится на множество отдельных height voxels в этой схеме. Высота всё ещё присутствует среди входных point features, но aggregation может потерять часть вертикальной структуры. Caps на число pillars и points ограничивают память и задают политику отбора при переполнении. Они влияют на качество в плотных регионах.

## Boxes, targets и fusion

3D box может задаваться $(x,y,z,l,w,h,\theta)$ в sensor или world frame. Head дополнительно предсказывает class и иногда velocity. Assignment, regression normalization, yaw convention и rotated NMS относятся к конкретному detector. Применение 2D IoU к projections не равно 3D IoU и может скрывать ошибку высоты.

Sensor fusion требует extrinsics и timestamps. LiDAR point можно проектировать в image через rigid transform и camera intrinsics, затем извлекать image features. Camera-to-BEV lift требует depth hypotheses или иной geometry: одному image pixel соответствует ray. Unsynchronized sensors при движении создают смещение даже при правильной static calibration.

## Разобранный пример: discretization и объём

Область $x\in[0,70.4)$ m, $y\in[-40,40)$ m и cell size 0.2 m даёт $352\times400=140800$ pillars. Point $(10.1,-3.9,1.2)$ имеет индексы $i_x=\lfloor10.1/0.2\rfloor=50$, $i_y=\lfloor36.1/0.2\rfloor=180$. Индексация должна учитывать выбранные axis order и half-open bounds.

Pseudo-image с 64 channels содержит $140800\cdot64=9011200$ values, 34.375 MiB float32. Dense voxel grid с той же xy-grid, высотой 8 m и z-step 0.2 m имеет 40 layers: $5632000$ cells. При 64 features это уже 360448000 values, 1375 MiB без intermediate tensors. Sparse methods экономят пустые cells, поэтому сравнение с dense baseline не описывает их полную стоимость.

## Самопроверка с ответами

**Практика:** уменьшите xy cell size вдвое. Число BEV cells увеличится примерно вчетверо; voxel grid при одновременном уменьшении z-step — примерно в восемь раз.

**Теряет ли pillar всю информацию о z?** Нет, z может кодироваться в point features, но после pooling остаётся сжатое представление.

**Почему image detector нельзя просто назвать BEV detector?** Системы координат и targets различны; нужна явная геометрия преобразования.

## Иллюстрации из первоисточников

![pointpillars: Figure 2 из авторской статьи](<../оформление/course/papers/pointpillars.png>)

**Источник:** [pointpillars, Figure 2](https://arxiv.org/abs/1812.05784), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [PointPillars — Lang et al., CVPR 2019](https://arxiv.org/abs/1812.05784)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<06-geometry/План.md>)
