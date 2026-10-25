# PointNet и PointNet++: обучение на множествах точек

**Уровень:** углубление.

**Предпосылки:** [Камера и геометрия: проекция, координаты и калибровка](<cv-camera.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** [Voxels, PointPillars и BEV: представления для 3D detection](<cv-bev.md>)

**Учебный вопрос:** Как сеть обрабатывает облако точек без заданного порядка?

## Схема вычислений

![PointNet и PointNet++: обучение на множествах точек: вычислительные этапы](<../оформление/course/expanded-cv-pointnet.svg>)

PointNet: N×3 points: unordered → Shared MLP → N×D → Symmetric max → global D → Class head / point-wise head. PointNet++: Points + features → Sample centers → group neighbours → Local PointNet → hierarchy → Feature propagation → labels.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Множество вместо grid

Point cloud — набор $N$ points с координатами и, возможно, цветом, интенсивностью или нормалями. Порядок строк не должен менять class всего объекта. В segmentation permutation входа должна согласованно переставлять point predictions. Эти свойства называют invariance и equivariance соответственно. Размер $N$ может меняться; padding требует маски, чтобы фиктивные points не влияли на aggregation.

## PointNet

[PointNet](https://arxiv.org/abs/1612.00593) применяет shared MLP к каждой точке, получая $h_i\in\mathbb R^D$, затем symmetric max pooling $g_d=\max_i h_{i,d}$. Порядок не влияет на максимум. Global vector классифицируется через head; для segmentation его повторяют для каждой точки и соединяют с local features. Training использует CE соответствующего уровня.

T-Net может предсказывать transform координат или features, но это обучаемое выравнивание, не строгая rotation invariance. Для feature transform используется regularization близости к ортогональному преобразованию. Global pooling ограничивает непосредственное описание соседств: точка обрабатывается одинаково независимо от nearby geometry до общей агрегации.

## PointNet++

[PointNet++](https://arxiv.org/abs/1706.02413) строит hierarchy local regions. Sampling выбирает centers, grouping собирает neighbours по ball radius или другому правилу, local PointNet агрегирует features области. Повторение set abstraction увеличивает spatial context. Для dense segmentation feature propagation интерполирует признаки обратно к более подробным point sets и объединяет skip features.

Radius задаётся в units координат; rescale облака без согласования radius меняет соседства. Farthest point sampling улучшает spatial coverage, но не заменяет плотные measurements. Multi-scale grouping помогает работе с разной sampling density, однако sparse далёкие points всё равно несут меньше информации.

## Разобранный пример: symmetric aggregation

Три point features: $(1,4)$, $(3,2)$, $(2,5)$. Max pooling даёт $(3,5)$. После любой перестановки результат тот же. Ни одна отдельная точка не имела vector $(3,5)$: global representation собирает компоненты разных points. Добавление point $(0,0)$ ничего не меняет, а outlier с feature $(100,0)$ меняет первую координату на 100. Invariance к порядку не равна устойчивости ко всем выбросам.

Для 1024 points и $D=64$ shared features содержат 65536 numbers. Повторённый global vector длины 1024 для каждой точки создаёт $1024^2=1048576$ values перед segmentation head. Локальная hierarchy уменьшает число centers на поздних stages, но grouping и nearest neighbour search тоже стоят времени и памяти.

## Ошибки и выбор

Смешивание camera и world frames приводит к ложным геометрическим признакам. Случайный subsampling может удалить редкий маленький объект. Нормировка каждого объекта к unit sphere удобна для object classification, но уничтожает absolute scale, который может быть важен для scene detection. Проверяют missing points, sensor range и density slices.

## Самопроверка с ответами

**Практика:** замените max pooling на concatenation входных строк. Перестановка теперь меняет global vector, и desired invariance потеряна.

**Гарантирует ли T-Net устойчивость к любому повороту?** Нет: это обучаемый компонент с ограничениями данных и objective.

**Зачем PointNet++ radius в metres?** Он определяет физический масштаб локальной структуры; те же числа в centimetres создадут другое соседство.

## Иллюстрации из первоисточников

![pointnet: Figure 2 из авторской статьи](<../оформление/course/papers/pointnet.png>)

**Источник:** [pointnet, Figure 2](https://arxiv.org/abs/1612.00593), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![pointnet2: Figure 2 из авторской статьи](<../оформление/course/papers/pointnet2.png>)

**Источник:** [pointnet2, Figure 2](https://arxiv.org/abs/1706.02413), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [PointNet — Qi et al., CVPR 2017](https://arxiv.org/abs/1612.00593)
- [PointNet++ — Qi et al., NeurIPS 2017](https://arxiv.org/abs/1706.02413)

[Следующие углубления: cv-bev](<README.md>) · [Сценарии подраздела](<06-geometry/План.md>)
