# DeepLabv3+ и SegFormer: контекст и декодирование

**Уровень:** основа ветки.

**Предпосылки:** [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>), [Vision Transformer: patches, attention и обучение](<cv-vit.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как DeepLabv3+ и SegFormer объединяют контекст с пространственными деталями?

## Схема вычислений

![DeepLabv3+ и SegFormer: контекст и декодирование: вычислительные этапы](<../оформление/course/expanded-cv-segformers.svg>)

DeepLabv3+: CNN с atrous features → ASPP: rates + global pooling → Up + low-level skip → Decoder → semantic logits. SegFormer: Overlapping patch stages → MiT: 4 scales of features → MLP project → up → concat → Fusion → K pixel logits.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Два способа получить multiscale context

Сегментация требует различать локально похожие пиксели по окружению. DeepLabv3+ использует atrous convolutions и ASPP, SegFormer — иерархический Transformer encoder и лёгкий decoder. Одинаковая итоговая shape $[B,K,H,W]$ не означает одинаковую вычислительную цепочку.

## DeepLabv3+: dilation, ASPP и decoder

Dilated convolution вставляет промежутки между точками kernel. Для kernel $k$ и dilation $r$ эффективный охват $k+(k-1)(r-1)$ при прежнем числе весов. ASPP параллельно применяет несколько dilation rates и image-level pooling, затем объединяет outputs. [DeepLabv3+](https://arxiv.org/abs/1802.02611) добавляет decoder с low-level features для уточнения границ. Высокоуровневую карту после ASPP увеличивают к сетке ранних features, уменьшают их каналы через $1\times1$, объединяют и уточняют.

Output stride определяет цену dense encoder: stride 8 сохраняет больше positions, чем stride 16. Dilation rates интерпретируют в координатах карты, а физический охват зависит и от stride backbone. Большие rates на маленькой карте могут преимущественно читать padding.

## SegFormer: encoder MiT и MLP decoder

[SegFormer](https://arxiv.org/abs/2105.15203) получает четыре разрешения через overlapping patch embeddings. Efficient attention уменьшает spatial resolution ключей и значений, а Mix-FFN вводит локальное spatial mixing через convolution. Отдельных абсолютных position embeddings в этой конструкции нет; это не означает отсутствия пространственной структуры.

Decoder проецирует каналы каждой стадии в общую ширину $D$, увеличивает все карты до сетки первой стадии, concatenates их и смешивает. Последняя projection даёт class logits. Encoder и decoder обучаются pixel loss; предобученный encoder можно адаптировать. Interpolation logits к размеру изображения — часть inference, а не новая разметка.

## Разобранный пример: охват и память

Kernel $3\times3$ с dilation 6 имеет эффективный размер $3+2\cdot5=13$ cells. На stride 16 расстояние крайних центров sampled positions составляет $12\cdot16=192$ pixels; это не полный receptive field всей сети, который включает ранние операции.

Для изображения $512^2$ SegFormer-карты могут иметь сетки $128^2,64^2,32^2,16^2$. После projection к 256 каналам и upsampling на $128^2$ concatenated tensor имеет $4\cdot256\cdot128^2=16777216$ чисел, 64 MiB float32 на один пример. Лёгкий по параметрам decoder всё ещё может требовать заметную память активаций.

## Диагностика

Плохие границы исследуют отдельно от semantic confusion: полезны boundary slices и ошибки маленьких объектов. Проверяют bilinear convention, resize масок только categorical-safe способом и ignore regions. Увеличение dilation не исправит неверные labels; увеличение decoder width не восстановит детали, отсутствующие во всех входных features.

## Самопроверка с ответами

**Практика:** смените output stride 16 на 8 при том же входе. Spatial area поздней карты вырастет примерно вчетверо; rates нужно согласовать с нужным охватом.

**Равен ли effective kernel полному receptive field?** Нет. Он описывает одну операцию на её входной карте.

**Почему SegFormer может работать без абсолютных positions?** Пространственная структура уже присутствует в overlapping embeddings и локальных операциях encoder.

## Иллюстрации из первоисточников

![deeplab: Figure 2 из авторской статьи](<../оформление/course/papers/deeplab.png>)

**Источник:** [deeplab, Figure 2](https://arxiv.org/abs/1802.02611), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![segformer: Figure 2 из авторской статьи](<../оформление/course/papers/segformer.png>)

**Источник:** [segformer, Figure 2](https://arxiv.org/abs/2105.15203), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [DeepLabv3+ — Chen et al., ECCV 2018](https://arxiv.org/abs/1802.02611)
- [SegFormer — Xie et al., NeurIPS 2021](https://arxiv.org/abs/2105.15203)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<04-segmentation/План.md>)
