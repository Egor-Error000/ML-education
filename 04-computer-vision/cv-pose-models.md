# HRNet, ViTPose и RTMPose: архитектуры pose estimation

**Уровень:** углубление.

**Предпосылки:** [Pose: keypoints, heatmaps и группировка](<cv-pose.md>), [Vision Transformer: patches, attention и обучение](<cv-vit.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Чем различаются HRNet, ViTPose и координатная классификация RTMPose?

## Схема вычислений

![HRNet, ViTPose и RTMPose: архитектуры pose estimation: вычислительные этапы](<../оформление/course/expanded-cv-pose-models.svg>)

HRNet: Person crop → high-res stem → Параллельные low/high branches → Повторное multi-scale fusion → K heatmaps → inverse affine. ViTPose: Person crop → patches → Plain ViT → patch features → Лёгкий upsampling decoder → K heatmaps → keypoints. RTMPose: Person crop → backbone → Features → SimCC head → K×Lx и K×Ly logits → 1D decoding → keypoints.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Единый контракт, разные представления

Для top-down pose detector сначала выделяет человека, affine transform переводит crop к размеру сети. Pose model возвращает $K$ keypoints и confidence или visibility. Координаты должны вернуться через обратное преобразование в исходный кадр. Ошибка detector crop является частью общей ошибки системы; нельзя оценивать только pose на идеальных GT crops и переносить вывод на весь pipeline.

## HRNet: сохранять высокое разрешение

[HRNet](https://arxiv.org/abs/1902.09212) поддерживает несколько параллельных ветвей разных resolutions. Высокая сетка не исчезает полностью; между стадиями features многократно обмениваются через up/downsampling. Каждая ветвь учится одновременно использовать локальные детали и контекст других масштабов. Head формирует heatmap каждого сустава. Обучение часто использует Gaussian targets и MSE с mask видимых/размеченных joints. Высокая spatial точность стоит памяти активаций.

## ViTPose и RTMPose

[ViTPose](https://arxiv.org/abs/2204.12484) использует plain ViT для person crop и лёгкий decoder, возвращающий heatmaps. Patch encoder даёт контекст, decoder увеличивает resolution. Предобучение и качество crop существенно влияют на результат; сам Transformer не задаёт анатомические ограничения skeleton.

[RTMPose](https://arxiv.org/abs/2303.07399) использует эффективный backbone и SimCC-представление: для каждого keypoint предсказываются распределения по discretized x и y отдельно. Вместо $K H_hW_h$ heatmap logits получаем $K(L_x+L_y)$. Training сравнивает predicted distributions с одномерными мягкими targets. Независимые осевые distributions не выражают произвольную совместную корреляцию x/y; их удобство не доказывает точности во всех неоднозначных позах.

## Разобранный пример: размер выхода и обратная геометрия

Crop $192\times256$ pixels, heatmap stride 4: выходные сетки $48\times64$. Для 17 joints нужно $17\cdot48\cdot64=52224$ logits. При SimCC split ratio 2 получаем $L_x=384,L_y=512$, всего $17\cdot896=15232$. Это сравнение output arrays, без backbone и head operations.

Пусть crop начинается в $(100,50)$ исходного кадра и имеет размер $96\times128$, затем увеличен до $192\times256$ без rotation. Predicted point $(60,80)$ соответствует $(100+60/2,50+80/2)=(130,90)$. При rotation нельзя отдельно делить координаты на scale: требуется обратная affine matrix. Для heatmap argmax сначала возвращают точку в координаты model crop согласно принятой convention центров cells.

## Ошибки и выбор

Пропущенный человек не появится после улучшения pose head. Occluded joint и отсутствующая annotation различаются; loss mask должен соблюдать dataset visibility policy. При horizontal flip переставляют left/right joints и восстанавливают x. Выбор архитектуры учитывает число людей: стоимость top-down pose растёт с количеством crops. Проверяют OKS по масштабам и сложным окклюзиям.

## Самопроверка с ответами

**Практика:** поменяйте местами shoulders при flip только в изображении. Даже верная модель будет получать противоречивые targets; исправить нужно transform labels.

**Является ли argmax heatmap уже координатой исходного кадра?** Нет, он относится к выходной сетке и требует обратных transforms.

**Почему меньший output не гарантирует более быстрый pipeline?** Detector, backbone, crop batching и transfer могут доминировать над стоимостью head.

## Иллюстрации из первоисточников

![hrnet: Figure 1 из авторской статьи](<../оформление/course/papers/hrnet.png>)

**Источник:** [hrnet, Figure 1](https://arxiv.org/abs/1902.09212), PDF, с. 1. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![vitpose: Figure 2 из авторской статьи](<../оформление/course/papers/vitpose.png>)

**Источник:** [vitpose, Figure 2](https://arxiv.org/abs/2204.12484), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![rtmpose: Figure 2 из авторской статьи](<../оформление/course/papers/rtmpose.png>)

**Источник:** [rtmpose, Figure 2](https://arxiv.org/abs/2303.07399), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [HRNet — Sun et al., CVPR 2019](https://arxiv.org/abs/1902.09212)
- [ViTPose — Xu et al., NeurIPS 2022](https://arxiv.org/abs/2204.12484)
- [RTMPose — Jiang et al., 2023](https://arxiv.org/abs/2303.07399)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<05-video/План.md>)
