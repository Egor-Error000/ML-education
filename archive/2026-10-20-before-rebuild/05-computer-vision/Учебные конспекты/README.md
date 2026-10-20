# Computer Vision — учебные конспекты по новому чату

[Версия для чтения с набранными формулами](<../../Читать конспекты.html#course-5-0>) · работает без интернета.

[Маршрут и сценарии модуля](<../../План — компьютерное зрение.md>) · [Исходный план](<../План — компьютерное зрение.md>) · [Источники и редакционные исправления](<../notes/Интеграция CV-чата S7.md>) · [Общий маршрут](../../README.md)

Основа — добавленный вами `chat-План изучения темы(2).txt`. Его девять задач соответствуют существующему CV-плану. Перед ними добавлена общая вводная V0: формы тензоров, preprocessing, свёртка и контракт backbone/neck/head. Десять глав дают самостоятельный учебный маршрут: задача и assumptions → механизм/формулы → ручной пример → ограничения → вопросы с ответами и практика.

- [V0. Изображение, тензоры и общий путь модели](<00 — Изображение, тензоры и общий путь модели.md>) — обязательная основа для всех задач CV.
- [А. Классификация, backbone и математика ViT](<А — Классификация, backbone и математика ViT.md>) — пункт 1; подробные patch embedding и attention, полный цикл обучения ViT, CLS/positions, Swin, MAE, DeiT и ограниченный comparison case DINOv3.
- [Б. Detection, IoU, matching, AP, NMS и архитектуры](<Б — Detection, IoU, AP, NMS и архитектуры.md>) — пункт 2.
- [В. Semantic segmentation, FCN, U-Net и DeepLab](<В — Semantic segmentation, FCN, U-Net и DeepLab.md>) — пункт 3.
- [Г. Instance segmentation, RoIAlign и prototype masks](<Г — Instance segmentation, RoIAlign и prototype masks.md>) — пункт 4; добавлена граница между closed-set masks и promptable concept segmentation SAM 3.
- [Д. Pose, heatmaps, OKS и grouping](<Д — Pose, heatmaps, OKS и группировка keypoints.md>) — пункт 5.
- [Е. OBB, periodicity и rotated geometry](<Е — OBB, поворот, periodicity и rotated IoU.md>) — пункт 6.
- [Ж. Tracking, Kalman, association и trajectory metrics](<Ж — Tracking, Kalman, association, MOTA, IDF1 и HOTA.md>) — пункт 7; SAM 3.1 приведён как hardware-specific throughput case.
- [З. Depth, scale ambiguity, DPT и regression metrics](<З — Depth, scale ambiguity, DPT и метрики.md>) — пункт 8.
- [И. CLIP, open-vocabulary detection и YOLO-World](<И — CLIP, open-vocabulary detection и YOLO-World.md>) — пункт 9; added boundary between text-conditioned boxes, concept masks and claims about prompt understanding.

## Источник и углубления

**S7:** [архивная копия чата](<../../archive/study-sources/chat-План изучения темы(2).txt>) · [исходный файл в общей папке](<../../chat-План изучения темы(2).txt>). 7816 строк; SHA-256 зарегистрирован в общем манифесте. В начале экспорт содержит сам план; его отдельная историческая копия сохранена в [архиве](<../../archive/2026-10-17/План CV из нового чата.md>).

В источнике нет надёжных USER/ASSISTANT разделителей: присутствуют английские служебные вставки и повторные испорченные формулы. В конспекты перенесены содержательные объяснения и ваши вопросы, а формулы восстановлены в согласованной нотации. Диапазоны происхождения указаны в главах и [реестре S7](<../notes/Интеграция CV-чата S7.md>).

Дополнены ручные расчёты метрик, examples inference/annotation errors, gradients patch projection, losses и diagnostic protocols. Основной текст не требует читать чат параллельно. Историческое «все пункты пройдены» означает охват обсуждения, а не автоматически подтверждённую практику или самостоятельное понимание.

## Как проходить

Основное ядро: V0 → А → Б → В → Г. Затем Д/Е/Ж/З/И по задаче; они не требуют искусственно проходить все предыдущие прикладные ветки. Для tracking нужна detection, для OVD — detection и image-text alignment. До практики проверить [план данных](<../../01-data/План — данные и качество ML-эксперимента.md>) и корректность split/аннотаций.

Контрольные расчёты: patch→Conv2d correspondence; TP/FP и AP; foreground IoU/Dice; bilinear RoIAlign; OKS; rotation/cyclic angle; assignment/MOTA/HOTA; raw/aligned depth; contrastive text-image scores. Выполнить расчёт до просмотра ответа в главе.
