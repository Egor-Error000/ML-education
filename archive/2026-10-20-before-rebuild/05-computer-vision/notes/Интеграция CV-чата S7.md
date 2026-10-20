# S7: новый CV-чат и его место в маршруте

Добавлен пользователем 17 октября 2026. Исходник: [chat-План изучения темы(2).txt](<../../chat-План изучения темы(2).txt>); [архивная копия](<../../archive/study-sources/chat-План изучения темы(2).txt>); [учебные главы](../Учебные%20конспекты/README.md).

## Сохранение и соответствие плану

268157 bytes, 7816 строк. SHA-256: `0375527447b080efba74d67131e47136e468a4d588ec88f117ef0fc87afccf6a`. Оригинал сохранён на месте, архивная копия совпадает по байтам. Запись S7 добавлена в существующий [манифест источников](../../06-llm/migration/source_manifest.json); прежние шесть checksum не изменены.

План в начале нового чата совпадает по девяти задачам с действующим CV-разделом. [Снимок исходного плана](<../../archive/2026-10-17/План CV из нового чата.md>) сохраняет его формулировки; рабочий [план CV](<../План — компьютерное зрение.md>) связывает главы и уточняет спорные утверждения. Архивный план — исторический источник, а не второй обязательный маршрут.

Соответствие диапазонов оригинального TXT:

- 1–205: вставленный план.
- 207–1617: classification, backbone и ViT; пользовательские вопросы на 823 и1164; подробная patch-projection математика начинается около1310. Глава А.
- 1618–2468: detection, IoU/AP/NMS, архитектуры. Глава Б.
- 2469–3080: semantic segmentation. Глава В.
- 3081–3712: instance segmentation. Глава Г.
- 3713–4523: pose/keypoints. Глава Д.
- 4524–5276: OBB и angle. Глава Е.
- 5277–6186: MOT. Глава Ж.
- 6187–6969: depth. Глава З.
- 6970–7816: OVD и итоговая сводка. Глава И.

Экспорт не имеет надёжной разметки ролей. Английские вставки о подготовке ответа исключены из учебного материала; они не считаются отдельным обсуждением пользователя. Дубли разрушенных математических строк восстановлены вручную, а не скопированы как есть.

## Существенные исправления

1. Patch embedding при перестановке patches **equivariant**: outputs переставляются; не invariant. Согласованы NCHW и flatten `(c,p,q)`; `Linear.weight` соответствует транспонированной E.
2. ResNet облегчает optimization и создаёт direct gradient path, но не даёт абсолютной гарантии против vanishing/exploding в любой глубине.
3. YOLOv1 отличается от anchor-based схем v2/v3. Anchor-free, decoupled head и NMS-free — независимые design choices; нельзя выводить одно из другого.
4. AP в ручном примере имеет объявленную interpolation convention; exact COCO evaluation требует официального protocol, crowd/ignore/maxDets и thresholds.
5. Semantic output и instance mask — разные задачи. Auxiliary semantic loss не равен самостоятельной semantic inference head. Task support проверена по официальной документации, а не по имени family.
6. RoIAlign извлекает spatial tensor с fractional sampling. Head separation не гарантирует отсутствия conflicts shared-backbone gradients.
7. COCO `v>0` включает размеченные закрытые keypoints. Не смешиваются stored OKS sigmas и k в альтернативной записи; PCK/MPJPE alignment определяется dataset.
8. YOLO-pose не описывается как обязательный двухстадийный detector+отдельная crop pose-model.
9. OBB имеет periodπ без swap и эквивалентность swap w/h сπ/2. Rotated IoU piecewise geometry; фраза о недифференцируемости везде убрана. Quadrilateral→rectangle может терять shape.
10. HOTA содержит sqrt на заданном threshold и последующее aggregation. IDF1 тоже зависит от missed/false detections; MOTA явно включает IDS.
11. Stereo disparity=fB/Z, а не безразмерная1/Z. Metric/relative outputs и raw/aligned eval разделены. δ_k uses1.25^k; log-scale invariance не равна affine scale+shift в произвольном domain.
12. CLIP objective восстановлен в двух направлениях. OVD не гарантирует любые concepts/languages; scope «novel» связан с supervision protocol. Название «YOLO26-OV» не превращается в придуманный model ID: проверяются released YOLO-World/YOLOE variants.

## Переработка глубины второй редакции

После замечания пользователя все девять учебных глав расширены по исходному чату. Подробные объяснения встроены в соответствующие разделы. Полнота второй редакции должна проверяться по содержанию, а не только по числу глав, формул или слов.

Дополнительные исправления, обнаруженные при подробном переносе:

13. **Detection, 1618–2468:** восстановлен исходный пример пяти GT/восьми predictions с AP=97/120 при объявленном continuous interpolation; multi-scale YOLOv2 меняет resolution каждые десять batches, а не эпох. Original YOLOv3 не описывается CIoU-loss поздних модификаций.
14. **Semantic, 2469–3080:** формула receptive field без множителей strides ограничена unit-stride стеком; добавлен общий recurrence. Pixel loss не означает независимой от контекста pixel-модели. Разобраны empty-target soft Dice, взвешивание границ U-Net и postprocessing.
15. **Instance, 3081–3712:** исходной mask `[0,0,1,1,1,0,0,1,1]` соответствуют alternating counts `[2,3,2,2]`; JSON из чата кодировал другую mask. COCO column-major RLE не равен произвольным `(start,length)` tuples. YOLACT включает detection; Fast NMS и Matrix NMS не один метод. SOLO не объявляется безусловно NMS-free.
16. **Pose, 3713–4523:** число размеченных joints не равно только v=2; skeleton edges и body/hand/face наборы различаются. Original Mask R-CNN keypoint loss использует spatial softmax CE, а не универсальную Gaussian MSE. Добавлены normalizations heatmap/SimCC, PAF grouping и continuity constant Wing.
17. **OBB, 4524–5276:** исправлена площадь axis-aligned envelope: произведение `(|w cosθ|+|h sinθ|)` и `(|w sinθ|+|h cosθ|)`. Для square на45° factor2, для10×2 rectangle —3.6. Исходник смешивал Wasserstein с ProbIoU и задавал similarity, растущую при удалении; приведено корректное различие distance/similarity через Gaussian comparison. Angle units, ordering и anisotropic resize разобраны отдельно.
18. **Tracking, 5277–6186:** DeepSORT cascade prioritizes recency, а не просто detector confidence. Online не равнозначно real-time. Low-score detections должны дойти до ByteTrack association; внешний score filter может удалить их заранее. Добавлены числовой trajectory split для IDF1 и пример неверного усреднения HOTA.
19. **Depth, 6187–6969:** optical-axis Z и Euclidean range различны; SqRel по записанной формуле имеет единицы depth. Median/affine alignment использует GT и не выдаётся за самостоятельное восстановление metric scale. Уточнены berHu, inverse-depth domain, reprojection assumptions и влияние intrinsics.
20. **OVD, 6970–7816:** сохранены дополнительные архитектурные направления и prompting; Detic не расшифровывается как «Detect Everything with Text». Negative prompt не является универсальным оператором NOT. Приведён symmetric CLIP расчёт2×2, разделены candidate-softmax и independent scores, разобрано entity binding для compositional queries.

Примеры — учебные вычисления, не результаты запусков пользователя. Исходный TXT и архивная копия не изменялись.

## Статусы обучения

Новый источник показывает подробный охват девяти CV-задач и углублённые вопросы по backbone/ViT. Он не содержит проверяемых пользовательских training artifacts или результатов запуска. Статусы: обсуждение — найдено; самостоятельное понимание — требует проверки; практика — не подтверждена этим экспортом. Автоматический «100% освоено» не присваивается.

Для продолжения использовать вопросы и небольшие расчёты в главах. Большой проект не обязателен для следующего теоретического занятия.
