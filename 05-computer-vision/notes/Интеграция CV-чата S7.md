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

## Прогресс пользователя

Новый источник показывает подробный охват девяти CV-задач и углублённые вопросы по backbone/ViT. Он не содержит проверяемых пользовательских training artifacts или результатов запуска. Статусы: обсуждение — найдено; самостоятельное понимание — требует проверки; практика — не подтверждена этим экспортом. Автоматический «100% освоено» не присваивается.

Для продолжения использовать вопросы и небольшие расчёты в главах. Большой проект не обязателен для следующего теоретического занятия.
