# Сквозные разборы: дефекты, видеопоток и документы

**Уровень:** основа ветки.

**Предпосылки:** [Anomaly detection: reconstruction, PaDiM и PatchCore](<cv-anomaly.md>), [SORT, DeepSORT, ByteTrack и BoT-SORT: разбор последовательности](<cv-tracking-models.md>), [Document vision: layout, reading order и таблицы](<cv-documents.md>), [Диагностика CV: ошибки данных, модели и post-processing](<cv-diagnosis.md>), [Deployment: экспорт, quantization, distillation и производительность](<cv-deployment.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как выбрать и проверить весь CV pipeline под конкретную цель?

## Схема вычислений

![Сквозные разборы: дефекты, видеопоток и документы: вычислительные этапы](<../оформление/course/expanded-cv-cases.svg>)

Контроль дефектов: Independent product + labels → Classifier / mask / anomaly → Threshold + review capacity → Miss cost + false alarms. Видео и документы: Frames / page regions → Tracking / OCR + structure → Event logic / field assignment → End-to-end errors + evidence.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Кейс 1: контроль дефектов

Цель — снизить пропуски брака при допустимом числе ручных проверок на линии. Единица оценки — независимое изделие, а не каждый почти одинаковый frame. Если достаточно reject/accept, начинают с classifier или frozen embeddings; если нужны coordinates — segmentation/detection; при отсутствии examples дефектов — normal-only anomaly baseline. Разметка должна различать допустимые вариации, критический брак и неопределённые случаи.

Выбирают split по партиям, камерам и времени. Сначала проверяют resolution: минимальный дефект должен сохранять измеримый visual signal. Затем сравнивают model choices при одном validation protocol. Порог подбирают по стоимости и review capacity, а final test не используют для цикла настройки. Дальнейшее добавление labels определяется типом ошибок: новые normal textures или редкие defect modes.

## Кейс 2: анализ видеопотока

Задача — считать уникальные проходы через линию, а не число найденных boxes. Pipeline: detector → tracker → state machine события. Track должен иметь историю стороны линии и направление crossing; jitter рядом с границей требует hysteresis или временного подтверждения. Нужны timestamp, обработка frame drops и политика lost tracks. Более высокий frame AP не гарантирует точного подсчёта, если один человек несколько раз меняет ID.

Train/test разделяют по videos и cameras, ground truth размечает именно events. Оценивают missed events, duplicate counts, direction errors и latency alarm. SORT/ByteTrack служат baselines; appearance и camera-motion compensation добавляют после диагностики конкретных ID failures. Monitoring включает image conditions и изменение operational rates.

## Кейс 3: обработка документов

Цель — извлечь поля счёта с координатами evidence. Для digital PDF сначала проверяют доступность корректного text layer, для scans — rasterization и OCR. Layout определяет regions и order, recognizer читает, extraction связывает ключи и values. Tables требуют structure recognition; итоговые monetary fields проходят type и arithmetic checks. Правдоподобное значение без source region не принимается за проверенное.

Split по организациям и templates показывает перенос на новые формы. Оценивают field exact match и долю документов, полностью прошедших без review. Редкий критический field нельзя спрятать в среднем CER. Отказ должен быть понятной частью pipeline: показать source crop и неподтверждённое поле человеку.

## Разобранный пример: operational cost

На линии за день 10000 изделий: 100 дефектных, 9900 нормальных. Candidate находит 90 дефектных и ошибочно отклоняет 198 нормальных. Recall 0.9, FPR $198/9900=0.02$, precision $90/(90+198)=0.3125$. Review получает 288 изделий. При цене FN 100 и FP 1 ожидаемая измеренная стоимость $10\cdot100+198=1198$ условных единиц.

Второй threshold даёт TP=95 и FP=396. Стоимость $5\cdot100+396=896$, но review 491. При capacity 300 второй threshold непригоден без изменения процесса, хотя стоимость ниже. Решение требует одновременно quality, capacity и latency, а не одного F1. Это учебные counts, не эксперимент курса.

## Проверяемый результат

Для каждого кейса сохраняют постановку, data contract, baseline, независимый split, error taxonomy, выбранный intervention и acceptance criterion. Deployment bundle включает transforms и thresholds. Изменение камеры, line geometry или template запускает повторную evaluation; нельзя считать его косметической правкой данных.

## Самопроверка с ответами

**Практика:** увеличьте capacity review до 500. Второй threshold становится допустимым по этому ограничению, но остальные требования ещё нужно проверить.

**Почему counting нельзя оценить только AP?** Он зависит от identity continuity и логики события после detection.

**Что проверять первым при неверной сумме документа?** Source crop, OCR и assignment поля; генерация правдоподобной суммы не исправляет provenance.

## Источники и продолжение

- [ByteTrack — Zhang et al., ECCV 2022](https://arxiv.org/abs/2110.06864)
- [PatchCore — Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265)
- [PubTables-1M — Smock et al., CVPR 2022](https://arxiv.org/abs/2110.00061)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<09-systems/План.md>)
