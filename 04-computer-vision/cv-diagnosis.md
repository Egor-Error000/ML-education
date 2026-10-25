# Диагностика CV: ошибки данных, модели и post-processing

**Уровень:** основа ветки.

**Предпосылки:** [Loss, дисбаланс, пороги и калибровка в CV](<cv-losses.md>), [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>)

**Используют:** [Сквозные разборы: дефекты, видеопоток и документы](<cv-cases.md>)

**Учебный вопрос:** Как локализовать причину плохого результата до замены архитектуры?

## Схема вычислений

![Диагностика CV: ошибки данных, модели и post-processing: вычислительные этапы](<../оформление/course/expanded-cv-diagnosis.svg>)

Error localization: Data / annotations → Preprocess → raw predictions → Decode → filter / NMS → Final metric + error slices. Controlled experiment: Наблюдение → hypothesis → Один intervention + baseline → Тот же split / budget / seeds → Сравнение → следующий шаг.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Разделить систему на проверяемые этапы

Наблюдаемая ошибка складывается из данных, preprocessing, представления модели, targets/loss и post-processing. Сначала воспроизводят один неудачный пример и сохраняют intermediate outputs: входной tensor, raw predictions, decoded coordinates и final output. Сравнение красивых итоговых картинок не показывает, на каком шаге потерялся объект. Полезный baseline должен проходить тот же evaluation pipeline.

## Диагностика по типу задачи

Для classification анализируют confusion, imbalance, calibration и slices по camera/background. Для detection отделяют пропуск candidates, ошибку класса, неточную рамку, дубликаты и подавление NMS. Для segmentation — semantic confusion, boundary error, маленькие instances и ignored pixels. [TIDE](https://arxiv.org/abs/2008.08115) систематизирует анализ detection errors; decomposition зависит от выбранного matching protocol.

Oracle intervention временно заменяет один компонент идеальным: например, GT boxes вместо detector crops. Разница показывает потенциальный bottleneck при таком intervention, но не гарантирует достижимый выигрыш реальной заменой. Компоненты взаимодействуют: более точные boxes способны изменить и classification, и NMS outcome.

## Ablation study

Контролируемое сравнение меняет один содержательный фактор при одинаковых data split, budget и evaluation. Для stochastic training нужны несколько seeds или анализ uncertainty по независимым evaluation units. Если одновременно поменять backbone, resolution, pretraining и augmentations, источник улучшения не установлен. Измерения на test не должны превращаться в цикл подбора hyperparameters.

Для segmentation overlays проверяют alignment маски, raw logits и threshold. Если labels сдвинуты на несколько pixels, более сильная модель может точнее обучиться этому сдвигу. Данные и geometry проверяют до длительного обучения.

## Разобранный пример: suppression и conditional recall

Есть 100 GT objects. До score filtering candidates покрывают 90, после threshold — 80, после NMS — 70. Recall стадий относительно GT: 0.90,0.80,0.70. Доля сохранённых после threshold среди покрытых candidates $80/90\approx0.88889$, после NMS $70/80=0.875$. Если улучшить только NMS до идеального сохранения этих 80, верхняя граница recall в этом intervention 0.80, не 1.

Для двух режимов по 200 независимым изображениям accuracy 0.80 и 0.82 разница составляет четыре изображения. Если результаты paired, важны именно discordant pairs, а не только две marginal accuracies. При 10 исправленных и 6 испорченных примерах улучшение +4, но выборка мала для уверенного общего вывода. Срез новых камер может ухудшиться при росте среднего.

## Артефакт диагностики

Для каждой гипотезы фиксируют наблюдение, intervention, ожидаемый эффект и результат. Порядок: проверить данные и geometry, измерить baseline, локализовать bottleneck, изменить компонент, повторить locked evaluation. Архитектурный выбор должен объяснять, какую наблюдаемую проблему он решает.

## Самопроверка с ответами

**Практика:** снизьте score threshold. Recall может вырасти вместе с FP; это не бесплатное улучшение качества. Сравнивайте при заданном operational criterion.

**Доказывает ли GT-box oracle качество нового detector?** Нет: он показывает чувствительность downstream к boxes, а не достижимость идеальной локализации.

**Почему один общий score недостаточен?** Разные виды ошибок и domain slices могут компенсировать друг друга в среднем.

## Источники и продолжение

- [TIDE — Bolya et al., ECCV 2020](https://arxiv.org/abs/2008.08115)

[Следующие углубления: cv-cases](<README.md>) · [Сценарии подраздела](<09-systems/План.md>)
