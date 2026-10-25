# Robustness, domain shift, OOD и отказ от ответа

**Уровень:** основа ветки.

**Предпосылки:** [Loss, дисбаланс, пороги и калибровка в CV](<cv-losses.md>), [Обучение и перенос: frozen features, fine-tuning и аугментации](<cv-transfer.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как обнаружить потерю качества при новой камере или неизвестных объектах?

## Схема вычислений

![Robustness, domain shift, OOD и отказ от ответа: вычислительные этапы](<../оформление/course/expanded-cv-robustness.svg>)

Shift evaluation: Independent domain slices → Baseline + controlled corruptions → Accuracy / AP / calibration → Failure analysis → adaptation. Selective prediction: Model score + OOD signal → Validation thresholds → Accept / reject decision → Coverage + selective risk.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Виды сдвига

Domain shift меняет distribution входов: камера, погода, season, texture или population. Label shift меняет class frequencies, concept shift — связь observations и targets. OOD detection пытается отличить observations вне принятой области применения. Это не то же самое, что распознавание известного редкого класса или поиск локального дефекта.

## Проверки invariance

Если допустимое преобразование сохраняет target, prediction желательно сделать устойчивым к нему. Но blur способен уничтожить мелкий дефект, crop — удалить объект, flip — изменить символ. Поэтому corruption test требует определения, какие метки остаются valid. [ImageNet-C](https://arxiv.org/abs/1903.12261) задаёт конкретный corruption benchmark; его результат не покрывает все новые domains.

Срезы строят по camera ID, lighting, object scale, occlusion и времени. Training augmentations должны моделировать допустимые variations; улучшение на synthetic noise не гарантирует перенос на реальную новую оптику. Можно обучать domain-specific normalization или адаптировать backbone, но adaptation оценивают на независимых data и не называют zero-shot.

## Confidence, OOD score и отказ

Maximum softmax probability — простой confidence baseline, но сеть может быть уверена на неизвестных objects. Energy, distance в embedding space или специализированный OOD detector дают другие scores с собственными assumptions. OOD threshold подбирают на representative validation; неизвестные classes test не должны использоваться для повторного tuning.

Selective prediction допускает отказ. Coverage — доля принятых examples, selective risk — ошибка среди принятых. Отказ снижает automation и создаёт нагрузку review. Калибровка старого domain может нарушиться в новом; высокое confidence не является сертификатом correctness.

## Разобранный пример: coverage и uncertainty

Из 1000 images принимаются 800; среди них 40 errors. Coverage $800/1000=0.8$, risk $40/800=0.05$. После повышения порога принимаются 500, из них 10 errors: coverage 0.5, risk 0.02. Ошибка среди принятых уменьшилась, но объём ручной проверки вырос с 200 до 500.

В новом camera domain 20 из 100 samples ошибочны, хотя mean confidence 0.95. Наблюдаемая accuracy 0.8 против заявленных 0.95 показывает miscalibration на этом срезе; одной mean недостаточно для полной calibration curve. Если на 100 independent negative samples нет FP, это не доказывает FPR 0. Грубая 95% верхняя оценка «rule of three» около $3/100=3\%$, при assumptions Bernoulli и независимости.

## Monitoring

Отслеживают input distributions, coverage, score distributions и delayed ground truth quality. Pixel histogram drift не равен accuracy drift; неизменный histogram тоже не гарантирует качества. Recalibration, retraining и rollback выбирают по подтверждённому типу проблемы. Временные data должны учитывать задержку появления истинных labels.

## Самопроверка с ответами

**Практика:** примените более строгий threshold только на новой камере. Это domain-specific policy, которую нужно отдельно оценить по coverage и ошибкам.

**Почему OOD detector не решает все ошибки?** Ошибки бывают и на обычных in-domain images.

**Можно ли объявить систему надёжной по отсутствию ошибок на десяти кадрах?** Нет: малая и зависимая выборка даёт слабую оценку редких сбоев.

## Источники и продолжение

- [ImageNet-C — Hendrycks, Dietterich, ICLR 2019](https://arxiv.org/abs/1903.12261)
- [Selective classification — Geifman, El-Yaniv, NeurIPS 2017](https://arxiv.org/abs/1705.08500)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<09-systems/План.md>)
