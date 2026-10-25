# Deployment: экспорт, quantization, distillation и производительность

**Уровень:** основа ветки.

**Предпосылки:** [Обучение и перенос: frozen features, fine-tuning и аугментации](<cv-transfer.md>), [Loss, дисбаланс, пороги и калибровка в CV](<cv-losses.md>)

**Используют:** [Сквозные разборы: дефекты, видеопоток и документы](<cv-cases.md>)

**Учебный вопрос:** Как сохранить смысл предсказаний после экспорта и измерить полную стоимость?

## Схема вычислений

![Deployment: экспорт, quantization, distillation и производительность: вычислительные этапы](<../оформление/course/expanded-cv-deployment.svg>)

Deployment bundle: Weights + input / output contract → Export → runtime graph → Numerical comparison + task eval → Versioned bundle → rollout. PTQ / performance: Representative calibration inputs → Activation ranges → scales → Quantized operators / backend → Full latency + quality validation.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Артефакт модели шире файла весов

Inference bundle содержит architecture, weights, preprocessing, class vocabulary, coordinate conventions, output decoding, thresholds и runtime version. Экспорт без правильного RGB/BGR или letterbox inversion может дать неверные ответы при исправном graph. Reference evaluation сохраняет несколько representative inputs и сравнивает raw outputs и final task metrics между training framework и target runtime.

## Precision и quantization

Float16 уменьшает размер многих tensors, но не все операции одинаково устойчивы к меньшему range и precision. PTQ, post-training quantization, подбирает scales по calibration data без обычного полного retraining. QAT моделирует quantization effects при training. Для affine quantizer $q=\operatorname{clip}(\operatorname{round}(x/s)+z)$, reconstruction $\hat x=s(q-z)$; $s>0$ — scale, $z$ — zero point. Per-channel и per-tensor schemes дают разные ошибки.

Calibration должна включать representative resolution, lighting и activation ranges. Outliers могут растянуть range и ухудшить точность большинства значений. [ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) описывает поддержку quantization в конкретном runtime; наличие INT8 weights не доказывает, что каждый operator исполняется INT8 на выбранном device.

## Distillation и runtime

Student может обучаться по teacher logits, features или predictions вместе с истинными labels. Это изменение objective и модели; teacher errors также могут переноситься. Структурное уменьшение channels способно ускорять dense kernels, а неструктурные нули в weights без sparse backend часто не дают ускорения.

Batching увеличивает throughput, но ожидание batch добавляет latency. Dynamic shapes удобны для разных resolutions, однако требуют проверки kernels и memory profiles. Измеряют decoding, preprocessing, transfer, model, post-processing и queue. GPU timing требует synchronization или событий device, иначе измеряется только постановка работы в очередь.

## Разобранный пример: quantization и latency

Пусть $s=0.1,z=0$, допустимый signed range -128..127. Значение 0.26 округляется в $q=3$, reconstruction 0.3, error 0.04. Значение 20 clips до 127, reconstruction 12.7, error -7.3. Малый typical rounding error не спасает от clipping outliers.

Один запрос тратит 5 ms на decode, 3 ms preprocessing, 2 ms transfer, 12 ms model, 4 ms post-processing. Sequential latency 26 ms, максимум такого неперекрытого pipeline около $1000/26=38.46$ requests/s. Ускорение model вдвое даёт 20 ms и 50 requests/s, не удвоение всего pipeline. Параллельная pipeline-система может иметь иной throughput; p95 latency всё равно измеряют отдельно.

Для 10 million weights float32 storage примерно 40 MB, INT8 — 10 MB без scales и metadata. Peak memory также включает activations, buffers и runtime caches.

## Самопроверка с ответами

**Практика:** измерьте только model forward и подпишите результат «end-to-end». Это неверный протокол: потеряны 14 ms остальных стадий примера.

**Почему quantized graph нужно повторно оценить по AP или Dice?** Малая numerical error может изменить ordering, threshold и NMS, влияя на итоговые объекты.

**Нужен ли rollback?** Да: candidate сравнивают с сохранённым baseline bundle и возвращают прежнюю версию при подтверждённой регрессии.

## Источники и продолжение

- [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

[Следующие углубления: cv-cases](<README.md>) · [Сценарии подраздела](<09-systems/План.md>)
