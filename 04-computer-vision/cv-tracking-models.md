# SORT, DeepSORT, ByteTrack и BoT-SORT: разбор последовательности

**Уровень:** углубление.

**Предпосылки:** [Tracking: Kalman, association и идентичность](<cv-tracking.md>)

**Используют:** [Сквозные разборы: дефекты, видеопоток и документы](<cv-cases.md>)

**Учебный вопрос:** Как сохранить ID при падении confidence и пересечении траекторий?

## Схема вычислений

![SORT, DeepSORT, ByteTrack и BoT-SORT: разбор последовательности: вычислительные этапы](<../оформление/course/expanded-cv-tracking-models.svg>)

SORT / DeepSORT: Detector → boxes / crops → Kalman predict + motion gate → IoU; appearance + cascade → Update ID / lifecycle. ByteTrack / BoT-SORT: Tracks → motion prediction → High-score association → Low-score recovery; camera motion → Matched update / lost / new.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Общий detector, разные правила association

Tracking-by-detection связывает найденные в каждом кадре boxes с существующими tracks. Track содержит ID, motion state, uncertainty, age и lifecycle state. SORT использует Kalman prediction и IoU-based assignment. DeepSORT добавляет appearance embeddings и motion gating; это помогает различать людей с похожим положением, но зависит от качества ReID. Детектор и его preprocessing фиксируют, чтобы сравнивать именно association.

## DeepSORT, ByteTrack и BoT-SORT

[DeepSORT](https://arxiv.org/abs/1703.07402) использует matching cascade, уделяя приоритет недавно наблюдавшимся tracks, а затем выполняет дополнительное IoU association для определённых оставшихся candidates. Appearance distance и motion gate решают разные задачи: похожая одежда не должна разрешать физически невозможный скачок.

[ByteTrack](https://arxiv.org/abs/2110.06864) сначала связывает tracks с high-score detections, затем пробует сопоставить оставшиеся подходящие tracks с low-score detections. Слабая detection может быть настоящим частично закрытым объектом. Это не означает, что любой low-score box должен создавать новый track: initialization и thresholds отдельны. [BoT-SORT](https://arxiv.org/abs/2206.14651) сочетает motion, appearance и компенсацию движения камеры; camera transform нельзя путать с движением самого объекта.

## Разобранный пример: три кадра

Рассмотрим ByteTrack-подобную учебную схему с high threshold 0.6, low threshold 0.1 и IoU gate 0.3. В кадре 1 объект A имеет box $(10,10,30,30)$, score 0.9; создаётся ID 7. Кадр 2: Kalman предсказывает $(12,10,32,30)$, detection $(13,10,33,30)$ имеет score 0.3. Она не участвует в high-score стадии, но её IoU с prediction $380/420\approx0.90476$; low-score association сохраняет ID 7. При отбрасывании всех scores ниже 0.6 track остался бы без observation.

Кадр 3: prediction $(15,10,35,30)$, detection A $(16,10,36,30)$ со score 0.85 и detection B $(60,10,80,30)$ со score 0.9. A имеет высокий IoU и обновляет ID 7; B не проходит motion/overlap для этого track и может создать новый ID. Confidence B выше, но score сам по себе не определяет identity.

Если после долгой окклюзии A и B встречаются близко, motion может стать неоднозначным. Appearance помогает, только если embeddings различают объекты; одинаковая форма, uniform и плохой crop создают риск ID switch. После истечения max age удалённый track не обязан получать прежний ID.

## Состояние и оценка

Отдельно фиксируют tentative/confirmed/lost/removed transitions. Число пропущенных кадров следует переводить во время: max age 30 означает секунду при 30 FPS и три секунды при 10 FPS. При frame drops используют реальное $\Delta t$, иначе motion prediction становится систематически неверной. HOTA, IDF1, MOTA и object recall показывают разные виды ошибок; результат на GT detections не равен качеству полного pipeline.

## Самопроверка с ответами

**Практика:** понизьте score кадра 2 до 0.05. В выбранной схеме detection исключается и observation не обновляет track; возможное сохранение ID зависит от lifecycle и последующей association.

**Почему нельзя выбирать detection только по confidence?** Confidence оценивает обнаружение, а identity требует согласования с состоянием track.

**Исправляет ли camera compensation любую окклюзию?** Нет. Она компенсирует общую геометрию кадра, но не возвращает скрытую информацию о внешнем виде объекта.

## Иллюстрации из первоисточников

![bytetrack: Figure 2 из авторской статьи](<../оформление/course/papers/bytetrack.png>)

**Источник:** [bytetrack, Figure 2](https://arxiv.org/abs/2110.06864), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [DeepSORT — Wojke et al., ICIP 2017](https://arxiv.org/abs/1703.07402)
- [ByteTrack — Zhang et al., ECCV 2022](https://arxiv.org/abs/2110.06864)
- [BoT-SORT — Aharon et al., 2022](https://arxiv.org/abs/2206.14651)

[Следующие углубления: cv-cases](<README.md>) · [Сценарии подраздела](<05-video/План.md>)
