# Video understanding: 3D CNN, SlowFast, TimeSformer и VideoMAE

**Уровень:** углубление.

**Предпосылки:** [Vision Transformer: patches, attention и обучение](<cv-vit.md>), [Углубление: SSL, MAE, DeiT и DINOv3](<cv-pretraining.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как модель использует время и почему выбор кадров входит в постановку?

## Схема вычислений

![Video understanding: 3D CNN, SlowFast, TimeSformer и VideoMAE: вычислительные этапы](<../оформление/course/expanded-cv-video.svg>)

SlowFast: Одно видео → два sampling → Slow: редкие; Fast: частые → 3D blocks + lateral fusion → Pooling → action logits. TimeSformer: T кадров → spatial patches → Temporal attention по позиции → Spatial attention по кадру → Aggregation → action logits. VideoMAE: Видео → tubelet tokens → Tube mask → visible encoder → Decoder + masked positions → Reconstruction loss → transfer.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Вход — временное наблюдение

Video classifier получает clip $[B,C,T,H,W]$ и выдаёт class logits; temporal localization дополнительно предсказывает интервалы событий. Sampling задаёт число кадров, частоту и длительность наблюдения. Два clips из 16 кадров с интервалами 1 и 4 кадра исходного видео покрывают разное время. Split делают по исходным видео, людям и сценам, а не по случайно нарезанным соседним clips.

## 3D CNN и SlowFast

3D convolution использует kernel $k_t\times k_h\times k_w$, смешивая время и пространство локально. При $k_t=1$ один такой слой не смешивает соседние временные позиции. [SlowFast](https://arxiv.org/abs/1812.03982) имеет slow branch с редкими кадрами для spatial semantics и fast branch с частыми кадрами и меньшим числом каналов для динамики. Lateral connections передают информацию между ветвями. Это две временные выборки одной сцены, которые нужно синхронизировать.

## TimeSformer и VideoMAE

[TimeSformer](https://arxiv.org/abs/2102.05095) превращает frames в patches и связывает их attention. Divided attention сначала связывает временные наблюдения одной позиции и отдельно spatial positions кадра. Full joint attention по $TN$ tokens требует квадратичного числа scores; factorization меняет вычисления и inductive bias.

[VideoMAE](https://arxiv.org/abs/2203.12602) задаёт self-supervised reconstruction скрытых video tubelets. Tube masking скрывает согласованные spatial области по времени, уменьшая простое копирование близких кадров. Encoder читает visible tokens, decoder восстанавливает masked targets. После pretraining reconstruction decoder можно заменить classification head. Низкий reconstruction loss не доказывает качество распознавания действий.

## Обучение, применение и утечки

Для clip classification CE сравнивает logits с меткой clip, но метка всего длинного видео может быть неверна для случайного короткого фрагмента. При test-time multi-clip evaluation фиксируют число clips, crop policy и способ агрегации. Future frames допустимы для offline classification, но недопустимы для причинного alarm в реальном времени. Temporal detection оценивают через IoU интервалов и matching, а не только frame accuracy.

## Разобранный пример: цена attention и длительность

Для $T=8$, $H=W=224$, patch size 16 имеем $N=14^2=196$, всего 1568 tokens без CLS. Joint attention содержит $1568^2=2458624$ scores на head. Divided attention приблизительно использует $N T^2+T N^2=12544+307328=319872$ scores. Это оценка attention matrices без projections, CLS и конкретного implementation overhead.

При 30 FPS и 16 samples с шагом 4 расстояние между первым и последним кадром $(16-1)\cdot4/30=2$ seconds. Сам clip не равен 16/30 seconds. Если действие занимает 0.1 seconds, такой sparse sampling способен полностью его пропустить.

## Самопроверка с ответами

**Практика:** переставьте порядок кадров. Для распознавания «открыть» и «закрыть» метка может измениться; это не безусловно label-preserving augmentation.

**Можно ли делить один ролик на train и test clips?** Это опасная утечка контекста; независимость определяют по задаче применения.

**Почему video FPS модели не равен частоте обработанных исходных кадров?** Один inference обрабатывает clip, возможны пересечения clips и отдельные decode/sampling затраты.

## Иллюстрации из первоисточников

![slowfast: Figure 1 из авторской статьи](<../оформление/course/papers/slowfast.png>)

**Источник:** [slowfast, Figure 1](https://arxiv.org/abs/1812.03982), PDF, с. 1. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![timesformer: Figure 1 из авторской статьи](<../оформление/course/papers/timesformer.png>)

**Источник:** [timesformer, Figure 1](https://arxiv.org/abs/2102.05095), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![videomae: Figure 1 из авторской статьи](<../оформление/course/papers/videomae.png>)

**Источник:** [videomae, Figure 1](https://arxiv.org/abs/2203.12602), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [SlowFast — Feichtenhofer et al., ICCV 2019](https://arxiv.org/abs/1812.03982)
- [TimeSformer — Bertasius et al., ICML 2021](https://arxiv.org/abs/2102.05095)
- [VideoMAE — Tong et al., NeurIPS 2022](https://arxiv.org/abs/2203.12602)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<05-video/План.md>)
