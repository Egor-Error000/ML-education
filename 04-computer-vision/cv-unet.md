# U-Net и nnU-Net: пространственный decoder и настройка pipeline

**Уровень:** основа ветки.

**Предпосылки:** [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>)

**Используют:** [Медицинские и спутниковые изображения: данные, объём и масштаб](<cv-domains.md>)

**Учебный вопрос:** Как U-Net восстанавливает детали и что автоматизирует nnU-Net?

## Схема вычислений

![U-Net и nnU-Net: пространственный decoder и настройка pipeline: вычислительные этапы](<../оформление/course/expanded-cv-unet.svg>)

U-Net: Image → encoder levels → Bottleneck → upsampling → Concat с encoder skip → Convolutions → pixel logits. nnU-Net pipeline: Данные + spacing + labels → Планирование patch / model → 2D / 3D training + validation → Sliding-window → original grid.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## От класса изображения к маске

В segmentation выход сохраняет spatial coordinates: для каждого пикселя нужны logits классов. Encoder снижает resolution и увеличивает контекст, decoder возвращает сетку. Одно upsampling не восстанавливает утраченную информацию. U-Net передаёт decoder также признаки ранних уровней через skip concatenation. Это соединение отличается от residual addition: число каналов после concatenation увеличивается.

## U-Net: две ветви одной сети

Encoder чередует convolution blocks и pooling; decoder — upsampling или transposed convolution, объединение с соответствующей encoder map и обработку convolution blocks. Head $1\times1$ переводит каналы в $K$ class logits. Для binary mask достаточно одного logit и BCE; multiclass с взаимоисключающими метками обычно использует $K$ logits и CE. Dice-компонент учитывает перекрытие целой области.

В [исходном U-Net](https://arxiv.org/abs/1505.04597) valid convolutions уменьшают карты, поэтому encoder features обрезаются перед соединением, а выход меньше входа. Часто используемый same-padding вариант — отдельная конфигурация. Transposed convolution является обучаемым преобразованием, но не точной математической обратной операцией pooling. Неудачное покрытие kernel/stride может давать checkerboard artifacts.

## nnU-Net — больше, чем название блока

[nnU-Net](https://arxiv.org/abs/1809.10486) согласует preprocessing, spacing, patch size, network configuration, обучение и inference с характеристиками датасета. Для volumes важны физические размеры voxel и доступная память. 2D, 3D и cascade-конфигурации решают разные компромиссы. Автоматическая настройка pipeline не делает медицинские метки безошибочными и не снимает необходимость split по пациентам.

При inference большого изображения используют overlapping patches, объединяя logits или probabilities по заранее выбранному правилу. Loss не должен учитывать padding или ignore voxels как обычный background. Переход к полному volume требует вернуть исходные spacing и ориентацию.

## Разобранный пример: каналы и ошибка

В учебном same-padding decoder карта $[B,128,32,32]$ увеличивается до $[B,64,64,64]$. Skip имеет $[B,64,64,64]$. Concatenation даёт $[B,128,64,64]$, затем свёртка $3\times3$, $128\to64$ содержит $9\cdot128\cdot64=73728$ весов без bias. Сложение вместо concatenation изменило бы вход следующей свёртки и всю архитектуру.

Для мягкого binary prediction $(0.8,0.3,0.1,0.9)$ и target $(1,0,0,1)$ пересечение $\sum py=1.7$, сумма predictions 2.1, сумма targets 2. Soft Dice без smoothing $2\cdot1.7/(2.1+2)=0.82927$, loss 0.17073. Hard mask при пороге 0.5 совпадёт с GT и имеет Dice 1. Значит, soft training loss и thresholded metric не обязаны совпадать.

## Нюансы выбора

При малом foreground доля пустых patches существенно меняет обучение. Контролируют sampling, расстояние до границ и false positives на пустых снимках. Большой Dice крупного органа может скрывать пропуски маленьких lesions. Дополнительно оценивают объекты и физическую ошибку границ.

## Самопроверка с ответами

**Практика:** увеличьте только spacing по глубине. Тот же voxel kernel теперь охватывает другой физический объём; изотропный pooling может стать неуместным.

**Обязательно ли U-Net имеет выход размера входа?** Нет: это зависит от padding и crops конкретной реализации.

**Почему skip не гарантирует правильный контур?** Она передаёт features, но head должна научиться отделять полезные детали от шума и фона.

## Иллюстрации из первоисточников

![unet: Figure 1 из авторской статьи](<../оформление/course/papers/unet.png>)

**Источник:** [unet, Figure 1](https://arxiv.org/abs/1505.04597), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [U-Net — Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597)
- [nnU-Net — Isensee et al., 2018](https://arxiv.org/abs/1809.10486)
- [nnU-Net official repository](https://github.com/MIC-DKFZ/nnUNet)

[Следующие углубления: cv-domains](<README.md>) · [Сценарии подраздела](<04-segmentation/План.md>)
