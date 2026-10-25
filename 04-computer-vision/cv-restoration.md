# Восстановление изображений: denoising, deblurring и super-resolution

**Уровень:** основа ветки.

**Предпосылки:** [Изображение, карты признаков и блоки модели](<cv-input.md>), [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как восстанавливать изображение, не принимая правдоподобные детали за измеренные?

## Схема вычислений

![Восстановление изображений: denoising, deblurring и super-resolution: вычислительные этапы](<../оформление/course/expanded-cv-restoration.svg>)

Super-resolution: Low-resolution observation → Feature extraction / residual blocks → Subpixel features r²C → Pixel shuffle → high-res image. Restoration training: Clean image x → Degradation D + noise → y → Restorer → estimate x → Pixel / feature loss → gradient.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Обратная задача

Пусть чистое изображение $x$ проходит degradation $y=\mathcal D(x)+n$. Denoising уменьшает noise, deblurring обращает blur, super-resolution восстанавливает high-resolution representation из low-resolution observation. Обратная задача часто неоднозначна: разным $x$ соответствует одно $y$. Learned model добавляет prior, а не получает потерянные measurements из ничего.

## Архитектуры и targets

Baseline super-resolution — bicubic upsampling. [SRCNN](https://arxiv.org/abs/1501.00092) обучает CNN отображать увеличенный low-resolution image в high-resolution target. Более поздние residual/U-Net структуры могут предсказывать residual или noise. Pixel shuffle перераспределяет channels в spatial positions: $[B,r^2C,H,W]\to[B,C,rH,rW]$; число элементов не меняется, но полезные subpixel features должны быть обучены.

Paired supervised training требует aligned clean/degraded images. Synthetic bicubic degradation не представляет автоматически camera blur, compression и sensor noise реальных данных. Для denoising нормальный уровень noise задают в согласованном диапазоне pixels. Data leakage возникает, если разные patches одного снимка или одной сцены попадают в train и test.

## Loss и конфликт качества

L2 pixel loss оптимизирует conditional mean, L1 связан с conditional median; при неоднозначных fine details среднее может быть размытым. Perceptual loss сравнивает features, adversarial loss поощряет похожесть на распределение реалистичных images. Эти цели могут улучшать субъективный вид и одновременно снижать точность конкретной мелкой детали. Для scientific measurements и чтения серийных номеров hallucinated texture недопустимо считать ground truth.

## Разобранный пример: PSNR и неоднозначность

Для pixels в $[0,1]$ $PSNR=10\log_{10}(1/MSE)$. При MSE 0.01 получаем 20 dB; при 0.0025 — $10\log_{10}400\approx26.02060$ dB. Уменьшение MSE вчетверо добавило около 6.02 dB. Для диапазона 0..255 numerator должен быть $255^2$, а не 1; иначе comparison бессмысленно.

Два допустимых high-resolution patches равны $(0,1)$ и $(1,0)$ с одинаковой вероятностью, а degradation усредняет их до $y=0.5$. L2-optimal prediction $(0.5,0.5)$ имеет error 0.5 по сумме squared differences для любого варианта. Выбор $(0,1)$ иногда точен, иногда даёт error 2, ожидаемое значение 1. Усреднение улучшает L2, но теряет резкий рисунок — это следствие ambiguity.

Для scale factor 2 и RGB pixel shuffle требует 12 channels на low-resolution grid. Например, $[1,12,32,32]\to[1,3,64,64]$; операция сама не увеличивает количество предсказанных scalar values.

## Диагностика

Проверяют реальные degradation slices, ringing, boundary effects и downstream accuracy. PSNR по RGB и luminance, с border crop и без него — разные протоколы. Заявление «восстановлен текст» требует сравнения с исходной записью, а не оценки визуальной резкости.

## Самопроверка с ответами

**Практика:** обучите мысленно модель только на Gaussian noise, примените к motion blur. Это смена degradation model, а не просто более сильный noise.

**Почему высокий PSNR не гарантирует natural texture?** Pixel objective может предпочитать усреднённые структуры.

**Можно ли использовать super-resolution как достоверное доказательство скрытого символа?** Нет: prior способен дорисовать правдоподобный, но неверный знак.

## Иллюстрации из первоисточников

![srcnn: Figure 2 из авторской статьи](<../оформление/course/papers/srcnn.png>)

**Источник:** [srcnn, Figure 2](https://arxiv.org/abs/1501.00092), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [SRCNN — Dong et al., TPAMI 2016](https://arxiv.org/abs/1501.00092)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<07-specialized/План.md>)
