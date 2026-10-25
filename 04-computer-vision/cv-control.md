# ControlNet, inpainting, image-to-image и LoRA

**Уровень:** углубление.

**Предпосылки:** [Latent diffusion: autoencoder, denoiser и conditioning](<cv-latent-diffusion.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как направлять генерацию геометрией, исходным изображением и небольшими обновлениями весов?

## Схема вычислений

![ControlNet, inpainting, image-to-image и LoRA: вычислительные этапы](<../оформление/course/expanded-cv-control.svg>)

ControlNet: Noisy latent + text condition → Locked model + trainable copy → Spatial control → zero convs → Residual injection → denoising. LoRA / editing: Base weights W, frozen → Train A,B → low-rank ΔW → Image + noise / edit mask → Conditional sampling → edited image.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Разные виды управления

Text prompt задаёт неполное semantic condition. Spatial controls — edges, depth или pose — добавляют расположение структуры. Image-to-image задаёт исходное observation, inpainting разделяет known и изменяемые regions. LoRA меняет модельные weights. Эти interventions не взаимозаменяемы: LoRA не является отдельной edge map, а mask не обучает новый style сама по себе.

## ControlNet

[ControlNet](https://arxiv.org/abs/2302.05543) добавляет trainable control branch к pretrained diffusion model. Locked branch сохраняет базовые weights, trainable copy принимает дополнительное spatial condition. Zero-initialized convolutions соединяют control outputs с backbone features; начальный contribution равен нулю, затем обучается. Gradient flow не исчезает навсегда: weights zero convolutions получают gradient, после их изменения сигнал распространяется дальше.

Training использует paired images, prompts и control representations с denoising objective. На inference control map должна иметь согласованную геометрию и scale. Depth map relative model и metric depth sensor не являются одним и тем же distribution. Сильный control может конфликтовать с text и снижать variety.

## Image-to-image и inpainting

Image-to-image кодирует исходное image в latent, добавляет noise выбранного уровня и выполняет reverse sampling. Больший noise обычно оставляет больше свободы, но semantic similarity не является простой линейной функцией strength. Конкретное соответствие strength и timestep зависит от scheduler.

Inpainting использует mask с явно заданным смыслом 1: здесь считаем 1 изменяемой region. Модель может получать mask и masked image как дополнительные inputs; другой подход — согласованно восстанавливать known region на каждом шаге. Простое окончательное наложение исходных pixels сохраняет known area, но не гарантирует seamless boundary.

## LoRA

Для linear projection $W\in\mathbb R^{d_o\times d_i}$ LoRA обучает $\Delta W=(\alpha/r)BA$, $A\in\mathbb R^{r\times d_i}$, $B\in\mathbb R^{d_o\times r}$. Базовая $W$ может быть frozen. Parameters меньше, но activations denoiser и optimizer по адаптеру всё ещё стоят памяти. Для convolution layers способы adaptation различаются. LoRA файла недостаточно без совместимого base checkpoint и правильных target layers.

## Разобранный пример: параметры и mask

Для $W$ размера $1024\times1024$ имеется 1048576 weights. Rank 8 даёт $8\cdot1024+1024\cdot8=16384$ adapter weights, 1.5625% исходного числа. Если $\alpha=8$, coefficient $\alpha/r=1$; при $\alpha=16$ он равен 2, contribution меняется даже при прежних $A,B$.

В scalar mask example known pixel 0.7, generated pixel 0.2, mask $m=0$ дают final $(1-m)0.7+m0.2=0.7$. При $m=1$ результат 0.2. Перепутанная convention mask меняет область редактирования на противоположную; shape masks при этом остаётся корректной.

## Самопроверка с ответами

**Практика:** подайте edge map, не прошедшую тот же crop/resize. Control будет направлять модель к смещённой geometry; исправление текста prompt не решает этот mismatch.

**Гарантирует ли LoRA малое влияние?** Нет: rank ограничивает algebraic form update, но его norm и semantic effect могут быть большими.

**Почему zero convolution полезна в начале?** Она начинает без дополнительного control contribution, позволяя постепенно обучить его взаимодействие с pretrained features.

## Иллюстрации из первоисточников

![controlnet: Figure 3 из авторской статьи](<../оформление/course/papers/controlnet.png>)

**Источник:** [controlnet, Figure 3](https://arxiv.org/abs/2302.05543), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [ControlNet — Zhang et al., ICCV 2023](https://arxiv.org/abs/2302.05543)
- [LoRA — Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<08-generation/План.md>)
