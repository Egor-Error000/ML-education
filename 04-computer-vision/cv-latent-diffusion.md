# Latent diffusion: autoencoder, denoiser и conditioning

**Уровень:** основа ветки.

**Предпосылки:** [DDIM, расписания шума и classifier-free guidance](<cv-sampling.md>), [Vision Transformer: patches, attention и обучение](<cv-vit.md>)

**Используют:** [DiT и flow matching: Transformer и цели обучения](<cv-dit-flow.md>), [ControlNet, inpainting, image-to-image и LoRA](<cv-control.md>), [Оценка генерации: fidelity, diversity, условия и FID](<cv-generation-eval.md>)

**Учебный вопрос:** Что меняется, когда diffusion работает в пространстве сжатых признаков?

## Схема вычислений

![Latent diffusion: autoencoder, denoiser и conditioning: вычислительные этапы](<../оформление/course/expanded-cv-latent-diffusion.svg>)

Autoencoder stage: Image x → Encoder E → latent z → Decoder D → reconstruction → Perceptual / reconstruction loss. Latent diffusion: z + noise + timestep → Text encoder → condition states → U-Net + cross-attention → ε → Sampler → z0 → decoder D.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Два этапа обучения

Latent diffusion сначала обучает image autoencoder, затем generative denoiser в его latent space. Encoder $E$ переводит $x\in\mathbb R^{3\times H\times W}$ в $z\in\mathbb R^{C_z\times H/f\times W/f}$, decoder $D$ возвращает image estimate. [Rombach et al.](https://arxiv.org/abs/2112.10752) отделяют perceptual compression от diffusion modelling; качество compression ограничивает доступные детали последующей генерации.

Первый этап может сочетать reconstruction, perceptual, adversarial и latent regularization losses. Это не означает, что любой LDM использует один и тот же VAE или latent scale. После первого этапа autoencoder обычно фиксируют для обучения denoiser; noise добавляют к latent с согласованным scale. Неправильный scaling factor меняет SNR, даже если shape подходит.

## Denoiser и cross-attention

Второй этап обучает $\epsilon_\theta(z_t,t,c)$ предсказывать latent noise. Text encoder переводит prompt в states $C\in\mathbb R^{L\times D_c}$. В cross-attention spatial states denoiser создают queries, text states — keys и values. Для $N$ spatial positions attention scores имеют $N\times L$, а не $N\times N$. Attention связывает features с condition, но не гарантирует точного количества объектов или binding атрибутов.

U-Net передаёт features между resolution levels и skip connections. Timestep embedding влияет на blocks; text condition поступает в предусмотренные layers. Это три разных потока: noisy latent, timestep и condition. Prompt не заменяет initial noise, decoder не выполняет denoising iterations.

## Sampling и границы переноса

Начинают с random latent, выбирают sampler и guidance, последовательно получают $z_0$, затем один раз декодируют $D(z_0)$. Image-to-image вместо pure noise использует corrupted encoded input. Inpainting требует mask policy и согласования known regions. Image quality, prompt adherence и factual correctness — разные характеристики результата.

## Разобранный пример: размеры и attention

Для $512\times512$ RGB image число values $3\cdot512^2=786432$. При $f=8$, $C_z=4$ latent $4\times64\times64$ содержит 16384 values, в 48 раз меньше. Spatial positions меньше в 64 раза, но это не обещает 48-кратного ускорения: channels blocks и conditioning имеют собственную стоимость.

На resolution $32\times32$ имеется 1024 image queries. При 77 text states одна cross-attention matrix содержит $1024\cdot77=78848$ scores на head. Полное spatial self-attention на том же уровне содержит $1024^2=1048576$ scores. При восьми heads эти числа умножаются на 8; optimization kernels могут не материализовать всю matrix одновременно.

## Диагностика

Если autoencoder плохо восстанавливает мелкий текст, denoiser не получает идеального latent representation этого текста автоматически. Проверяют reconstruction отдельно от generation. При переносе checkpoints сверяют architecture, latent channels, scaling, text tokenizer, prediction type и scheduler contract. Низкий denoising loss не доказывает отсутствие memorization.

## Самопроверка с ответами

**Практика:** увеличьте image size до $1024^2$ при том же $f$. Latent positions вырастут вчетверо; attention и convolution затраты растут по-разному.

**Кто переводит итоговый latent в pixels?** Autoencoder decoder после sampling.

**Почему prompt не является GT для всех pixels?** Он неполно задаёт изображение; множество scenes совместимо с одним описанием.

## Иллюстрации из первоисточников

![ldm: Figure 3 из авторской статьи](<../оформление/course/papers/ldm.png>)

**Источник:** [ldm, Figure 3](https://arxiv.org/abs/2112.10752), PDF, с. 4. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [Latent diffusion — Rombach et al., CVPR 2022](https://arxiv.org/abs/2112.10752)

[Следующие углубления: cv-dit-flow, cv-control, cv-generation-eval](<README.md>) · [Сценарии подраздела](<08-generation/План.md>)
