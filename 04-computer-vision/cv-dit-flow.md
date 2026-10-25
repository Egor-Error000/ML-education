# DiT и flow matching: Transformer и цели обучения

**Уровень:** углубление.

**Предпосылки:** [Latent diffusion: autoencoder, denoiser и conditioning](<cv-latent-diffusion.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Почему Transformer-backbone и flow matching являются разными выборами?

## Схема вычислений

![DiT и flow matching: Transformer и цели обучения: вычислительные этапы](<../оформление/course/expanded-cv-dit-flow.svg>)

DiT: Noisy latent → patchify → Tokens + position / timestep → Transformer + adaptive norm → Unpatchify → prediction tensor. Flow matching: Noise x0 + data x1 + t → Sample xt along known path → Velocity net → vθ(xt,t) → MSE target; ODE at inference.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Архитектура и objective

DiT отвечает, как параметризовать denoiser, flow matching — какой vector field и по какой цели обучать. Transformer может предсказывать diffusion noise или velocity, а convolutional model тоже способна обучаться flow objective. Нельзя определять метод только названием блока или утверждать, что любой DiT уже является flow model.

## Diffusion Transformer

[DiT](https://arxiv.org/abs/2212.09748) разбивает noisy latent на patches, проецирует их в tokens, добавляет positional information и обрабатывает Transformer blocks. Timestep и class conditioning могут модулировать normalization через adaptive LayerNorm. В adaLN-Zero residual modulation и initialization организованы так, чтобы начальные branches были близки к identity. Финальная projection возвращает patch values, unpatchify — spatial tensor prediction.

Attention cost зависит от числа latent patches. При меньшем patch size tokens больше, что увеличивает стоимость даже при прежнем числе weights. DiT architecture не требует image-space diffusion: исходная работа использует latent inputs. Prediction type и variance outputs задаются конкретной model configuration.

## Conditional flow matching

[Flow Matching](https://arxiv.org/abs/2210.02747) обучает vector field, задающий continuous trajectory между простым distribution и data distribution. Для простого учебного straight path берём noise $x_0$, data $x_1$, $x_t=(1-t)x_0+tx_1$, target velocity $u=x_1-x_0$. Сеть $v_\theta(x_t,t)$ минимизирует MSE к этому target. Sampling интегрирует ODE $dx/dt=v_\theta(x,t)$ от noise к data. Направление времени здесь явно noise→data; другие conventions меняют знаки.

Pairs noise/data могут строиться разными coupling policies. Даже если conditional paths прямые, learned marginal field и numerical trajectories не обязаны быть прямыми для всех samples. Training не требует выполнять полный ODE solver на каждом example, но inference solver нужен.

## Разобранный пример: patch count и velocity

Latent $4\times32\times32$, patch size 2 даёт $16^2=256$ tokens, размер flattened patch $4\cdot2^2=16$. При patch size 1 — 1024 tokens; full attention scores вырастают с 65536 до 1048576 на head, в 16 раз. Число latent scalar values не меняется.

Scalar noise $x_0=-1$, data $x_1=3$, timestep $t=0.25$. Path point $x_t=0.75(-1)+0.25\cdot3=0$, velocity target 4. Если prediction 3.5, squared error 0.25. Для constant velocity 4 Euler step $\Delta t=0.1$ переносит point 0 в 0.4. В реальной learned field velocity зависит от state и timestep; это только ручной пример интегрирования.

## Ограничения и сравнение

Меньше function evaluations не гарантирует одинакового качества или wall-clock time. Фиксируют solver, tolerance, NFE, guidance, resolution и hardware. Сравнивают matching fidelity/diversity и condition adherence. Утверждение «flow всегда лучше diffusion» не следует из различия objectives.

## Самопроверка с ответами

**Практика:** разверните время data→noise. Для того же straight path target velocity меняет знак; сохранить старый solver direction нельзя.

**Является ли DiT новой функцией потерь?** Нет, это architecture family.

**Почему training называют simulation-free?** Target можно sample из conditional path без полного numerical integration learned field на каждом update.

## Иллюстрации из первоисточников

![dit: Figure 3 из авторской статьи](<../оформление/course/papers/dit.png>)

**Источник:** [dit, Figure 3](https://arxiv.org/abs/2212.09748), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [DiT — Peebles, Xie, ICCV 2023](https://arxiv.org/abs/2212.09748)
- [Flow Matching — Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<08-generation/План.md>)
