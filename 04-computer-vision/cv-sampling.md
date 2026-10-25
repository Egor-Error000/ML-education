# DDIM, расписания шума и classifier-free guidance

**Уровень:** основа ветки.

**Предпосылки:** [DDPM: прямой процесс, noise prediction и обучение](<cv-ddpm.md>)

**Используют:** [Latent diffusion: autoencoder, denoiser и conditioning](<cv-latent-diffusion.md>)

**Учебный вопрос:** Как менять число шагов и силу условия, не смешивая sampler с обучением?

## Схема вычислений

![DDIM, расписания шума и classifier-free guidance: вычислительные этапы](<../оформление/course/expanded-cv-sampling.svg>)

DDIM step: xt + timestep t → Predict ε → estimate clean x0 → Выбранный следующий noise level s → Согласованный переход → xs. CFG: Noisy latent + condition → Conditional εc; empty εu → εu + scale×(εc−εu) → Guided prediction → sampler.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Три независимых решения

Training noise schedule задаёт распределение $x_t$, inference timestep schedule — какие состояния посещает sampler, guidance — как используется condition. Одинаковый checkpoint допускает разные samplers, но не любую произвольную формулу перехода. Уменьшение числа шагов должно сохранять согласованную parameterization и выбранные noise levels.

## DDIM

[DDIM](https://arxiv.org/abs/2010.02502) строит семейство sampling trajectories, используя тот же тип denoising objective. Для детерминированного варианта и перехода из $t$ к менее noisy $s$ сначала оценивают $\hat x_0$, затем $x_s=\sqrt{\bar\alpha_s}\hat x_0+\sqrt{1-\bar\alpha_s}\epsilon_\theta(x_t,t)$. При stochastic вариантах добавляется согласованный noise term и меняется коэффициент направления. Просто пропускать шаги в формуле соседнего DDPM transition некорректно.

Детерминированность означает одинаковый результат для фиксированных initial noise, weights, condition, schedule и численной реализации. Она не означает одинаковую картинку для разных seeds. Numerical solvers других типов требуют собственной интерпретации timestep и prediction type.

## Classifier-free guidance

При training часть conditions заменяют пустым условием. Одна сеть учит conditional и unconditional predictions. На inference часто используют convention:

$$
\epsilon_g=\epsilon_u+s(\epsilon_c-\epsilon_u).
$$

При $s=0$ получаем unconditional, при $s=1$ — обычное conditional prediction. Другие тексты могут обозначать scale иначе. [Ho и Salimans](https://arxiv.org/abs/2207.12598) вводят classifier-free guidance: отдельный classifier не нужен, но два denoiser predictions обычно нужны. Их batching не делает вычисления бесплатными.

Сильный guidance может повысить соответствие prompt и одновременно снизить diversity, усилить oversaturation и artifacts. Negative prompt в implementations подменяет unconditional branch конкретным condition; это не строгая логическая операция отрицания.

## Разобранный пример: guidance и один переход

Пусть scalar predictions $\epsilon_u=0.2$, $\epsilon_c=0.5$. При $s=3$ результат $0.2+3\cdot0.3=1.1$, выходящий за интервал исходных predictions. Guidance является extrapolation, поэтому сильное условие может увести trajectory в плохо обученные области.

Для $x_t=0.34$, $\bar\alpha_t=0.64$, predicted noise $-0.5$ получили $\hat x_0=0.8$. При $\bar\alpha_s=0.81$ deterministic DDIM step даёт $x_s=0.9\cdot0.8+\sqrt{0.19}(-0.5)\approx0.50206$. Финальный clean estimate и очередное noisy state различаются.

Если sampler использует 30 timesteps и каждый требует conditional/unconditional forward, это до 60 branch evaluations. При времени одного sequential branch 20 ms грубая оценка 1.2 seconds без decoder и overhead. Совместный batch меняет реальное время и peak memory, поэтому измерения важнее этого сложения.

## Самопроверка с ответами

**Практика:** задайте $s=1$. Формула возвращает conditional prediction, а не сумму двух predictions.

**Почему меньше steps может ухудшить результат?** Меняется numerical approximation trajectory; checkpoint обучался не гарантировать точность любого короткого solver.

**Означает ли classifier-free отсутствие дополнительной стоимости?** Нет, отсутствует отдельный classifier, но остаются branch evaluations.

## Источники и продолжение

- [DDIM — Song et al., ICLR 2021](https://arxiv.org/abs/2010.02502)
- [Classifier-free guidance — Ho, Salimans, 2022](https://arxiv.org/abs/2207.12598)

[Следующие углубления: cv-latent-diffusion](<README.md>) · [Сценарии подраздела](<08-generation/План.md>)
