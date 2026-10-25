# Autoencoder, VAE и GAN: вероятностные основы генерации

**Уровень:** основа ветки.

**Предпосылки:** [Классификация, CNN и residual learning](<cv-classification.md>)

**Используют:** [DDPM: прямой процесс, noise prediction и обучение](<cv-ddpm.md>), [Оценка генерации: fidelity, diversity, условия и FID](<cv-generation-eval.md>)

**Учебный вопрос:** Чем различаются восстановление входа и обучение распределения новых изображений?

## Схема вычислений

![Autoencoder, VAE и GAN: вероятностные основы генерации: вычислительные этапы](<../оформление/course/expanded-cv-generative.svg>)

VAE: Image x → encoder → Mean / variance → z via noise → Decoder → likelihood parameters → Reconstruction + KL loss. GAN: Prior noise z → generator → Generated and real images → Discriminator → real score → Раздельные updates D и G.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Вероятностный вход в генерацию

Генеративная модель должна создавать samples из распределения, похожего на данные. Latent variable $z$ — скрытый vector, prior $p(z)$ задаёт, как его выбирать. Decoder переводит latent в изображение или distribution параметров pixels. Reconstruction одного входа и sampling нового изображения — разные проверки: хороший autoencoder может иметь неудобное latent space для случайного sampling.

## Autoencoder и VAE

Детерминированный encoder $E$ и decoder $D$ минимизируют reconstruction loss $\ell(D(E(x)),x)$. Без regularization произвольный vector из prior не обязан попадать в область, где decoder обучался. [VAE](https://arxiv.org/abs/1312.6114) задаёт approximate posterior $q_\phi(z|x)$ и обучает likelihood reconstruction вместе с KL regularization к prior:

$$
L=-\mathbb E_{q_\phi(z|x)}\log p_\theta(x|z)+D_{KL}(q_\phi(z|x)\|p(z)).
$$

Это отрицательная variational lower bound. Для diagonal Gaussian encoder возвращает $\mu,\log\sigma^2$, sampling записывают $z=\mu+\sigma\odot\epsilon$, $\epsilon\sim\mathcal N(0,I)$. Reparameterization отделяет фиксированный noise от differentiable parameters. Gaussian pixel likelihood с фиксированной variance приводит к scaled MSE; Bernoulli likelihood — к BCE при соответствующем смысле данных.

## GAN

[GAN](https://arxiv.org/abs/1406.2661) обучает generator $G(z)$ и discriminator $D(x)$. Discriminator различает real и generated samples; generator получает gradient через discriminator. Исходная minimax цель $\mathbb E_x\log D(x)+\mathbb E_z\log(1-D(G(z)))$ максимизируется по discriminator и минимизируется по generator. На практике non-saturating generator objective $-\mathbb E_z\log D(G(z))$ даёт иной gradient при слабом generator. Это не reconstruction loss paired target изображения.

Mode collapse означает недостаточное покрытие разнообразия, например generator выдаёт один тип лица. Realistic отдельный sample не доказывает coverage. VAE может давать размытые детали, GAN — резкие, но неполные modes; это характерные риски, а не гарантированный ranking всех реализаций.

## Разобранный пример: KL и sampling

Для одного latent dimension $q=\mathcal N(\mu,\sigma^2)$, $p=\mathcal N(0,1)$:
$KL=\tfrac12(\mu^2+\sigma^2-\log\sigma^2-1)$.
При $\mu=1$, $\sigma=0.5$ получаем $0.5(1+0.25-\ln0.25-1)\approx0.81815$. При $\mu=0,\sigma=1$ KL равна 0. Для noise $\epsilon=-2$ reparameterized sample первого posterior $z=1+0.5(-2)=0$.

Если discriminator даёт generated sample вероятность real 0.1, non-saturating loss generator $-\ln0.1=2.30259$. При 0.8 — 0.22314. Эти значения не являются FID или достоверностью конкретной детали изображения.

## Самопроверка с ответами

**Практика:** занулите KL coefficient у VAE. Обучение приближается к stochastic autoencoder; согласование posterior с prior перестаёт оптимизироваться этой частью цели.

**Равен ли latent истинным факторам сцены?** Не обязательно: disentanglement не следует из одного наличия bottleneck.

**Почему генератору GAN не нужна GT-картинка для каждого noise?** Сигнал задаёт discriminator по distribution samples, а не paired reconstruction.

## Иллюстрации из первоисточников

![vae: Figure 1 из авторской статьи](<../оформление/course/papers/vae.png>)

**Источник:** [vae, Figure 1](https://arxiv.org/abs/1312.6114), PDF, с. 2. Это directed graphical model VAE: сплошные линии описывают генерацию, пунктир — approximate posterior. Это не схема CNN layers.

![gan: Figure 1 из авторской статьи](<../оформление/course/papers/gan.png>)

**Источник:** [gan, Figure 1](https://arxiv.org/abs/1406.2661), PDF, с. 3. Показана динамика распределений и discriminator исходного GAN; рисунок объясняет objective, а не фиксирует архитектуру generator.

## Источники и продолжение

- [VAE — Kingma, Welling, ICLR 2014](https://arxiv.org/abs/1312.6114)
- [GAN — Goodfellow et al., NeurIPS 2014](https://arxiv.org/abs/1406.2661)

[Следующие углубления: cv-ddpm, cv-generation-eval](<README.md>) · [Сценарии подраздела](<08-generation/План.md>)
