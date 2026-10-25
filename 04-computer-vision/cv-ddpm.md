# DDPM: прямой процесс, noise prediction и обучение

**Уровень:** основа ветки.

**Предпосылки:** [Autoencoder, VAE и GAN: вероятностные основы генерации](<cv-generative.md>)

**Используют:** [DDIM, расписания шума и classifier-free guidance](<cv-sampling.md>)

**Учебный вопрос:** Как обучение предсказанию шума превращается в генерацию изображения?

## Схема вычислений

![DDPM: прямой процесс, noise prediction и обучение: вычислительные этапы](<../оформление/course/expanded-cv-ddpm.svg>)

Training DDPM: Clean x0 + random t, ε → Fixed corruption → xt → Denoiser(xt,t) → ε estimate → Noise MSE → θ update. Sampling DDPM: Gaussian xT → Denoiser → reverse mean → Variance + fresh noise → Повтор до x0 → image.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Прямой процесс фиксирован

DDPM постепенно добавляет Gaussian noise к данным. Пусть $\beta_t\in(0,1)$ — noise schedule, $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Условное распределение одного шага $q(x_t|x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_tI)$. Последовательное раскрытие Gaussian переходов даёт прямой sample любого шага:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
$$

Коэффициенты schedule заданы, а не обучаются обычной denoiser head. [Ho et al.](https://arxiv.org/abs/2006.11239) связывают variational training и denoising; часто используемая simple objective — MSE noise prediction.

## Что учится предсказывать сеть

Denoiser $\epsilon_\theta(x_t,t)$ получает noisy tensor и timestep embedding и возвращает tensor той же shape. Для conditional generation добавляется condition. U-Net связывает multiscale features и timestep signal; output — оценка noise, не class probability и не обязательно готовый clean image. На одном training step выбирают data sample, timestep и noise, строят $x_t$, минимизируют $\|\epsilon-\epsilon_\theta\|^2$. Не требуется последовательно генерировать все промежуточные training states.

Из predicted noise восстанавливается estimate $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}$. При малом $\bar\alpha_t$ ошибка noise сильно усиливается этим делением, поэтому веса timesteps и parameterization имеют значение.

## Обратный sampling

Generative process начинается с $x_T\sim\mathcal N(0,I)$ и последовательно применяет learned reverse transitions. В epsilon-parameterization mean одного шага:

$$
\mu_\theta=\frac1{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

Затем добавляют noise с выбранной reverse variance, кроме terminal step по правилам sampler. Mean formula без указания variance не задаёт весь DDPM sampler. Timestep convention и scaling входа должны совпадать с training.

## Разобранный пример: corruption и reconstruction

В scalar-примере $x_0=0.8$, $\bar\alpha_t=0.64$, $\epsilon=-0.5$. Тогда $x_t=0.8\cdot0.8+0.6(-0.5)=0.34$. При точном noise estimate восстановление $(0.34-0.6(-0.5))/0.8=0.8$. При estimate $-0.4$ получаем $(0.34+0.24)/0.8=0.725$, noise MSE 0.01, ошибка clean estimate $-0.075$.

Signal-to-noise ratio здесь $\bar\alpha/(1-\bar\alpha)=0.64/0.36\approx1.77778$. При $\bar\alpha=0.01$ SNR около 0.01010: почти весь observation — noise. Но difficult timestep не означает отсутствие population-level information, которой учится denoiser.

## Самопроверка с ответами

**Практика:** при том же noise увеличьте $\bar\alpha$ до 0.81. $x_t=0.9\cdot0.8+\sqrt{0.19}(-0.5)\approx0.50206$: чистый сигнал сильнее.

**Почему обучение не требует T network calls на изображение?** Corruption любого выбранного timestep samples напрямую по closed form.

**Предсказывает ли denoiser физически исходный noise однозначно?** По одному noisy observation обычно нет; MSE учит условную оценку, определённую distribution данных.

## Иллюстрации из первоисточников

![ddpm: Figure 2 из авторской статьи](<../оформление/course/papers/ddpm.png>)

**Источник:** [ddpm, Figure 2](https://arxiv.org/abs/2006.11239), PDF, с. 2. Граф показывает прямую и обратную цепи. Сеть denoiser используется многократно, а не является отдельной новой сетью на каждый timestep.

## Источники и продолжение

- [DDPM — Ho et al., NeurIPS 2020](https://arxiv.org/abs/2006.11239)

[Следующие углубления: cv-sampling](<README.md>) · [Сценарии подраздела](<08-generation/План.md>)
