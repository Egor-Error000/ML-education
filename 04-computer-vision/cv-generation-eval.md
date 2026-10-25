# Оценка генерации: fidelity, diversity, условия и FID

**Уровень:** основа ветки.

**Предпосылки:** [Autoencoder, VAE и GAN: вероятностные основы генерации](<cv-generative.md>), [Latent diffusion: autoencoder, denoiser и conditioning](<cv-latent-diffusion.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как проверить качество генератора без одного вводящего в заблуждение числа?

## Схема вычислений

![Оценка генерации: fidelity, diversity, условия и FID: вычислительные этапы](<../оформление/course/expanded-cv-generation-eval.svg>)

Feature metric: Real images / generated images → Одинаковый feature extractor → Mean + covariance estimates → FID + protocol / sample count. Conditional evaluation: Фиксированные prompts / controls → Seeds + sampler + budget → Task checks + blind preference → Slices + uncertainty + failures.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Несколько независимых свойств

Fidelity означает правдоподобность отдельных samples, diversity — покрытие разнообразия, condition adherence — соответствие prompt или control. Memorization проверяет повторение training images. Ни одна из этих характеристик не гарантирует остальные. Протокол заранее задаёт prompts, seeds, sampler, steps, guidance, resolution и число samples; best-of-N selection отдельно меняет оценку и стоимость.

## FID и feature distributions

Fréchet Inception Distance сравнивает mean и covariance features real и generated sets, приближая их Gaussian distributions. Для $(\mu_r,\Sigma_r)$ и $(\mu_g,\Sigma_g)$:

$$
FID=\|\mu_r-\mu_g\|^2+\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}).
$$

Это симметричная PSD-запись covariance term. [Heusel et al.](https://arxiv.org/abs/1706.08500) вводят FID как feature-based metric. Features, resize, implementation и sample count существенно влияют на число. При малой выборке covariance estimates шумны; нельзя сравнивать результаты с разными reference sets как один benchmark.

Совпадение первых двух moments не определяет полное distribution. FID не знает, соответствует ли generated изображение конкретному prompt, и не проверяет метрическую geometry. Уменьшение FID не является доказательством отсутствия copies training data.

## Оценка условия и человеческое сравнение

Image-text similarity даёт один сигнал соответствия, но может пропускать счёт объектов, отрицания и relations. Нужны task-specific prompts: число объектов, spatial relations, attribute binding и text rendering. Для control maps отдельно оценивают соответствующую geometry после извлечения depth/pose/edges и ограничения extractor.

Human preference tests скрывают названия моделей, рандомизируют порядок и фиксируют instructions: visual quality и prompt fidelity оценивают раздельно. Несколько ratings одного prompt зависимы; доверительные интервалы должны учитывать grouping. Отдельно смотрят типичные и worst-case samples, а не только выбранные красивые изображения.

## Разобранный пример: FID в одном dimension

Real features имеют mean 0 и variance 1, generated — mean 1 и variance 4. В 1D formula упрощается до $(0-1)^2+1+4-2\sqrt{1\cdot4}=1+5-4=2$. Если mean и variance совпадут, FID станет 0 даже для распределений с разными высшими moments. Значит, «FID=0» не доказывает побитовое совпадение datasets.

В парном сравнении 100 независимых prompts model A выигрывает 60 раз, B — 40, без ties. Оценка preference rate 0.6; standard error Bernoulli approximation $\sqrt{0.6\cdot0.4/100}\approx0.04899$. Приближённый 95% interval 0.504..0.696. Малый перевес требует учитывать uncertainty; эта approximation не заменяет подходящий interval для малых samples и зависимых ratings.

## Диагностика и стоимость

Сравнение quality–cost включает число generated candidates и reranking. Модель с одним sample и pipeline из 32 attempts решают задачу с разным budget. Проверяют coverage по категориям и backgrounds, ближайших training neighbours и ошибки control. Все красивые иллюстрации в курсе — примеры mechanisms, а не собственный benchmark моделей.

## Самопроверка с ответами

**Практика:** выберите лучший из десяти seeds для каждого prompt. Укажите selection policy и десятикратный sampling budget; это уже другой протокол.

**Почему FID не измеряет правильный текст на вывеске?** Сжатые global features не проверяют каждый символ.

**Можно ли по FID одной выборки доказать generalization?** Нет: нужны независимые data, uncertainty и проверки memorization.

## Источники и продолжение

- [FID — Heusel et al., NeurIPS 2017](https://arxiv.org/abs/1706.08500)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<08-generation/План.md>)
