# Генерация изображений

[Модуль «Компьютерное зрение»](<../README.md>) · [Сценарии глав](<План.md>)

- [Autoencoder, VAE и GAN: вероятностные основы генерации](<../cv-generative.md>) — основа ветки. Чем различаются восстановление входа и обучение распределения новых изображений?
- [DDPM: прямой процесс, noise prediction и обучение](<../cv-ddpm.md>) — основа ветки. Как обучение предсказанию шума превращается в генерацию изображения?
- [DDIM, расписания шума и classifier-free guidance](<../cv-sampling.md>) — основа ветки. Как менять число шагов и силу условия, не смешивая sampler с обучением?
- [Latent diffusion: autoencoder, denoiser и conditioning](<../cv-latent-diffusion.md>) — основа ветки. Что меняется, когда diffusion работает в пространстве сжатых признаков?
- [DiT и flow matching: Transformer и цели обучения](<../cv-dit-flow.md>) — углубление. Почему Transformer-backbone и flow matching являются разными выборами?
- [ControlNet, inpainting, image-to-image и LoRA](<../cv-control.md>) — углубление. Как направлять генерацию геометрией, исходным изображением и небольшими обновлениями весов?
- [Оценка генерации: fidelity, diversity, условия и FID](<../cv-generation-eval.md>) — основа ветки. Как проверить качество генератора без одного вводящего в заблуждение числа?

**CV: восстановление и генерация:** [Постановка задачи и паспорт данных](<../../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../../01-data-features/01-reliability/df-loading.md>) → [Изображения, аннотации и синхронные преобразования](<../../01-data-features/03-modalities/df-images.md>) → [Изображение, карты признаков и блоки модели](<../cv-input.md>) → [Классификация, CNN и residual learning](<../cv-classification.md>) → [Восстановление изображений: denoising, deblurring и super-resolution](<../cv-restoration.md>) → [Autoencoder, VAE и GAN: вероятностные основы генерации](<../cv-generative.md>) → [DDPM: прямой процесс, noise prediction и обучение](<../cv-ddpm.md>) → [DDIM, расписания шума и classifier-free guidance](<../cv-sampling.md>) → [Vision Transformer: patches, attention и обучение](<../cv-vit.md>) → [Latent diffusion: autoencoder, denoiser и conditioning](<../cv-latent-diffusion.md>) → [DiT и flow matching: Transformer и цели обучения](<../cv-dit-flow.md>) → [ControlNet, inpainting, image-to-image и LoRA](<../cv-control.md>) → [Оценка генерации: fidelity, diversity, условия и FID](<../cv-generation-eval.md>)
