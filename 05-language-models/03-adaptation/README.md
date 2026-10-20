# Адаптация

[Модуль «LLM и языковые модели»](<../README.md>) · [Сценарии глав](<План.md>)

- [Задача адаптации, baseline и выбор вмешательства](<adapt-choice.md>) — основа ветки. Как отличить нехватку знаний, ошибку поведения и проблему исполнения?
- [SFT: данные, шаблоны, masks и objective](<adapt-sft.md>) — основа ветки. Какие позиции становятся целями supervised fine-tuning?
- [Полное обучение или PEFT](<adapt-regimes.md>) — основа ветки. Какие параметры обновляются и почему это не меняет смысл SFT?
- [LoRA: low-rank обновление и градиенты](<adapt-lora.md>) — основа ветки. Как low-rank ветвь обновляет конкретную линейную проекцию?
- [DoRA, AdaLoRA, LoRA+ и QLoRA](<adapt-variants.md>) — углубление. Как различаются изменяемый объект, бюджет ранга и формат хранения базы?
- [Adapters, prefix tuning, soft prompts и BitFit](<adapt-alternatives.md>) — углубление. Где добавляются обучаемые параметры и что меняется при inference?
- [RLHF, PPO, DPO, IPO, KTO и ORPO](<adapt-preference.md>) — основа ветки. Как предпочтения превращаются в objective и каких данных он требует?
- [Model soup, TIES, DARE и переключение адаптеров](<adapt-merging.md>) — углубление. Когда операции над весами совместимы и как проверить итоговую функцию?
- [Дистилляция и обучение на траекториях](<adapt-distillation.md>) — углубление. Какой сигнал teacher передаёт student и как проверить перенос?
- [AI feedback и continual learning](<adapt-continual.md>) — углубление. Как обновлять поведение, контролируя источник feedback и забывание?
- [Оценка, perplexity, judge и экспорт](<adapt-evaluation.md>) — основа ветки. Как сравнить candidate с baseline и подтвердить сохранение поведения после экспорта?

**SFT перед LoRA:** [Постановка задачи и паспорт данных](<../../01-data-features/01-reliability/df-task.md>) → [Split, cross-validation и утечки](<../../01-data-features/01-reliability/df-split.md>) → [Dataset, batching и воспроизводимость](<../../01-data-features/01-reliability/df-loading.md>) → [Текст, токенизация и разбиение документов](<../../01-data-features/03-modalities/df-text.md>) → [Языковая модель и её задачи](<../01-foundations/lm-task.md>) → [Токенизация: от текста к индексам](<../01-foundations/lm-tokenizer.md>) → [Embeddings и контекстные состояния](<../01-foundations/lm-states.md>) → [Attention, Q/K/V и несколько голов](<../01-foundations/lm-attention.md>) → [Позиции, causal mask и padding](<../01-foundations/lm-position.md>) → [Transformer-блок: residual, normalization и MLP](<../01-foundations/lm-transformer.md>) → [Encoder, decoder-only и encoder–decoder](<../01-foundations/lm-families.md>) → [Выход модели: LM head, logits и вероятности](<../01-foundations/lm-output.md>) → [Обучение: targets, loss и обновление весов](<../01-foundations/lm-training.md>) → [Обычная генерация по одному токену](<../01-foundations/lm-generation.md>) → [Задача адаптации, baseline и выбор вмешательства](<adapt-choice.md>) → [SFT: данные, шаблоны, masks и objective](<adapt-sft.md>) → [Полное обучение или PEFT](<adapt-regimes.md>) → [LoRA: low-rank обновление и градиенты](<adapt-lora.md>) → [Оценка, perplexity, judge и экспорт](<adapt-evaluation.md>)
