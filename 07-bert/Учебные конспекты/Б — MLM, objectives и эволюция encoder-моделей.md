# Б. Чему учится encoder до появления прикладной головы

[Оглавление](README.md). Модули **6–9**. **Происхождение:** полный BERT-pretraining — дополнение по плану. Чаты про next-token LLM дают полезный контраст, но не заменяют MLM. Каталог ниже — объяснение mechanisms и критериев, а не рейтинг checkpoint для вашей задачи.

## 1. MLM: labels выбраны отдельно от corruption

В masked language modeling выбирается set target positions $\mathcal M$. Input повреждается, encoder получает $\widetilde x$, а target остаётся исходный token $x_t$:

$$
L_{\mathrm{MLM}}=-\frac1{|\mathcal M|}
\sum_{t\in\mathcal M}\log p_\theta(x_t\mid\widetilde x).
$$

Исторический BERT выбирал 15% positions; среди выбранных 80% заменял MASK, 10% случайным token, 10% оставлял unchanged. Последние тоже дают loss: selection mask и actual replacement не одно и то же. На невыбранных positions labels обычно ignored. Attention mask отвечает за context validity, а не за set MLM targets. [Исходный BERT](https://aclanthology.org/N19-1423/).

MLM head преобразует token states, затем проецирует в vocabulary; классический recipe включает transform/nonlinearity/LN перед projection и может использовать shared input/output embeddings. Logits shape $B\times L\times V$; реализация может вычислять только selected positions для экономии. Tying не отменяет gradient из output path.

## 2. CE и точный gradient

Пусть vocabulary size 4. Для selected position 1 prediction $(0.1,0.6,0.2,0.1)$, correct ID 2; для position 2 — $(0.5,0.2,0.2,0.1)$, correct ID 1. Mean CE:

$$
L=-\frac{\log0.6+\log0.5}{2}\approx0.601986.
$$

Derivative softmax-CE по logits одного example — $p-e_y$; после mean по двум positions делим на 2. First row gradient $(0.05,-0.20,0.10,0.05)$, second $(-0.25,0.10,0.10,0.05)$. Sums gradients равны нулю, что соответствует invariance softmax к common logit shift.

Выбранный unchanged token доступен модели в input: objective становится проще, но позиция всё равно supervised по recipe. Это не повод удалить label автоматически. Слишком много unchanged targets способно изменить task difficulty; corruption policy — часть experiment.

MLM loss не factorizes ordinary autoregressive joint likelihood $\prod_t p(x_t\mid x_{<t})$. Pseudo-log-likelihood отдельно маскирует positions и оценивает conditional scores; её exp нельзя без объявления назвать той же perplexity, что causal LM. Different masking/evaluation recipes тоже меняют score.

## 3. NSP, SOP, spans и RTD

NSP обучает отличать настоящий следующий segment от sampled negative. Если negatives случайно приходят из другого domain, auxiliary task может решаться по style вместо discourse relation. SOP в [ALBERT](https://arxiv.org/abs/1909.11942) сравнивает правильный и переставленный порядок segments, изменяя именно supervision coherence. Для «A: оплатите счёт; B: уже оплатил» swapped order и unrelated sentence являются разными negatives.

[RoBERTa](https://arxiv.org/abs/1907.11692) пересматривает training recipe, включая dynamic masking и отсутствие NSP. Controlled conclusion требует учитывать одновременно data, batch и compute; из этого не следует, что NSP всегда вреден для любой архитектуры/задачи.

[SpanBERT](https://arxiv.org/abs/1907.10529) маскирует contiguous spans и добавляет span-boundary supervision. Идея — обучить representations восстанавливать span information по boundaries, что соответствует span-oriented tasks. Это отличается от independent token masking и не даёт автоматически готовые NER labels.

[ELECTRA](https://arxiv.org/abs/2003.10555) использует generator для corruption, а discriminator предсказывает, какие resulting tokens заменены. Если sampled token совпал с original, он не становится incorrect merely because sampling happened. RTD supervises binary decisions по actual input/original relation; generator и discriminator имеют разные objectives. Это не GAN-discriminator, который обязан работать как realtime answer verifier.

Для всех objectives проверяется negative sampling: pair leakage, источник, length и easy artifacts. Сравнение mechanism полезно даже без дорогого собственного pretraining; опубликованные ablations нужно читать с их compute/data constraints.

## 4. Семейства: какую проблему меняют

ALBERT factorizes vocabulary embedding: $V\times E$ плюс $E\times D$ вместо $V\times D$ при $E<D$. Layer parameter sharing уменьшает число unique weights, но shared block всё равно исполняется несколько раз. Memory weights и FLOPs не одно число.

DistilBERT переносит teacher behaviour через distillation и compact architecture. Это обучение student по нескольким signals, а не просто удаление каждого второго слоя без adaptation. Detailed task-specific compression находится в И.

DeBERTa отделяет content/relative-position contributions attention. DeBERTa-v3 использует RTD и gradient-disentangled embedding-sharing recipe; это не просто BERT с другим количеством layers. [DeBERTa-v3](https://arxiv.org/abs/2111.09543). Для реализации смотрят actual config/tokenizer/modules, а не совпадение общего слова Transformer.

XLM-R — multilingual pretraining с общей model/vocabulary; potential transfer сочетается с конкуренцией языков за capacity. Language coverage в corpus не доказывает одинаковое task quality каждого языка. [XLM-R](https://arxiv.org/abs/1911.02116).

## 5. Современный срез: 2024–2026

**ModernBERT** сочетает RoPE, local/global attention и современный FFN/efficiency recipe. Опубликован native context 8192. Вместимость не гарантирует успешные distant dependencies или русскоязычную задачу; нужен соответствующий task-trained checkpoint. [Работа](https://arxiv.org/abs/2412.13663).

**NeoBERT** исследует depth/width и modern recipe; в публикации описан encoder примерно 250M с context 4096 и unified fine-tuning comparisons. Это useful controlled case study, а не доказательство лучшего качества каждого downstream task. [Работа](https://arxiv.org/abs/2502.19587).

**mmBERT** описывает multilingual encoder с phased language inclusion и annealing; публикация ICML 2026 позволяет изучить timing language exposure. При выборе оценивают нужный язык, morphology и target objective отдельно. [Первичная публикация](https://proceedings.mlr.press/v306/marone26a.html).

**Ettin** — paired encoder/decoder suite для более контролируемого сравнения objectives/data/compute. Использовать как пример experiment design: сравнение двух randomly выбранных checkpoints с разными corpora не изолирует влияние mask/objective. [Seq vs Seq](https://arxiv.org/abs/2507.11412).

## 6. Как выбрать кандидатов

Для русского NER важны tokenizer fragmentation, language/task adaptation и span alignment. Для длинной English classification — native length и actual evidence coverage. Для multilingual retrieval нужен retrieval-trained geometry и cross-language evaluation, а не только multilingual MLM backbone.

Shortlist включает model/revision/license, objective, width/depth, position mechanism, context, language/domain и runtime support. Затем один split, fair tuning budget, cost и error slices. Catalog помогает предложить hypothesis; validation выбирает итог.

## 7. Контроль и практика модулей 6–9

1. Даёт ли selected unchanged position loss? **Ответ:** да, если она входит в MLM target set.
2. Почему ALBERT sharing не равно пропуску layers? **Ответ:** weights общие, но вычисления повторяются на каждом применении.
3. Является ли sampled original token заменённым для RTD? **Ответ:** label определяется совпадением с original, а не фактом запуска sampler.
4. Доказывает ли native length качество длинного reasoning? **Ответ:** нет; это capacity context, которое проверяют задачей.

**Практика:** повторите CE/gradients и составьте batch с MASK/random/unchanged selected tokens. Для короткого диалога создайте NSP/SOP/hard-negative cases. По двум configs сравните механизм, data/objective и budget. Для трёх прикладных сценариев выше выберите критерии shortlist, не объявляя winner без evaluation.
