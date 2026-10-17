# Источники и современные encoder-модели

Проверено: **12 октября 2026**. Это подборка первичных публикаций, документации разработчиков и model cards, связанная с [планом BERT](<../План — BERT и encoder-модели для NLP.md>). Подборка не является рейтингом всех доступных моделей.

## Что записывать при выборе checkpoint

- ID, author, revision, license и task-specific intended use.
- Backbone/architecture, tokenizer, language/domain и pretraining objective.
- Task-trained head или embedding/reranker objective; pooling, normalization, prefixes и label mappings.
- Actual context configuration, runtime support и ограничения длины/языка.
- Целевой held-out набор, metrics по slices, latency/memory и tuning budget.

Название BERT/ModernBERT характеризует семейство; оно само не сообщает, что checkpoint уже обучен выдавать retrieval vectors, relevance score, NER labels или calibrated class probabilities.

## Современные опорные модели

### ModernBERT — английский/кодовый encoder и efficiency case study

[Публикация, декабрь 2024](https://arxiv.org/abs/2412.13663), [официальный base checkpoint](https://huggingface.co/answerdotai/ModernBERT-base), [Transformers model documentation](https://huggingface.co/docs/transformers/main/model_doc/modernbert).

Изучить RoPE, local/global attention, FFN/normalization и исполнение padded/unpadded batches. Native context — 8192 tokens в опубликованном семействе. Для русского применения и sentence retrieval требуется отдельное сравнение подходящего checkpoint: английский/code pretraining не доказывает качество другого языка, а base MLM не заменяет embedding fine-tuning.

Модули: 9, 13, 22, 31, 39. Источник speed/quality claims — соответствующий experiment protocol статьи; переносить цифры на другое hardware/workload нельзя.

### NeoBERT — архитектура, scaling и controlled fine-tuning

[Публикация, февраль 2025; проверенная редакция июня 2025](https://arxiv.org/abs/2502.19587).

Авторы рассматривают encoder около 250M параметров с контекстом 4096 tokens и единым fine-tuning protocol для сравнения. Изучить depth/width, современные архитектурные компоненты и влияние training recipe. Заявленные сравнения относятся к datasets/objectives статьи и требуют downstream проверки под свою задачу.

Модули: 8–9, 13, 24, 35.

### mmBERT — multilingual pretraining и языковая адаптация

[ICML 2026 / PMLR publication](https://proceedings.mlr.press/v306/marone26a.html).

Работа описывает encoder, обученный на multilingual corpus с phased language inclusion и annealing. Изучить, почему момент добавления low-resource languages влияет на обучение и как устроено language-specific evaluation. Наличие большого числа языков в data mix не означает одинаковую надёжность classifier/retriever на каждом из них.

Модули: 9, 14, 30, 35–36.

### Ettin — сравнительное исследование encoder/decoder

[Seq vs Seq: An Open Suite of Paired Encoders and Decoders, июль 2025](https://arxiv.org/abs/2507.11412).

Paired encoder/decoder families позволяют изучать влияние objective при более контролируемом сравнении data/compute/model size. Читать как пример дизайна эксперимента: conclusions для NLU и generation не подменяют evaluation конкретного application.

Модули: 1, 9, 13, 35.

### Multilingual E5 и BGE-M3 — task-trained retrieval models

[Multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-base), [M3-Embedding paper](https://arxiv.org/abs/2402.03216), [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).

Для E5 проверить language coverage, query/document prefixes и pooling конкретного checkpoint. BGE-M3 объединяет dense/sparse/multi-vector retrieval modes; protocol и candidate corpus должны быть одинаковыми при сравнении modes. Это примеры retrieval-trained моделей, поэтому их следует отделять от pretrained MLM backbones.

Модули: 22–24, 26, 28–30.

## Архитектура и objectives: что читать

- [BERT, Devlin et al., 2019](https://aclanthology.org/N19-1423/) — bidirectional pretraining, MLM/NSP, task adaptation; сначала модули 1–6.
- [Официальный исходный BERT repository](https://github.com/google-research/bert) — исторические configs/implementation. Старый TensorFlow code не использовать как обещание совместимости с современным environment.
- [BERT в Transformers](https://huggingface.co/docs/transformers/main/model_doc/bert) — config, output shapes, pooler и task head interfaces. Main docs меняются; для запуска фиксировать release.
- [RoBERTa, 2019](https://arxiv.org/abs/1907.11692) — контролируемый пересмотр recipe и objectives; модуль 7–8.
- [ALBERT, 2019](https://arxiv.org/abs/1909.11942) — factorized embeddings, shared parameters и SOP; модули 7–8.
- [ELECTRA, 2020](https://arxiv.org/abs/2003.10555) — replaced-token detection, generator/discriminator роли; модуль 7.
- [DeBERTa-v3, 2021](https://arxiv.org/abs/2111.09543) — RTD и gradient-disentangled embedding sharing; модуль 8.
- [SpanBERT, 2019/2020](https://arxiv.org/abs/1907.10529) — span-oriented pretraining; модули 7 и 18.
- [XLM-R, 2019](https://arxiv.org/abs/1911.02116) — multilingual transfer и ресурс языков; модули 8 и 30.
- [DistilBERT, 2019](https://arxiv.org/abs/1910.01108) — encoder distillation; модули 8 и 38.
- [Don’t Stop Pretraining, 2020](https://aclanthology.org/2020.acl-main.740/) — DAPT/TAPT и downstream adaptation; модуль 14.

## Прикладные головы и few-shot

- [Transformers: text classification](https://huggingface.co/docs/transformers/main/tasks/sequence_classification) — pipeline для модуля 15; детали loss/labels проверять по постановке.
- [Transformers: token classification](https://huggingface.co/docs/transformers/main/tasks/token_classification) — word_ids, subword alignment и ignored labels; модуль 17.
- [Transformers: question answering](https://huggingface.co/docs/transformers/main/tasks/question_answering) — question/context offsets и span labels; модуль 20.
- [SetFit: Efficient Few-Shot Learning Without Prompts, 2022](https://arxiv.org/abs/2209.11055) — sentence representation adaptation и classification с малой разметкой; модуль 21.
- [RuBERT model card](https://huggingface.co/ai-forever/ruBert-base) — пример Russian-language checkpoint; проверить objective и пригодность конкретного варианта для модуля 30.
- [SciBERT, 2019](https://aclanthology.org/D19-1371/) — scientific domain pretraining; модуль 32.
- [CodeBERT, 2020](https://arxiv.org/abs/2002.08155) — natural/programming language representation; модуль 32.
- [LayoutLM, 2019/2020](https://arxiv.org/abs/1912.13318) — текст и layout в document understanding; модуль 34.

## Embeddings и retrieval

- [Sentence-BERT, 2019](https://aclanthology.org/D19-1410/) — независимое encoding предложений и task-trained geometry; модуль 22.
- [SimCSE, 2021](https://arxiv.org/abs/2104.08821) — contrastive learning и dropout views; модуль 23.
- [Sentence Transformers: losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) — связь dataset format, objective и negative sampling; модули 23–24. API фиксировать по установленной версии.
- [Matryoshka Representation Learning, 2022](https://arxiv.org/abs/2205.13147) — обучение полезных embedding prefixes; модуль 24.
- [CrossEncoder training](https://sbert.net/docs/cross_encoder/training_overview.html) — актуальный workflow reranker и objectives; модуль 27.
- [SPLADE, 2021](https://arxiv.org/abs/2107.05720) — sparse learned lexical representations; модуль 28.
- [SparseEncoder training documentation](https://sbert.net/docs/sparse_encoder/training_overview.html) — современная library implementation и format/objective choices; модуль 28.
- [ColBERTv2, 2021/2022](https://arxiv.org/abs/2112.01488) — late interaction, compression и denoised supervision; модуль 29.
- [Multi-vector model documentation](https://sbert.net/docs/multi_vector_encoder/pretrained_models.html) — отдельный тип моделей и supported loading; модуль 29.
- [Longformer, 2020](https://arxiv.org/abs/2004.05150) — local/global attention для длинных документов; модуль 31.

## Оценка и deployment

- [BEIR, 2021](https://arxiv.org/abs/2104.08663) — heterogeneous retrieval evaluation, task transfer и relevance judgements; модуль 35.
- [MTEB, 2022/2023](https://arxiv.org/abs/2210.07316) — diversity embedding tasks; модули 25 и 35.
- [MMTEB, 2025](https://arxiv.org/abs/2502.13595) — multilingual expansion и task/language slices; модули 30 и 35. Статус лидера из текста статьи относится к дате и protocol статьи.
- [Text Embeddings Inference, официальный repository](https://github.com/huggingface/text-embeddings-inference) — supported architectures/heads, batching и serving; модуль 39. Не переносить benchmark цифры на свою систему без измерений.

## Как читать статью глубоко

1. Выписать task, input/output и supervision до чтения leaderboard.
2. Восстановить shapes/objective и один ручной пример.
3. Отделить architecture change от данных, budget и evaluation protocol.
4. Найти baseline, ablation, language/domain slices и scope вывода.
5. Сформулировать один controlled experiment или теоретический вопрос, который можно проверить самостоятельно.

Приведённые источники подтверждают существование и механизм методов. Выбор лучшего checkpoint, settings и качества для собственного corpus остаётся задачей evaluation.
