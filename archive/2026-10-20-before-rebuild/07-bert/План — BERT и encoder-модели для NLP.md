# План: BERT и encoder-модели для NLP — от архитектуры до прикладных систем

[Подробные учебные конспекты — 9 глав](<Учебные конспекты/README.md>): все 40 модулей по группам A–I, с LaTeX, расчётами, ответами и шестью проектными протоколами.

Срез источников: **12 октября 2026**. Под BERT здесь понимается классический Bidirectional Encoder Representations from Transformers и развившееся вокруг него семейство encoder-моделей. План включает современные модели и способы обучения представлений; опубликованные результаты сравниваются только в условиях соответствующих работ.

**Цель:** научиться выбирать encoder, данные, task head, objective и оценку для конкретной NLP-задачи; понимать устройство вычислений и причины ошибок; аргументировать выбор между encoder, классическим baseline и генеративной моделью.

**Материалы:** [конспекты и источники](notes/README.md). Существующие разборы Transformer — в [основах LLM](../06-llm/01-foundations/notes/llm-foundations.md). Общие split/leakage/preprocessing — в [плане по данным](<../01-data/План — данные и качество ML-эксперимента.md>), общая регуляризация — в [своём плане](<../02-regularization/План — регуляризация.md>). Общие LoRA/distillation принципы — в [адаптации LLM](<../06-llm/02-adaptation/План — адаптация LLM.md>); здесь изучается их применение к encoder и выбранной задаче.

Это новая ветка: прежние обсуждения Q/K/V являются предпосылкой, но не подтверждают прохождение BERT, обучение классификатора или retrieval-модели.

## Как использовать большой план

Модуль ниже — группа занятий, а не один длинный ответ. На занятии выбирается одна подтема: проблема → механизм и shapes → маленький пример → 2–3 контрольных вопроса. Сначала можно работать в теоретическом режиме; эксперимент выполняется отдельным этапом.

- **Ядро:** устройство encoder, MLM, tokenization, головы и losses, корректный fine-tuning, оценка и одна прикладная ветка.
- **Углубление:** доказательство/числовой расчёт, альтернативный objective, сравнение методов и ограничений.
- **По задаче:** длинные документы, multilingual, sparse/late-interaction retrieval, доменные и мультимодальные варианты.

Обсуждение, самостоятельное понимание и запуск практики отмечаются независимо. Примеры в источниках и демонстрации преподавателя не считаются пользовательским экспериментом. Календарь зависит от выбранной ветки и доступного времени.

## Навигация

- [A. Архитектура и представления: 1–5](#a)
- [B. Pretraining и эволюция encoder: 6–9](#b)
- [C. Данные, fine-tuning и адаптация: 10–14](#c)
- [D. Задачи с supervised head: 15–21](#d)
- [E. Embeddings и representation learning: 22–25](#e)
- [F. Поиск и reranking: 26–29](#f)
- [G. Языки, домены и специальные задачи: 30–34](#g)
- [H. Оценка, надёжность и интерпретация: 35–37](#h)
- [I. Производительность и deployment: 38–40](#i)
- [Маршруты, проекты и критерии завершения](#projects)

<a id="a"></a>
## A. Архитектура и представления

### 1. Постановка NLP-задачи и место encoder — ядро

**Разобрать:** вход, единицу предсказания и supervision. Sentence classification, multi-label prediction, token tagging, span extraction, pair scoring и vector representation требуют разных выходов. Разделить encoder-only, decoder-only и encoder-decoder по attention mask и training objective.

Bidirectional encoder использует разрешённый контекст с обеих сторон позиции. Это помогает анализировать уже данный текст; обычный BERT не становится autoregressive chat-моделью от добавления текстового prompt. Вместе с task head encoder применим к language inference и extractive QA. Основа — [исходная работа BERT](https://aclanthology.org/N19-1423/).

**На бумаге:** для пяти задач указать вход, labels и output shape: тема письма, список опасных категорий, имена людей, ответ-фрагмент, релевантность документа запросу. Объяснить, когда ответом является класс, span, вектор или новый текст.

**Эксперимент:** для одной задачи сравнить правила/TF-IDF + linear model, frozen encoder + head и fine-tuning. Генеративный baseline добавлять при содержательном сравнении, учитывая весь budget.

**Готово по пониманию:** выбор encoder объясняется свойствами задачи, а ограничения сформулированы до выбора конкретного checkpoint.

### 2. Tokenizer, WordPiece и соответствие тексту — ядро

**Разобрать:** normalization, cased/uncased, словарь, subwords, special tokens, unknown tokens, offsets и границы слов. У классического BERT используется WordPiece; другие encoder могут использовать BPE/SentencePiece и иной набор специальных токенов. Нельзя подменять tokenizer только потому, что совпадает размер словаря.

Изучить input IDs, attention mask и segment/token-type IDs. Для одиночного текста и пары вручную показать, что означает каждый тензор. Segment IDs есть не у всех моделей; необходимость и значения определяются tokenizer/config выбранного checkpoint.

**На бумаге:** разбить русскую фразу с фамилией, отрицанием и дефисом на условные subwords; восстановить word-level labels. Проверить, что происходит с символами после lowercasing/Unicode normalization, с ё/е, латинско-кириллическими омографами и числами.

**Эксперимент:** собрать статистику tokens per word, доли truncation и редких фрагментаций на целевом корпусе; сравнить два tokenizer при одинаковых текстах.

**Готово:** известны политика truncation, место special tokens и способ восстановления исходного char span. Реализация и детали checkpoint — в [документации BERT](https://huggingface.co/docs/transformers/main/model_doc/bert).

### 3. Входные embeddings и Transformer block — ядро с выводом shapes

Для классического BERT разобрать сумму token, position и token-type embeddings, LayerNorm и dropout; затем multi-head self-attention, residual connections и feed-forward network. Сравнить post-LN исходного BERT с pre-LN современного encoder по пути градиента.

~~~text
input_ids: B × L
hidden states X: B × L × D
одна проекция W_Q: D × (H·d_k)
Q/K после reshape: B × H × L × d_k
scores: B × H × L × L
attention output до объединения: B × H × L × d_v
FFN: D → D_ff → D
~~~

Классический BERT-base использует 12 слоёв, D=768 и 12 голов; BERT-large — 24 слоя, D=1024 и 16 голов. Это конфигурации конкретных исходных моделей, а не определение всех BERT-подобных encoder. Проверить значения по [официальному репозиторию](https://github.com/google-research/bert).

**На бумаге:** для B=2, L=5, D=12, H=3 проверить каждое умножение и broadcast маски. Посчитать параметры Q/K/V/O и FFN, включая выбранные bias.

**Эксперимент:** пройти один batch через backbone и отдельный head, распечатать только shapes; проверить train/eval поведение dropout.

**Готово:** матричные соглашения согласованы, а добавление головы не путается с заменой backbone.

### 4. Bidirectional attention, padding и стоимость — ядро

**Разобрать:** padding mask, pair boundaries, отсутствие causal mask в обычном encoder; разрешённые взаимодействия токенов. SEP разделяет сегменты, но сам по себе не блокирует cross-segment attention. Ошибка маски может дать модели доступ к padding или запрещённому контексту.

Для dense attention выделить работу порядка B·L²·D и матрицу attention порядка B·H·L²; отдельно учесть проекции и FFN порядка B·L·D² при сопоставимых ширинах. Реальная memory footprint включает веса, activations, workspace и выбранный attention backend. FlashAttention улучшает обращение к памяти и выполнение; математическая плотность полного attention от этого не исчезает.

**На бумаге:** сравнить L=128 и L=512 при одинаковом D: что растёт в 4 раза, а что в 16? Показать, почему один длинный текст в padded batch может увеличить работу всей группы.

**Эксперимент:** fixed padding против dynamic padding/length bucketing; измерить latency и memory для реального распределения длин, проверив совпадение задачи.

**Готово:** различаются логическая длина, padded length и фактически исполняемые kernels. Формулы attention/backprop повторять точечно по [уже имеющемуся конспекту](../06-llm/01-foundations/notes/llm-foundations.md).

### 5. CLS, pooling, token states и геометрия — ядро

**Разобрать:** last_hidden_state, raw CLS state, pooler_output, masked mean/max pooling, выбор слоя и layer aggregation. Для классического BERT pooler_output добавляет обученное преобразование и tanh; это не тождественно raw CLS. См. [описание выходов модели](https://huggingface.co/docs/transformers/main/model_doc/bert).

Разделить token representation, sentence representation и retrieval embedding. Masked mean должен учитывать padding и выбранную политику special tokens. Исходный MLM checkpoint не гарантирует полезную cosine geometry предложений; именно задача и objective делают embedding пригодным для сравнения.

**На бумаге:** посчитать masked mean матрицы 4×3, где последняя строка — padding. Показать, как случайное включение padding меняет направление/норму вектора.

**Эксперимент:** raw CLS, pooler и mean на одинаковом наборе STS/классификации; не выбирать pooling по test. Обсудить anisotropy, norm distribution и collapse без вывода «красивый UMAP = хорошая модель».

**Готово:** pooling является частью versioned pipeline и проверяется вместе с tokenizer/backbone/head.

<a id="b"></a>
## B. Pretraining и эволюция encoder

### 6. Masked language modeling и его градиенты — ядро с числовым примером

**Разобрать:** выбор позиций для prediction, corruption policy, labels на выбранных позициях и связь MLM head с vocabulary projection. В исходном рецепте BERT выбирают 15% токенов и используют правило 80% MASK / 10% random / 10% unchanged среди выбранных; это исторический рецепт, не обязательная норма для всех encoder. Источник: [BERT](https://aclanthology.org/N19-1423/).

~~~text
H: B × L × D
MLM logits: B × L × V
L_MLM = -(1/|M|) Σ[t∈M] log p(x_t | corrupted input)
для выбранной позиции: grad_logits = probabilities - one_hot(target)
~~~

Masked token может быть выбранным target даже при unchanged corruption. Labels и attention mask решают разные задачи. Разобрать weight tying, transform перед vocabulary projection и усреднение loss при разном числе выбранных позиций.

**На бумаге:** словарь из четырёх токенов, две выбранные позиции; вручную посчитать CE и gradient. Объяснить, почему MLM loss не является обычной autoregressive likelihood всего текста.

**Эксперимент:** маленький MLM batch с проверкой, какие позиции дают loss; сравнить static/dynamic masking без смены evaluation set.

**Готово:** objective воспроизводится независимо от готового collator; pseudo-log-likelihood обсуждается отдельно от perplexity causal LM.

### 7. NSP, SOP, span masking и replaced-token detection — углубление

**Разобрать:** original NSP и sampling negative sentences; SOP в ALBERT; dynamic masking и removal of NSP в RoBERTa; span corruption в SpanBERT; ELECTRA generator/discriminator и RTD labels.

В ELECTRA discriminator классифицирует, заменён ли токен, а generator задаёт процесс corruption; это не discriminator из inference-фреймворка. Разобрать случай, когда sampled token совпал с исходным, и где возникает supervision. Общий тезис «NSP всегда нужен/всегда вреден» заменить сравнением выбранного objective и controlled ablation.

**На бумаге:** для короткого диалога построить positive NSP, swapped SOP и lexical hard negative. Показать, как неверный sampling позволяет решить auxiliary task по простому артефакту.

**Эксперимент:** небольшой controlled objective comparison только при достаточном budget; иначе изучить ablations статей с одинаковым compute/data и выписать ограничения выводов.

**Готово:** сравнение отделяет architecture, data, objective и training budget. Первоисточники: [RoBERTa](https://arxiv.org/abs/1907.11692), [ALBERT](https://arxiv.org/abs/1909.11942), [ELECTRA](https://arxiv.org/abs/2003.10555), [SpanBERT](https://arxiv.org/abs/1907.10529).

### 8. Семейство BERT: какие изменения решают какие проблемы — ядро по смыслу

**Разобрать:** BERT как исходная точка; RoBERTa как пересмотр training recipe; ALBERT как factorized embeddings и parameter sharing; DistilBERT как distillation; DeBERTa как disentangled content/position processing и DeBERTa-v3 как RTD/embedding-sharing recipe; XLM-R как multilingual pretraining.

Отличать уменьшение числа уникальных параметров от уменьшения FLOPs: shared layer weights всё ещё могут многократно исполняться. Distillation переносит поведение teacher, а не просто удаляет каждый второй слой. Architectural similarity не означает совпадение checkpoint keys, tokenizer или task-head интерфейсов.

**На бумаге:** для каждой модели сформулировать «проблема → изменение → возможный компромисс». По двум config найти width, depth, position mechanism, vocabulary и objective; не сравнивать только parameter count.

**Эксперимент:** одна задача, одинаковый split и tuning budget, несколько подходящих encoder; отчёт об ошибках, cost и seed variation.

**Готово:** каталог моделей служит проверке гипотез. Дополнительные первоисточники: [DistilBERT](https://arxiv.org/abs/1910.01108), [DeBERTa-v3](https://arxiv.org/abs/2111.09543), [XLM-R](https://arxiv.org/abs/1911.02116).

### 9. Современные encoder-модели: срез 2024–2026 — ядро выбора, углубление архитектуры

**ModernBERT:** изучить RoPE, сочетание local/global attention, gated FFN, training data/recipe и efficiency engineering. В опубликованной модели предусмотрен native context 8192 токена; это техническая вместимость, а не гарантия решения любой long-context задачи. [Работа](https://arxiv.org/abs/2412.13663).

**NeoBERT:** проверить depth/width, position/normalization/FFN решения и protocol оценки. Авторы описывают 4096-token encoder; сравнение в статье проводится при конкретных fine-tuning условиях. [Работа](https://arxiv.org/abs/2502.19587).

**mmBERT:** multilingual encoder с phased data inclusion/annealing; изучить отбор языков и task-specific transfer. Работа опубликована на ICML 2026. Широта pretraining не доказывает одинаковое качество на всех языках. [Первичная публикация](https://proceedings.mlr.press/v306/marone26a.html).

**Ettin:** paired encoder/decoder models для более контролируемого сравнения objectives, model sizes и compute/data. Читать как исследовательский дизайн сравнения, а не обещание безусловного превосходства одного типа модели. [Seq vs Seq](https://arxiv.org/abs/2507.11412).

**На бумаге:** выбрать кандидатов для русского NER, длинной английской классификации и multilingual retrieval; указать, какой task-trained checkpoint нужен поверх pretrained backbone.

**Эксперимент:** classical baseline + современный encoder на целевом data slice. Проверить revisions, license, tokenizer, language/domain и actual runtime support.

**Готово:** сделан обоснованный shortlist, включая ограничение по памяти/длине/языку. Актуальные опорные источники — в [отдельной подборке](<notes/Источники и современные encoder-модели.md>).

<a id="c"></a>
## C. Данные, fine-tuning и адаптация

### 10. Датасет, split и annotation protocol — ядро до обучения

**Разобрать:** annotation unit, label ontology, ambiguous/unknown cases, duplicate texts и boilerplate; group/time/domain splits. Для пар и retrieval исключать утечку сущностей/документов между train/test, а не только случайно делить строки.

В NER фиксировать char spans и policy nesting; в QA — источник контекста, допустимые ответы и unanswerable cases; в retrieval — queries, corpus и qrels с происхождением judgement. Не считать unlabelled doc автоматически нерелевантным. Синтетические labels требуют качества, teacher provenance и held-out human evaluation.

**На бумаге:** найти утечку в корпусе писем одной цепочки, парах-дубликатах и overlapping chunks одного документа. Составить маленькую annotation guideline с пограничными примерами.

**Эксперимент:** versioned data manifest, дедупликация до split, проверка group overlap и baseline без neural encoder.

**Готово:** видно, что именно измеряет test, какой shift ожидается в deployment и почему часть ambiguity нельзя «исправить LR». Общая методика — в [данных](<../01-data/План — данные и качество ML-эксперимента.md>).

### 11. Task head, loss и pipeline contract — ядро

**Разобрать:** backbone vs MLM/classification/tagging/QA head; случайная инициализация новой головы; конфигурация num_labels/id2label/problem_type. Название AutoModel-класса не заменяет проверку output и labels. Точное API сверять с установленной версией [Transformers](https://huggingface.co/docs/transformers/main/model_doc/bert).

~~~text
sequence head: pooled [B,D] → logits [B,C]
token head: H [B,L,D] → logits [B,L,C]
QA head: H [B,L,D] → start/end logits [B,L]
embedding head: H → pooling/projection → vectors [B,d_e]
~~~

Разделить single-label CE, independent multi-label BCE, regression MSE/Huber и contrastive loss. Softmax по классам не подходит для независимых labels; применение sigmoid ко всем heads тоже неверно.

**На бумаге:** для B=2, C=3 построить labels для single-label и multi-label задачи и посчитать один loss. Определить, где label отсутствует и нужна mask.

**Эксперимент:** overfit tiny clean batch, проверить gradient flow к backbone/head, permutation/label-map ошибки и сохранение/reload.

**Готово:** контракт tokenizer → model → objective → decoding → metric описан и проверяем.

### 12. Full fine-tuning, frozen features и PEFT — ядро сравнения

**Разобрать:** frozen encoder + linear/MLP probe; head-only warmup; full fine-tuning; постепенное размораживание; layer-wise learning-rate decay. Учить отдельно оптимизацию новых head-параметров и pretrained backbone.

LoRA/adapters применимы к encoder, но target modules и task head нужно выбирать по реальной архитектуре. Проверить, что head действительно trainable и сохраняется вместе с adapter. Смысл low-rank update изучен в [адаптации](<../06-llm/02-adaptation/План — адаптация LLM.md>); здесь сравнить encoder attention/FFN placements и budget.

**На бумаге:** посчитать trainable parameters головы и выбранных LoRA-матриц; объяснить, почему frozen base всё ещё занимает память и иногда требует backward activations.

**Эксперимент:** head-only, full и PEFT на одинаковом split/model; сравнить learning curves, стабильность, память, cost и regression slices. Не увеличивать tuning budget только у одного метода.

**Готово:** выбран режим адаптации, а вывод о дешевизне подтверждён accounting и измерением, а не одним числом adapter params.

### 13. Training loop и диагностика оптимизации — ядро с ручной проверкой

**Разобрать:** AdamW, decay groups, warmup/schedule, clipping, dropout, mixed precision, gradient accumulation и checkpoint selection. Значения LR/epochs/weight decay выбираются по validation; универсального «правильного LR для BERT» нет.

Проверить train/eval mode, padding labels, размер effective batch, порядок zero_grad/backward/step и воспроизводимость. Для contrastive objectives gradient accumulation без объединения embeddings не создаёт такой же набор negatives, как один большой batch.

**На бумаге:** рассчитать число optimizer steps из N, per-device batch, world size и accumulation; учесть last batch и scheduler units. Разделить exploding loss, отсутствующий gradient, label noise и validation leakage.

**Эксперимент:** tiny-batch overfit → несколько seeds → LR/length ablation с фиксированным budget. Записывать learning curves, gradient norms, memory и число обработанных target units, а не только epochs.

**Готово:** можно воспроизвести run из config и определить, был ли прирост следствием нового метода, увеличенного budget или иного split.

### 14. Domain/task-adaptive MLM и continual adaptation — углубление

**Разобрать:** DAPT на доменном корпусе и TAPT на данных целевой задачи с выбранным MLM objective; continued pretraining → supervised tuning. Учитывать стоимость и forgetting; улучшение MLM loss не гарантирует улучшения downstream task. Основа: [Don’t Stop Pretraining](https://aclanthology.org/2020.acl-main.740/).

Проверить происхождение unlabeled corpus и transductive assumptions. Использование test texts для adaptation должно быть отдельным явно обозначенным протоколом, а не скрытым «нет labels — значит нет утечки».

Разобрать vocabulary mismatch, learning rate, mixing general/domain data, replay и teacher regularization. Добавление новых tokens требует обучения их embeddings и проверки tokenizer compatibility.

**На бумаге:** выбрать между DAPT, SFT/head tuning и новым domain-trained checkpoint для маленького шумного корпуса; объяснить, какие наблюдения могли бы изменить выбор.

**Эксперимент:** исходный checkpoint, DAPT→task tuning и task tuning без DAPT при сопоставимом total budget; проверить general-domain regression.

**Готово:** решение основано на downstream metrics и проверках забывания, а не только на уменьшении pretraining loss.

<a id="d"></a>
## D. Задачи с supervised head

### 15. Классификация текста, intent, sentiment и moderation — ядро

**Разобрать:** single-label topic/intent classification, short/long texts, class imbalance и semantic overlap. Pooling + linear/MLP head, CE/class weights, confusion matrix и per-class errors. [Официальный task guide](https://huggingface.co/docs/transformers/main/tasks/sequence_classification).

Sentiment не совпадает с topic/stance; негативное слово в цитате не обязательно означает негатив автора. Moderation зависит от контекста, policy и annotation guideline; оценить negation, quotation, sarcasm и domain vocabulary.

**На бумаге:** создать пары с изменением только отрицания, цитаты или адресата и ожидаемой сменой label. Выделить ошибки label policy отдельно от model mistakes.

**Эксперимент:** TF-IDF baseline, frozen embeddings + linear model, full encoder tuning; macro-F1, per-class recall, calibration и time/domain split. Рассмотреть ordinal objective, если классы имеют порядок.

**Готово:** известны ошибочные классы, цена false positives/negatives и влияние truncation. Accuracy на сбалансированном игрушечном наборе не заменяет целевую оценку.

### 16. Multi-label, hierarchical и selective classification — углубление после 15

**Разобрать:** BCEWithLogits по независимым labels, label dependency, missing labels и отрицательные примеры. Hierarchical taxonomy может требовать согласованности child/parent, но метод должен следовать реальной структуре labels.

Изучить global/per-label thresholds, micro/macro F1, PR-AUC и precision/recall targets. Выбор threshold — на calibration/validation set. Reject/abstain label требует правила решения и покрытия, а не просто «маленький softmax score».

**На бумаге:** sigmoid scores трёх labels и два threshold policy; посчитать TP/FP/FN, сравнить micro и macro. Показать ситуацию, когда редкий label исчезает из предсказаний при хорошей общей micro-F1.

**Эксперимент:** сравнить pooled head и joint/hierarchy-aware вариант; проверить рост precision ценой coverage, калибровку редких labels и slice performance.

**Готово:** threshold policy версионируется вместе с head; taxonomy edits имеют explicit migration/retraining plan.

### 17. NER, POS и slot filling: от subwords к spans — ядро

**Разобрать:** token classification head, BIO/BILOU, label alignment и char offsets. Special tokens и padding обычно исключаются из loss; continuation subwords либо mask-ятся, либо получают согласованные labels по выбранной схеме. [Token classification guide](https://huggingface.co/docs/transformers/main/tasks/token_classification).

Нужно отличать word-level label, subword label и финальный entity span. Для NER считать entity-level strict F1, per-type quality и boundary errors; token accuracy легко завысить из-за класса O. Для slot filling учитывать связку с intent и полноту необходимых slots.

**На бумаге:** токенизировать составную фамилию/организацию, перенести BIO labels, восстановить span. Проверить, что B→I policy при дроблении слова не создаёт отдельную сущность.

**Эксперимент:** linear token head, затем CRF при основаниях; сравнить границы, незаконные transitions, rare entities и noisy/OCR inputs.

**Готово:** prediction преобразуется в исходный char span и тип без потери alignment; strict и relaxed span metrics не смешиваются.

### 18. Span extraction, nested NER, relations и events — по задаче

**Разобрать:** token tagging против span classification; nested/overlapping entities; span representation из endpoints/pooling; relation head над парой entity representations. Число всех spans растёт квадратично по L до ограничения максимальной длины.

Изучить staged NER→RE и joint extraction, propagation of entity errors, негативные пары и NA class. Для event extraction разделить trigger, argument и role. Coreference выделить как отдельную сложную ветку с antecedent scoring и document context.

**На бумаге:** текст с двумя вложенными сущностями и отрицанием отношения. Построить candidate spans и указать, какие annotations нельзя выразить одним плоским BIO sequence.

**Эксперимент:** gold entities против predicted entities в RE; boundary/type/role ablation; split по сущностям и документам, чтобы не измерять memorization.

**Готово:** качество измеряется для всей pipeline, а не только conditional head на идеальных upstream inputs. Span-oriented pretraining можно изучить по [SpanBERT](https://arxiv.org/abs/1907.10529).

### 19. NLI, paraphrase и scoring пары текстов — ядро для reranking

**Разобрать:** совместное кодирование пары premise/hypothesis, sentence pair inputs и pooled classification head. NLI labels entailment/neutral/contradiction не совпадают с похожестью, paraphrase или relevance. Изменение порядка пары может менять задачу.

Для STS регрессии сопоставить label scale, MSE/Huber и rank correlations. Для paraphrase/duplicate detection исследовать сильные lexical shortcuts и asymmetric cases.

**На бумаге:** составить пары «лексически близкие, но противоречат», «разные слова, но entailment», «общая тема без ответа». Указать expected labels отдельно для NLI, STS и query-document relevance.

**Эксперимент:** bi-encoder pair features против cross-encoder; один held-out protocol и бюджет. Проверить влияние truncation второй части пары.

**Готово:** task-trained NLI checkpoint не объявляется универсальным scoring model без целевого evaluation.

### 20. Extractive question answering и ответ «не найден» — ядро прикладной ветки

**Разобрать:** question+context, start/end head, offsets, mask допустимых позиций и windowing длинного контекста. [QA guide](https://huggingface.co/docs/transformers/main/tasks/question_answering).

~~~text
start_logits[i] = w_sᵀ h_i
end_logits[j]   = w_eᵀ h_j
L = CE(start_logits, true_start) + CE(end_logits, true_end)
выбор span: max(start_logits[i] + end_logits[j])
при i ≤ j, допустимой длине и позициях только контекста
~~~

Unanswerable detection, несколько допустимых spans, ответы на границе windows и aggregation across windows требуют отдельного решения. Extractive QA возвращает фрагмент данного контекста; убедительный span ещё не означает наличие верного ответа.

**На бумаге:** вручную сопоставить char answer с token positions; разобрать окно, в котором ответ отсутствует. Выбрать валидный span при максимумах start/end в несовместимом порядке.

**Эксперимент:** EM/F1 + unanswerable precision/recall и calibrated abstention; retrieval+QA оценить отдельно от QA на gold context.

**Готово:** pipeline не выбирает ответы из вопроса/padding и сохраняет документ/offsets для проверки.

### 21. Zero-shot, few-shot, NLI labels, MLM verbalizers и SetFit — углубление

**Разобрать:** NLI-based candidate labels, hypothesis templates и multi-label versus mutually-exclusive decision. Zero-shot здесь означает отсутствие task-specific labels при адаптации к новой label set, а не отсутствие pretraining/supervised NLI обучения.

MLM verbalizer использует label words в MASK-позициях; multi-token labels, prompt sensitivity и calibration требуют явного протокола. Это специализированная схема, а не получение general chat behavior.

SetFit сочетает адаптацию sentence representations на парах и classification head для малой разметки. [Первичная работа](https://arxiv.org/abs/2209.11055).

**На бумаге:** показать, как смена phrasing candidate label меняет задачу; разделить few-shot examples и test data. Объяснить, почему generated pairs из одних и тех же 16 текстов не становятся тысячами независимых наблюдений.

**Эксперимент:** несколько независимых few-shot samples/seeds; сравнить NLI zero-shot, frozen vectors, SetFit и ordinary fine-tuning. Отчитываться о variance и качестве rare classes.

**Готово:** не делается универсальный вывод по одному удачному sample/prompts; оценивается реальная потребность в разметке.

<a id="e"></a>
## E. Embeddings и representation learning

### 22. Sentence embeddings: SBERT, pooling и similarity — ядро

**Разобрать:** siamese/shared-weight encoders, pooling/projection и comparison function. В bi-encoder тексты кодируются независимо; document vectors можно вычислить заранее. В cross-encoder взаимодействие пары происходит внутри attention, поэтому один standalone document vector обычно не заменяет pair scoring.

SBERT адаптирует encoder для sentence representations и сравнения вектором, а не просто переименовывает обычный BERT. [Первичная работа](https://aclanthology.org/D19-1410/).

~~~text
z_x = normalize(pool(Encoder(x)))
z_y = normalize(pool(Encoder(y)))
cosine(z_x,z_y) = z_xᵀ z_y при единичных нормах
||z_x-z_y||² = 2 - 2·z_xᵀ z_y при единичных нормах
~~~

**На бумаге:** сравнить cosine, dot product и Euclidean на трёх векторах с разными нормами. Объяснить, почему равенство ranking после normalization не означает совпадения ranking исходных vectors.

**Эксперимент:** pretrained MLM + mean против sentence-trained checkpoint; проверить pooling, normalization и prefixes выбранной модели.

**Готово:** sentence embedding checkpoint выбирается по target task, а не по одному имени backbone; полученные vectors воспроизводятся из versioned pipeline.

### 23. Contrastive objectives, batch и hard negatives — ядро с выводом loss

**Разобрать:** pair/triplet/cosine-regression losses, multiple negatives ranking/InfoNCE и температурный scale. Формат данных должен соответствовать objective. SimCSE использует разные dropout views в unsupervised режиме и NLI supervision в supervised; два identical embeddings без stochasticity — другая постановка. [SimCSE](https://arxiv.org/abs/2104.08821).

~~~text
для N anchor-positive пар:
scores S[i,j] = sim(q_i, d_j) / τ
L_i = -log(exp(S[i,i]) / Σ[j=1..N] exp(S[i,j]))
L = mean_i L_i
~~~

Обсудить symmetric objectives, explicit hard negatives, multi-positive labels и false negatives. Дубликаты, парафразы и несколько релевантных документов нельзя автоматически использовать как отрицательные. Mining должен отделяться от held-out evaluation; teacher scores не являются безошибочной разметкой.

**На бумаге:** матрица scores 3×3, вычислить одну строку loss и gradient по logits. Показать, что false negative заставляет model отталкивать релевантный текст.

**Эксперимент:** random против mined negatives, несколько batch sizes и cached loss при fixed data budget; отдельно проверить влияние actual negative pool и learning rate. Актуальные реализации — в [Sentence Transformers losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html).

**Готово:** учитывается различие effective optimization batch и contrastive negative pool; рост batch не объявляется бесплатным.

### 24. Embedding distillation, Matryoshka и геометрия — углубление

**Разобрать:** matching embeddings, pairwise scores и rankings teacher; projection между размерностями и различие temperature KL/score MSE objectives. Не смешивать язык teacher, corpus coverage и capacity student с выбором loss.

Matryoshka обучает полезность нескольких префиксов vector representation. Усечение первых координат обычного embedding не обладает тем же свойством без обучения/проверки. После truncation нормировку и similarity согласуют заново. [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147).

Изучить anisotropy, hubness, norm drift и representation collapse. Whitening/centering/projection — отдельные обучаемые/подбираемые преобразования; их fit на test создаёт иной протокол.

**На бумаге:** определить loss для teacher dimension 768 и student 256; показать, какие pair relationships сохраняются при score matching, но не гарантируются coordinate MSE.

**Эксперимент:** full-size student, distilled student и несколько Matryoshka dimensions; сравнить retrieval/classification quality, storage и reindexing cost.

**Готово:** экономия measured, teacher overhead включён в training budget; compatibility разных embedding versions не предполагается автоматически.

### 25. STS, deduplication, clustering, anomaly detection и рекомендации — по задаче

**Разобрать:** similarity как обученная функция, а не универсальное определение «одинакового смысла». Semantic similarity, topical relatedness, duplicate identity и retrieval relevance требуют разных labels.

Для dedup оценивать precision pairs/clusters, transitive closure errors и последствия объединения разных записей. Для clustering использовать подходящую геометрию, стабильность и предметную оценку; details algorithms — в [кластеризации](<../04-clustering/План — кластеризация.md>). Для anomaly detection отделять необычный текст, новую тему и опасное содержание.

Для рекомендаций encoded item text может помогать cold start; текстовая близость не является user preference. Collaborative labels, exposure/popularity bias и temporal evaluation — отдельная постановка.

**На бумаге:** три текста A≈B, B≈C, но A не дубликат C; разобрать graph threshold clustering. Сформулировать labels для каждого прикладного случая.

**Эксперимент:** lexical baseline, embeddings и task-adapted embeddings; precision/recall dedup, STS correlation и оценка clusters отдельно. Сохранить ошибки редких/длинных/числовых текстов.

**Готово:** одна STS-метрика не переносится на все приложения. Полезная рамка разнообразия задач — [MTEB](https://arxiv.org/abs/2210.07316).

<a id="f"></a>
## F. Поиск и reranking

### 26. Dense retrieval: query encoder, document encoder и индекс — ядро

**Разобрать:** symmetric/asymmetric retrieval, shared/separate weights, query/document prefixes, pooling и contrastive supervision. Corpus vectors вычисляются offline, queries — online; ANNs приближают поиск в выбранной геометрии.

Пример современных task-trained embeddings — [multilingual E5](https://huggingface.co/intfloat/multilingual-e5-base); prefixes и recommended usage следует брать из model card конкретной revision. MLM backbone, sentence-trained model и retrieval-trained model — разные checkpoints, даже если используют одно семейство encoder.

Сравнить exact search и ANN; ANN recall считается относительно exact vector search, а semantic retrieval recall — относительно qrels. Это две разные проверки. Изучить filtering, namespaces, IDs, document/chunk aggregation и timestamp.

**На бумаге:** дать query/doc score matrix и выбрать top-k; показать, как pooling chunks в документ меняет ranking. Посчитать только vector storage как N·d_e·bytes, затем добавить index/metadata.

**Эксперимент:** BM25, dense и hybrid на фиксированных queries; Recall@k, nDCG/MRR, language/domain slices и latency. Corpus preprocessing — в [плане по данным](<../01-data/План — данные и качество ML-эксперимента.md>).

**Готово:** качество первого этапа отделено от reranker и генератора; индекс привязан к embedding revision и policy доступа.

### 27. Cross-encoder reranking и learning to rank — ядро после 19/26

**Разобрать:** совместное encoding query+document, pair relevance score и применение к candidate set. Чем выше k, тем больше работы; reranker не вернёт отсутствующий в candidate pool документ.

~~~text
pointwise BCE: relevance label для пары
pairwise: L = log(1 + exp(-(s(q,d+) - s(q,d-))))
listwise: supervision над набором документов одного query
~~~

Разделить binary relevance, graded relevance, distillation scores и probabilistic calibration. Score разных queries может иметь разные scales; ranking logits не автоматически пригодны как единый confidence threshold.

**На бумаге:** один query, пять candidates и relevance grades; вычислить pairwise loss и показать конфликт noisy/false-negative пары. Оценить cost при k=20 и k=100.

**Эксперимент:** pointwise против pairwise/listwise objective при сопоставимом data budget; checkpoint selection на held-out queries, не на passage rows. Современный API/training — в [CrossEncoder documentation](https://sbert.net/docs/cross_encoder/training_overview.html).

**Готово:** измерен прирост ranking поверх неизменного retrieval baseline, включая tail latency и candidate recall ceiling.

### 28. Learned sparse retrieval, SPLADE и hybrid search — углубление

**Разобрать:** lexical matching/BM25, vocabulary-space sparse representation и learned term expansion. SPLADE использует encoder/MLM-vocabulary signals для sparse query/document vectors и управляет sparsity; это не просто dense embedding с большим количеством нулей. [SPLADE](https://arxiv.org/abs/2107.05720).

~~~text
упрощённый пример vocabulary weights:
v_j = max_t log(1 + ReLU(logit[t,j]))
score(q,d) = Σ_j q_j · d_j
~~~

Конкретные pooling/regularization/query encoder choices различаются по вариантам. Рассмотреть L1/FLOPS-style regularization, inverted index и frequency skew; слишком широкая expansion может повышать latency и давать ложную lexical relevance.

**На бумаге:** малый словарь, two token positions и sparse weights; вычислить dot score и effect of extra expansion term. Объяснить sparsity/quality trade-off.

**Эксперимент:** BM25, learned sparse, dense и hybrid; одинаковый corpus, relevance labels и budget, tuning fusion на validation.

**Готово:** fusion не подбирается по test, а увеличение recall оценено вместе с cost и влиянием доменных терминов.

### 29. ColBERT и multi-vector late interaction — углубление

**Разобрать:** token-level vectors, query/document projection, masking и MaxSim. В отличие от cross-encoder, document representations могут индексироваться заранее; в отличие от single-vector retrieval, один документ представлен набором vectors.

~~~text
Q: L_q × d_e; D: L_d × d_e
score(q,d) = Σ[i=1..L_q] max[j∈valid_doc_tokens] Q_iᵀ D_j
~~~

Изучить storage, compression, candidate generation и late-interaction computation. Special tokens, punctuation и query padding имеют implementation-specific policies; формулу нужно читать вместе с tokenizer/masks. [ColBERTv2](https://arxiv.org/abs/2112.01488).

**На бумаге:** два query vectors и три document vectors; посчитать MaxSim и показать, какое точное совпадение теряется при mean pooling.

**Эксперимент:** single-vector, late interaction и cross-encoder на одной retrieval task; показать quality/storage/latency trade-offs. BGE-M3 — опубликованный пример dense, sparse и multi-vector modes, а не доказательство их равного качества на каждом корпусе. [M3-Embedding](https://arxiv.org/abs/2402.03216).

**Готово:** budget включает размер индекса и offline encoding; benchmark claims не отрываются от experiment settings.

<a id="g"></a>
## G. Языки, домены и специальные задачи

### 30. Русский язык, multilingual и cross-lingual transfer — ядро выбранной языковой ветки

**Разобрать:** morphology, cased/uncased, token fragmentation, transliteration, code switching, named entities и multilingual sampling. Сравнить task-appropriate Russian encoder, mBERT/XLM-R и современный multilingual encoder; language count в pretraining не является гарантией качества русского.

Для cross-lingual retrieval разделить query language и document language. Для transfer отделить training on one language, translated supervision, multilingual task fine-tuning и genuinely zero-shot language evaluation.

**На бумаге:** одна задача на русском и английском с одинаковой label ontology; разобрать ошибки перевода negation/entity boundaries. Посчитать token inflation для нескольких tokenizer.

**Эксперимент:** фиксированный русскоязычный held-out set + code-switch slice; mono/multilingual baselines, calibration и transfer по каждому языку. У bilingual pairs исключить попадание translation-equivalent примеров в разные splits.

**Готово:** язык — явная ось оценки. Опорные checkpoints/публикации: [RuBERT model card](https://huggingface.co/ai-forever/ruBert-base), [XLM-R](https://arxiv.org/abs/1911.02116), [mmBERT](https://proceedings.mlr.press/v306/marone26a.html).

### 31. Длинные документы и hierarchical encoding — углубление

**Разобрать:** truncation, overlapping windows, hierarchical sentence/chunk encoders, sparse/global attention и long-context native checkpoint. Full document labels и chunk labels не взаимозаменяемы; присвоение label всем chunks может породить слабую/ошибочную разметку.

Сравнить Longformer local/global attention с sliding BERT и современными long-context encoder. Не увеличивать max length в config классического checkpoint без обсуждения learned positions и адаптации. [Longformer](https://arxiv.org/abs/2004.05150); [ModernBERT](https://arxiv.org/abs/2412.13663).

**На бумаге:** документ из восьми chunks, где evidence есть в двух; предложить mean/max/attention aggregation и объяснить возможный shortcut. Рассмотреть ответ, разрезанный границей окна.

**Эксперимент:** truncation, chunk hierarchy и native long context; одинаковые полные документы, метрики по длине/evidence position и memory/latency.

**Готово:** длина учитывается вместе с information coverage; факт «модель принимает 8192 tokens» не заменяет оценку distant dependencies.

### 32. Domain-specific encoder: наука, специальные тексты и код — по задаче

**Разобрать:** term distribution, vocabulary, domain pretraining и label policy. SciBERT — пример encoder для scientific text; CodeBERT — для natural/programming language representation. Это специализированные corpora/objectives, а не универсальная гарантия успеха. [SciBERT](https://aclanthology.org/D19-1371/), [CodeBERT](https://arxiv.org/abs/2002.08155).

Для научных/медицинских/юридических/финансовых документов выбрать задачу: entity extraction, relation extraction, document classification, search или matching. Кодовые embeddings оценивать на retrieval/clone detection с разбиением по repo/time; token similarity не доказывает functional equivalence.

**На бумаге:** сравнить generic vs domain tokenizer на терминах; выделить domain shift отдельно от смены labels и требований к ошибкам.

**Эксперимент:** generic task-trained checkpoint, domain checkpoint и DAPT; сопоставимый budget и general-domain regression.

**Готово:** pipeline проверяет нужную задачу и сохраняет provenance; снижение MLM loss или «медицинское» имя модели не используется как доказательство correctness.

### 33. Aspect sentiment, entity linking и структурированное понимание — по задаче

**Разобрать:** target/aspect-conditioned sentiment, span extraction + polarity, implicit aspects и conflicting opinions. Для entity linking: mention detection → candidate retrieval → context-based disambiguation → NIL/unknown. Совпадение surface form не определяет entity ID.

Обсудить intent + slots, taxonomy/entity typing и classification of relations between spans. Выбрать staged или joint formulation по метрикам upstream errors и объёму supervision.

**На бумаге:** «экран отличный, батарея слабая» — построить aspect spans и labels; неоднозначное имя — сформировать candidates и признаки disambiguation.

**Эксперимент:** evaluate gold mentions vs predicted mentions для linking; seen/unseen entities, alias shifts и NIL cases. Для aspect task считать extraction и polarity отдельно и joint.

**Готово:** прикладная система возвращает проверяемые structured fields с offsets/IDs; generic text sentiment head не подменяет aspect-dependent задачу.

### 34. OCR, layout, таблицы и границы text-only BERT — обзор с отдельной практикой

**Разобрать:** plain OCR tokens теряют 2D layout и visual features; document classification, key-value extraction и document QA могут требовать layout-aware model. LayoutLM соединяет text/layout signals в document understanding. [Первичная работа](https://arxiv.org/abs/1912.13318).

Для таблиц выделить row/column relations, cell coordinates и numerical reasoning; text serialization — один baseline, а не универсальное сохранение структуры. Text-only encoder representations могут входить в multimodal pipeline, но модель при этом не приобретает visual understanding автоматически.

**На бумаге:** две формы с одинаковым набором OCR words, но разным расположением полей. Показать, какие признаки отсутствуют у plain text serialization.

**Эксперимент:** OCR+text encoder baseline против layout-aware решения на фиксированных документах; отдельная оценка OCR error, span extraction и field correctness.

**Готово:** выбор modality обоснован; при необходимости продолжить в [Computer Vision](<../05-computer-vision/План — компьютерное зрение.md>).

<a id="h"></a>
## H. Оценка, надёжность и интерпретация

### 35. Метрики, benchmarks и честное сравнение — ядро

**Разобрать по задачам:** classification macro/micro F1 и PR-AUC; NER strict entity F1; QA EM/F1 и abstention; STS Spearman/Pearson; retrieval Recall@k/MRR/nDCG; latency/memory/стоимость отдельно от качества.

~~~text
MRR = mean_q (1 / rank_first_relevant(q)), либо 0 если не найден
DCG@k = Σ[i=1..k] (2^rel_i - 1) / log2(i+1)
nDCG@k = DCG@k / IDCG@k при IDCG>0
~~~

Указать выбранное определение relevance, cutoffs и обработку queries без judged relevant docs. Incomplete judgements не дают полную информацию о false negatives.

Benchmarks задают набор задач/языков и протокол, а не универсальный ranking для production. Изучить [BEIR](https://arxiv.org/abs/2104.08663), [MTEB](https://arxiv.org/abs/2210.07316) и multilingual [MMTEB](https://arxiv.org/abs/2502.13595); зафиксировать task versions и subset.

**На бумаге:** посчитать MRR/nDCG двух rankings и объяснить расхождение; выбрать единицу bootstrap: query/document/group, а не коррелированные token rows.

**Эксперимент:** paired comparisons, seeds, confidence intervals и error slices; equal hyperparameter-search budget.

**Готово:** вывод показывает effect size и ограничения; нет «лучшей модели» по одному среднему leaderboard score.

### 36. Калибровка, OOD, robustness и отказ — ядро для применения

**Разобрать:** probability calibration против ranking, temperature scaling, Brier/NLL, reliability bins, selective prediction и coverage-risk curves. Softmax confidence не является доказательством понимания или detector всех OOD cases.

Проверять temporal/domain shift, неизвестные labels, negation, typos, числа, quotations, OCR noise, code switching и чрезмерную длину. Counterfactual tests имеют expected behavior: harmless spelling change не должна менять решение без причины, а изменение отрицания иногда обязано менять его.

**На бумаге:** два classifiers с одинаковой accuracy и разной confidence; показать, как выбирается abstention threshold. Описать failure, при котором температурная калибровка не исправит ranking/error.

**Эксперимент:** calibration set отдельно от test; проверить selective risk и tail slices, а также baseline human-review policy. Для moderation/entity extraction указать стоимость обоих типов ошибок.

**Готово:** пороги согласованы с допустимыми ошибками и покрытием; unknown detection проверено на отдельном наборе, а не на confidence in-distribution.

### 37. Интерпретация, probes, приватность и устойчивость данных — углубление

**Разобрать:** gradient/saliency attribution, integrated-gradient-style methods, attention inspection, representation probes и controlled input interventions. Decodability признака из hidden state не доказывает его причинное использование моделью; attention weights не являются готовым объяснением решения.

Проверить shortcut learning: длина, источник, именованные сущности и шаблоны. Encoder classifier/retriever тоже получает adversarial input; специальные token strings и embedded instructions требуют task-specific tests, а не предположения об абсолютной защищённости.

Для PII extraction и retrieval определить data retention, ACL/filtering, logging и ошибки masking. PII head может пропустить сущность; удаление части именованных сущностей не доказывает полной anonymization.

**На бумаге:** предложить counterfactual и probe control для гипотезы «модель использует отрицание». Отделить association, attribution и causal evidence.

**Эксперимент:** perturbations, human audit и тест на spurious source metadata; проверка фильтра доступа до выдачи результатов.

**Готово:** объяснение подкрепляет diagnosis, а claim о privacy/robustness имеет конкретный scope и evidence.

<a id="i"></a>
## I. Производительность и deployment

### 38. Distillation, quantization и compression encoder — углубление после baseline

**Разобрать:** teacher logits/embeddings/intermediate states, temperature and KL direction, objective mixing и capacity student. Типичный soft-target KL включает T² при соответствующей temperature convention; scale нельзя копировать без определения loss.

DistilBERT — ранний пример encoder distillation, но task-specific student можно обучать под другой objective. Quantization weight/activation и pruning — разные изменения; sparse weights не ускоряют runtime без подходящих kernels. [DistilBERT](https://arxiv.org/abs/1910.01108).

**На бумаге:** выбрать supervision для маленького classifier и маленького retriever. Посчитать memory weights/gradients/optimizer/activations отдельно; не использовать одно «bytes per parameter» для всех режимов.

**Эксперимент:** compact task-trained model, distilled model и supported quantized runtime; quality by slice, calibration, latency и memory. Для retrieval измерить не только encoding, но и index footprint.

**Готово:** compression evaluated после export/reload; teacher data leakage и total training cost учтены.

### 39. Encoder inference: batching, export и сервис — ядро практического маршрута

**Разобрать:** tokenization overhead, dynamic batching/length bucketing, CPU/GPU, precision, supported attention kernels, ONNX/другие export backends и runtime contract. Поддержка конкретной архитектуры/head должна проверяться по runtime version.

В обычном bidirectional encoding нет последовательного autoregressive decode с KV cache той же семантики. Оптимизировать нужно workload целых texts/pairs и выбранных lengths. Для embeddings precompute/offline index и online query encoding измеряются раздельно; cross-encoder обслуживает pairs.

**На бумаге:** throughput versus p95 latency при batch policy; учесть очередь и padding длинного запроса. Определить versioned bundle: tokenizer, weights, head/pooling, normalization, thresholds и config.

**Эксперимент:** eager baseline → один export/runtime change; output/task-metric comparison, warm/cold runs, representative length histogram. API проверять на пустом/длинном/Unicode input.

**Готово:** измерен полный pipeline, а export artifact воспроизводим. Общий serving-маршрут — в [inference](<../06-llm/03-inference/План — инференс LLM.md>); encoder-specific details сохранять здесь.

### 40. Production monitoring, active learning и обновление индекса — углубление

**Разобрать:** data/label drift, class frequency, embedding distribution, retrieval relevance feedback и delayed labels. Confidence/OOD proxy требует валидации; drift в vectors не равен гарантированному ухудшению задачи.

Active learning: uncertainty/diversity sampling, human annotation и sampling bias. Добавление пользовательского feedback в training требует происхождения, review и held-out набора. Calibration/threshold changes также меняют систему без смены backbone.

При смене embedding model/tokenizer/pooling/normalization обычно требуется согласованное re-encoding или отдельно проверенный migration/compatibility protocol. Query vectors новой версии нельзя без оценки искать по старому embedding space.

**На бумаге:** план rollout нового encoder: offline eval → shadow/canary → rollback; определить, когда нужен полный reindex и как проверять document IDs/ACL.

**Эксперимент:** versioned dataset/index snapshots, incremental ingestion, regression suite и feedback audit; проверить удаление устаревших документов.

**Готово:** качество и cost отслеживаются по slices; модель, индекс и decision policy имеют согласованные версии и процедуру восстановления.

<a id="projects"></a>
## Маршруты под разные цели

**База encoder:** 1–6 → 8–13 → 15 → 35–36. Затем выбранная прикладная ветка. Пункты 7/9/14 изучать по вопросам objectives/model choice; не требуется обучать BERT с нуля.

**Классификация русского текста:** база → 16/21 → 30 → 38–40. Сначала классическая и frozen-feature модели, затем tuning; не выбирать ModernBERT только по дате релиза.

**NER и извлечение фактов:** база → 17 → 18/33 → 30/31 → 35–40. Начать с flat entities и корректного alignment, затем nested/relation/event задачи.

**QA по документам:** база → 19–20 → 26–27 → 31 → 35–40. Отдельно проверить retrieval и extractive QA на gold/retrieved context.

**Embeddings и semantic search:** база → 22–24 → 26–27 → 30/31 → 35–40. Затем sparse/late-interaction 28–29 и integration с RAG при необходимости.

**Исследовательский encoder-маршрут:** база → 6–9 → 14 → 22–24 → 28–29. Одна содержательная ablation с controlled compute/data полезнее поверхностного «прохождения всех моделей».

## Сквозные проекты и артефакты

### P1. Классификация обращений — минимальный практический проект

Russian intent/topic data, group/time split, TF-IDF+linear baseline, frozen task-appropriate vectors и tuned encoder. Один training intervention. Сохранить per-class errors, calibration и abstention policy. Итог: notebook/script, config, data manifest и отчёт о качестве/cost.

### P2. NER и structured extraction — следующая ступень

Char-span annotation → tokenizer alignment → token/span head → восстановленные spans. Сравнить rare types и boundaries; добавить relations только после оценки NER. Итог: annotation guideline, alignment examples, strict entity metrics и ошибки.

### P3. Поиск по документам — основная retrieval-практика

Corpus + held-out queries/qrels → BM25 → dense retrieval → reranker. Изменять один этап за раз; выбирать pooling/prefixes по checkpoint. Итог: versioned index, retrieval metrics, latency/memory и список queries, где методы расходятся.

### P4. Вопросы по русским документам — композиция задач

P3 → extractive QA → unanswerable handling с source offsets. Сравнить gold-context/retrieved-context regimes и несколько evidence positions. Генеративный RAG добавить как отдельную сравниваемую pipeline. Итог: answer correctness, retrieval failures и обоснованный reject policy.

### P5. Малый бюджет разметки и domain shift — углубление

Несколько few-shot samples → frozen embeddings/SetFit/task tuning → target-domain shift. При наличии corpus сравнить DAPT; total compute отдельно. Итог: learning curves по числу human labels, variance, ошибки new/rare labels и следующий annotation batch.

### P6. Компактный encoder service — системная ветка

Выбрать успешно проверенный P1/P3; compact/distilled/quantized alternative → export → нагрузочный test. Итог: versioned bundle, post-export metrics, p50/p95 latency, throughput, memory, rollback и index compatibility.

В теоретическом режиме для каждого проекта сначала достаточно спецификации входов/выходов, ручного примера, выбранных losses/метрик и разбора возможных ошибок. Это подтверждает теоретическую работу; запуск и численные результаты засчитываются после пользовательского выполнения.

## Контрольные задачи для углублённого понимания

1. Вывести shapes encoder слоя при B=2, L=7, D=24, H=4; объяснить разницу padding и causal mask.
2. Для vocabulary size 5 и двух MLM positions посчитать CE/gradient; показать loss на unchanged corruption target.
3. Согласовать classification head, labels и loss для single-label и multi-label задачи.
4. Превратить word BIO labels в subword targets и восстановить исходные char spans.
5. Выбрать валидный QA span при конфликтующих start/end maxima и unanswerable threshold.
6. Посчитать одну строку contrastive loss, затем повторить с false negative; объяснить изменение сигнала.
7. Сравнить mean-vector similarity, MaxSim и pairwise cross-encoder по вычислениям/storage.
8. Посчитать MRR/nDCG и показать, почему улучшение ANN recall не обязательно повышает relevance recall.
9. Обосновать encoder shortlist для русского short-text NER и длинного multilingual retrieval.
10. Спроектировать reindex/retraining procedure после смены tokenizer/pooling и указать критерии rollback.

## Как вести прогресс

Для каждого модуля фиксировать:

~~~text
Модуль и подтема:
Обсуждение: не начато | обзор | подробный разбор
Понимание: не проверено | частично подтверждено | проверено самостоятельным ответом
Практика: не выполнена | есть пользовательский артефакт | воспроизведена
Заметка / источник / revision:
Контрольный вопрос и мой ответ:
При эксперименте: data split, seeds, baseline, intervention, metrics, errors, cost
Следующая подтема:
~~~

**Критерий завершения выбранной ветки:** объяснены shapes/objective, решены соответствующие контрольные задачи, сделан вывод об ограничениях. Практическое освоение дополнительно требует воспроизводимого pipeline, честного baseline comparison и оценки ошибок/cost. Не нужно заканчивать все 40 модулей, чтобы получить полезную компетенцию в одной задаче.

**Откуда начать сейчас:** 1 → 2 → 3, точечно используя прежний конспект attention. После проверки 5–6 выбрать classification или embedding/retrieval ветку. Новый BERT-прогресс автоматически из прежних LLM-обсуждений не начисляется.
