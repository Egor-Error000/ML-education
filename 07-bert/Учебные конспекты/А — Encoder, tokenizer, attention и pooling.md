# А. От текста до контекстных состояний BERT

[Оглавление](README.md). Модули **1–5** плана. **Происхождение:** BERT — новая ветка по плану. Предпосылка из чатов S2:1817–3404 — токены/Transformer; подробный QKV/backward уже есть в [основах LLM](../../06-llm/01-foundations/Учебные%20конспекты/README.md). Здесь объясняются именно bidirectional encoding и interfaces задач.

## 1. Что возвращает encoder

Encoder преобразует уже данный текст в contextual token states. Он может видеть разрешённые tokens с обеих сторон позиции. Decoder-only causal model ограничивает attention прошлым; encoder-decoder добавляет отдельное encoding входа и генерацию output. Architecture mask и training objective нужно рассматривать вместе.

Для batch size $B$, length $L$, hidden width $D$ encoder выдаёт $H\in\mathbb R^{B\times L\times D}$. Дальнейший смысл задаёт head: тема письма — logits $B\times C$; независимые moderation labels — тоже $B\times C$, но с другим loss; NER — $B\times L\times C$; extractive QA — два массива $B\times L$; retrieval — vectors $B\times d_e$. Если нужен новый произвольный ответный текст, обычный MLM BERT не становится chat generator от одного prompt.

Сначала выбирается prediction unit и baseline: правила/TF-IDF, frozen encoder+head, затем tuning. Advantage encoder зависит от task/data/cost, а не от универсального превосходства над всеми generative systems.

## 2. Tokenizer и связь с исходными characters

У классического BERT WordPiece vocabulary; cased/uncased checkpoints имеют разные normalization contracts. Другие encoders могут использовать BPE или SentencePiece. Совпадение vocabulary size не делает token IDs семантически совместимыми: row 100 одного checkpoint может обозначать другой token.

Для пары классическая layout-схема: CLS, tokens A, SEP, tokens B, SEP. Input IDs определяют lookup, attention mask отмечает valid positions, token-type IDs различают segments A/B. Наличие token-type embeddings не universal: config конкретной модели определяет fields. SEP не запрещает cross-segment attention.

Рассмотрим условное, не реальное WordPiece-разбиение: «Анна не Иванова-Петрова» → «Анна», «не», «Иван», «##ова», «-», «Петр», «##ова». Для NER нужно знать word IDs и char offsets исходной фразы; continuation pieces не являются отдельными новыми entities. Normalization может изменить lowercase/ё/е/омографы и offsets mapping. Поэтому восстанавливать spans по длинам token strings ненадёжно: используются offsets tokenizer по его contract.

До training измеряются tokens-per-word, UNK/fragmentation, truncation rate и language slices. Truncation выбрасывает информацию; увеличение `max_length` не исправляет потерянный конец текста, если модель/positions не поддерживают его.

## 3. Входы и один block: согласованные shapes

Классическая embedding layer складывает token, learned position и token-type vectors:

$$
X_{b,t}=\operatorname{Dropout}\!\left(
\operatorname{LN}(E_{x_{b,t}}+P_t+S_{s_{b,t}})\right).
$$

Для row-oriented tokens проекции $W_Q,W_K,W_V\in\mathbb R^{D\times D}$ дают tensors $B\times L\times D$, после reshape — $B\times H_a\times L\times d_h$, где $D=H_a d_h$. Одна attention head:

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\qquad O=AV.
$$

Softmax по key axis, padding keys имеют additive mask $-\infty$; finite approximation зависит от dtype/backend. Не нужно маскировать valid будущие keys causal mask, если задача — обычный bidirectional encoder. Полностью masked row требует корректной обработки, иначе softmax может дать NaN.

После concat heads и $W_O$ исходный post-LN block имеет residual+LN после attention, затем FFN и снова residual+LN. FFN применяет одну и ту же нелинейную двухслойную функцию отдельно к token positions. В pre-LN нормализация стоит перед sublayer, меняя путь gradients; это architecture change, не перестановка без последствий.

Для $B=2,L=5,D=12,H_a=3$ head dimension 4; scores shape $2\times3\times5\times5$, после concat output $2\times5\times12$. Если все four projection matrices имеют biases, parameters:

$$
N_{QKVO}=4(D^2+D)=624.
$$

При $D_{ff}=48$ FFN с two biases содержит $12\cdot48+48+48\cdot12+12=1212$. Two LayerNorm с scale/bias добавляют $4D=48$. Это toy block count без embedding tables и других architecture choices.

Исторические base/large configs — depth 12/24, widths 768/1024, heads 12/16 соответственно; это конкретные версии, не определение всех encoders. [Официальные BERT configs](https://github.com/google-research/bert).

## 4. Mask и стоимость

Dense attention score storage порядка $BH_aL^2$, attention matmul work — $BL^2D$; projections/FFN при proportional width — $BLD^2$. При переходе 128→512 length linear parts растут в 4 раза, quadratic — в 16. Actual memory также включает weights, saved activations, workspace и backend.

FlashAttention-подобные algorithms меняют порядок обработки и memory traffic, позволяя не materialize все scores одновременно, но dense attention математически остаётся полной. Один long sample при padding до batch maximum увеличивает work всех shorter samples. Dynamic padding и bucketing помогают, если runtime действительно использует уменьшенные lengths.

Padding query states не обязательно нули: даже при masked keys valid states могут агрегироваться в padded queries. Для pooling и token loss padding исключается своими masks. Одна attention mask не означает автоматическое правильное masking каждой последующей операции.

## 5. CLS, pooler и masked pooling

Raw CLS — token state $h_0$. Классический `pooler_output` — learned affine transform CLS с tanh, а не идентичный vector. `last_hidden_state` содержит все positions. [Interfaces outputs](https://huggingface.co/docs/transformers/main/model_doc/bert).

Masked mean с выбранным set valid/special-token positions:

$$
z=\frac{\sum_t m_th_t}{\sum_t m_t}.
$$

Для rows $(1,0,0),(0,2,0),(0,0,3),(9,9,9)$ и mask $(1,1,1,0)$ mean $(1/3,2/3,1)$. Если padding ошибочно включён, mean $(2.5,2.75,3)$ и direction меняется. Padding здесь намеренно nonzero: это проверяет реальную необходимость mask. Empty denominator обрабатывается отдельно.

Masked max также исключает padding, но выбор нулей вместо $-\infty$ может ошибочно победить отрицательные valid coordinates. Layer selection/aggregation, projection и normalization — versioned components. MLM pretraining не гарантирует sentence cosine geometry: task-trained embedding models рассматриваются в Е.

## 6. Контроль и практика модулей 1–5

1. Можно ли видеть обе части pair через SEP? **Ответ:** да; SEP boundary marker сам по себе не attention barrier.
2. Почему pooler не равен CLS? **Ответ:** есть дополнительные learned transform и tanh.
3. Что растёт в 16 раз при length 128→512? **Ответ:** quadratic attention terms, при fixed прочих dimensions.
4. Почему token loss mask отдельна от attention mask? **Ответ:** первая выбирает supervised targets, вторая — допустимые attention interactions.

**Практика:** для пяти задач запишите inputs/labels/output shapes. Повторите block shapes при $B=2,L=7,D=24,H_a=4$; посчитайте projections/FFN. Вручную восстановите NER char span из условных pieces и вычислите masked mean. Для реального checkpoint проверьте tokenizer/config/train-eval outputs, не объявляя toy calculations результатом training.
