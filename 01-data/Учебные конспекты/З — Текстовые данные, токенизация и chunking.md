# З. Как превратить документы и диалоги в корректные примеры

[Оглавление](README.md). Пункт 8 плана. **Основа:** S2:1817–3404 — токены и обучение LLM; S3:228–339 — SFT на книгах и masks; S6:182–1373 — RAG, embeddings и хранение. Здесь объединены data-протоколы; детали моделей находятся в LLM-конспектах.

## 1. Текст до tokenizer

Unicode допускает визуально похожие, но разные последовательности symbols. Нормализация, переносы строк, HTML и invisible characters влияют на tokens. При этом «очистить всё» опасно: регистр, формулы, таблицы, код и знаки отрицания могут быть смыслом задачи. Для каждого transformation нужна цель и обратимость там, где она требуется.

BPE последовательно объединяет частые пары units по правилам обученного vocabulary. WordPiece тоже использует subwords, но его training/segmentation conventions отличаются; конкретный tokenizer определяет, как разбивать неизвестное слово. SentencePiece — toolkit, который поддерживает несколько algorithms, а не синоним одного BPE. Byte fallback и UNK зависят от модели; наличие одного нельзя предполагать по названию инструмента.

Tokenizer связан с model embeddings и special-token IDs. Нельзя заменить vocabulary «на более удобный» и ожидать, что pretrained weights сохранят смысл всех IDs. Сравнивать длины следует в tokens конкретной model, а не только в словах/characters. Один язык может требовать существенно больше tokens для того же содержания.

## 2. Dialog template и supervised target

Диалог содержит roles, boundaries, tool calls и turn-end/EOS markers. Chat template превращает эти структуры в последовательность. Ручное добавление уже автоматически вставляемых special tokens может их дублировать. Проверяется decoded sample и список IDs, а не только исходный JSON.

В SFT causal LM на позиции $t$ предсказывает следующий token. Если обучаются только assistant responses, prompt/user/tool observations остаются context, но исключаются из loss. Для mask $m_t\in\{0,1\}$:

$$
\mathcal L=-\frac{1}{\sum_t m_t}
\sum_t m_t\log p_\theta(x_{t+1}\mid x_{\le t}).
$$

Нулевой знаменатель означает пустой обучающий пример и требует обработки. Mask должна соответствовать реальному shifted target, а не случайно сдвинутой на одну позицию роли. Простая проверка — распечатать tokens с активным loss и убедиться, что это нужные outputs.

Truncation может удалить ответ, начало инструкции или закрывающий marker. Сохранение только последних tokens иногда теряет цель, а только первых — все targets. Packing нескольких examples требует продуманного attention/EOS protocol; границы между независимыми документами не должны незаметно превращаться в ложное продолжение. Выбор зависит от training setup и фиксируется явно.

## 3. Chunking для retrieval и обучения

Документ сначала получает source ID/version и split; затем нарезается. Соседние overlapping chunks в разных splits почти дублируют содержание. Разделение готовых chunks случайным образом — типичная leakage. Если цель — перенос на новые документы, их IDs должны быть disjoint.

Для длины документа $L$, chunk size $c$ и overlap $o<c$ шаг равен $s=c-o$. В простой sliding-window схеме, где последний chunk может быть короче:

$$
N=\begin{cases}
1,&0<L\le c,\\
1+\left\lceil\dfrac{L-c}{c-o}\right\rceil,&L>c.
\end{cases}
$$

При $L=1000,c=300,o=50$ получаем четыре chunks. В пустом документе policy отдельная. Реальные section-aware splitters могут дать другое число: formula относится к заданной схеме, а не ко всем библиотекам.

Малый chunk повышает локальную точность поиска, но может потерять условие в соседнем абзаце. Большой хранит контекст, но добавляет noise и занимает budget. Overlap снижает риск разрыва фразы, увеличивая storage и duplicate retrieval. Нужно сохранять source offsets, title/section, version и связи с родительским документом. Если retrieval ищет child, а в prompt возвращается parent, evaluate надо фактически переданное evidence.

## 4. Instruction/preference mix и contamination

Task mix определяет, какие behaviours чаще оптимизируются: извлечение, reasoning, отказ при отсутствии evidence, tool use. Больше examples одного простого типа не обязательно улучшает другой сложный. Learning curve строится при фиксированных evaluation tasks и понятном compute budget; важно различать рост числа уникальных sources и повторные epochs.

Preference pairs должны относиться к одному task/context, иметь rubric и provenance. Если teacher видел benchmark solutions, synthetic examples могут принести contamination даже при непохожем тексте. Exact dedup недостаточен; сохраняются parent sources и способы generation.

Сравнение chunk sizes проводится на фиксированных questions/relevance labels: retrieval recall/ranking, answer correctness, evidence attribution, tokens и latency. Подбор размера по test answers превращает test в development. В чатах вопрос о multi-vector storage показал, почему кроме quality важен объём индекса; изменение chunking меняет обе стороны компромисса.

## 5. Контроль и практика

1. Почему SentencePiece не равен BPE? **Ответ:** это toolkit с несколькими алгоритмами и настройками.
2. Почему сначала split документов, потом chunks? **Ответ:** иначе связанное и перекрывающееся содержание попадает в train/test одновременно.
3. Что проверять перед SFT? **Ответ:** template, token IDs, shift, loss masks, truncation и наличие активных targets в batch.

**Практика:** токенизируйте три языка и фрагмент кода выбранной моделью; сравните lengths и UNK/byte fallback. Для одного документа вручную постройте chunks двух размеров с metadata. На одном диалоге пометьте каждый loss target и проверьте, что tool observation не обучается как собственный assistant output.
