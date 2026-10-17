# Д. Оценка приложения: где именно пропадает правильный ответ

[Оглавление](README.md). Пункт5 и итоговый checkpoint плана.

**Основа:** S6:182–1373 о RAG, S5/S6 о tools и S3:820–918 об evaluation. **Углубление:** ручные ranking metrics, oracle context, различие correctness/groundedness/citation и стоимость успешного решения. Никакие числа ниже не объявляются результатом вашего запуска.

## 1. Независимый evaluation set

Нужны questions с user intent/constraints, допустимыми sources/versions, qrels для retrieval и rubric/answers для end-to-end. Для tool tasks — initial state, allowed actions, expected final state и oracle/checks. Train/eval разделяются по source/group/time согласно deployment task, а не только random rows.

Включить answerable/unanswerable, exact terminology/paraphrases, rare entities, misleading/stale documents, contradictory sources и permissions. Если проверить только простые FAQ с буквальным совпадением query/doc, получится узкая уверенность, а не доказательство general quality.

Gold answers тоже проверяются: несколько formulations могут быть допустимы, а source может обновиться. Eval snapshot/revision нужно сохранять, чтобы model regression не путать с изменением test ground truth.

## 2. Retrieval metrics

При наборе relevant documents $R_q$ и первых $k$ найденных $S_{q,k}$:

$$
\operatorname{Recall@}k=\frac{|S_{q,k}\cap R_q|}{|R_q|},\qquad
\operatorname{Precision@}k=\frac{|S_{q,k}\cap R_q|}{k}.
$$

Если qrels содержит два relevant chunks, а top3 нашёл один, recall $1/2$, precision $1/3$. Если нужна хотя бы одна достаточная passage, отдельная hit@k измеряет наличие такого passage; это не та же самая доля всех relevant chunks.

Reciprocal rank — $1/r$, где $r$ rank первого relevant item; при отсутствии — 0 по выбранному cutoff. MRR усредняет по queries. При ranks $(1,3,\infty)$ получаем:

$$
\operatorname{MRR}=\frac{1+\frac13+0}{3}=\frac49\approx0.4444.
$$

Для graded relevance используют DCG/nDCG. Объявим convention:

$$
\operatorname{DCG@}k=\sum_{i=1}^{k}\frac{2^{g_i}-1}{\log_2(i+1)},\qquad
\operatorname{nDCG@}k=\frac{\operatorname{DCG@}k}{\operatorname{IDCG@}k}.
$$

Для grades $(2,0,1)$ DCG $3+0+1/2=3.5$, ideal order $(2,1,0)$ даёт $3+1/\log_2 3\approx3.63093$, nDCG около $0.96394$. Если denominator нулевой, handling queries без relevant results определяется protocol.

Incomplete qrels не доказывает irrelevance каждого неразмеченного document. Human audit retrieval disagreements полезен, особенно при смене lexical/dense method.

## 3. Correctness, groundedness и citations

**Correctness:** отвечает ли result на вопрос верно. **Groundedness:** поддерживают ли переданные sources claims. **Citation correctness:** ведут ли ссылки к нужным passages и подтверждают ли конкретный claim. **Coverage/completeness:** не пропущены ли существенные части ответа.

Модель может дать правильный факт из памяти с неправильной citation; grounded answer по устаревшему документу может быть неверным для текущего вопроса; несколько citations могут не поддерживать главный вывод. Эти cases требуют разных labels, а не одного «есть ссылки».

Unknown/abstention оценивается с учётом answerability. Отказ без причины на answerable question — ошибка coverage; уверенный ответ без evidence на unanswerable — hallucination/unsupported claim. Нельзя улучшить factual score, просто отказываясь на всех сложных задачах, если denominator скрывает отказы.

## 4. Oracle context и разложение pipeline failures

Сравнить no retrieval, actual retrieved context и gold/oracle context при одинаковом generator/config. Если actual fail, oracle correct — evidence pipeline вероятный bottleneck. Если oracle тоже fail — причина может быть в instructions, reasoning, context interpretation или task specification.

Нужно проверить и context assembly: relevant document может быть в top20, но не попасть в final prompt из-за reranker, dedup или token truncation. Поэтому сохранять candidate IDs/ranks, selected snippets и фактический model input. Успех retriever по большому $k$ не доказывает, что generator увидел evidence.

Для tools отделять selected-tool accuracy, argument correctness, execution success и final task success. Валидный call к search может быть бесполезным; HTTP200 не гарантирует правильное изменение состояния; модель могла верно выполнить action, но неверно сообщить result.

## 5. Стоимость, latency и reproducibility

Считать total input/output tokens, retrieval/reranker/model/tool calls, retries и latency components. Для набора задач полезна стоимость успешного решения:

$$
C_{\mathrm{success}}=
\frac{\text{общая стоимость всех попыток}}{\text{число успешно решённых задач}}.
$$

Если 100 attempts стоили 20 условных units и 80 успешны, получается $0.25$ unit/success. Деление только стоимости successful attempts исключило бы wasted work и занизило реальную цену. При нуле successes ratio не определён/бесконечен по report convention, а не равен нулю.

Зафиксировать model/template/tokenizer, corpus/index revision, encoder/pooling, reranker, retrieval settings, prompt budget, decoding и tool stubs/environment. Не приписывать улучшение model, если одновременно расширили corpus и дали вдвое больше calls.

## 6. Judge и uncertainty

LLM judge получает rubric и sources, а names outputs скрыты; менять порядок для парных сравнений, допускаются ties. Проверять human subset и систематические disagreements по длине/стилю. Для объективных fields/state использовать deterministic oracle, если он корректен и достаточно полон.

Paired bootstrap по query groups оценивает uncertainty различия systems. Repeated seeds/decoding runs показывают stochastic variation, но не устраняют dataset bias. Один case useful для debugging, а не достаточное основание universal ranking.

## 7. Итоговая самопроверка

1. Чем hit@k отличается от recall@k по всем relevant documents? **Ответ:** hit — факт наличия хотя бы достаточного evidence, recall — доля relevant items.
2. Почему gold context полезен? **Ответ:** отделяет часть upstream failures от generation при доступном evidence.
3. Что доказывает schema validity? **Ответ:** только соответствующий структурный слой, не final task success.

**Практика:** собрать 12 учебных questions с qrels/answerability, три tool traces с ошибками и явную rubric. Вручную посчитать Recall/MRR/nDCG examples; затем спроектировать baseline→одно изменение→error review. Большой production project не требуется для теоретической проверки, но реальные metrics можно заявлять только после запуска.
