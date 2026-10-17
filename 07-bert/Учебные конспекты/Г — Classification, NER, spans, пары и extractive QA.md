# Г. Прикладные головы encoder: семь разных постановок

[Оглавление](README.md). Модули **15–21**. **Происхождение:** новое раскрытие BERT-плана; общие structured outputs из чатов не означают обучение этих heads. Один backbone может обслуживать разные задачи, но labels, objective и decoding должны совпадать с постановкой.

## 1. Classification: тема, intent, sentiment и moderation

Pooled vector $z$ подаётся в linear/MLP head, logits $Wz+b$ обучаются CE для mutually exclusive labels. Sentiment — оценка отношения, topic — тема, stance — позиция относительно конкретного утверждения. «В статье сказано: товар ужасный» содержит negative quotation, но позиция автора зависит от контекста. Moderation дополнительно зависит от annotation policy, адресата и цитирования.

Tiny counterfactual pairs помогают проверить смысл: «сервис понравился»/«сервис не понравился» должны различаться для sentiment; typo без смены смысла не обязан менять label. Ошибка guideline отличается от model error. Для ordinal classes стоит оценить ordinal objective/стоимость дальнего промаха, а не считать все classes unrelated.

Baseline TF-IDF+linear, frozen vectors+head и tuning сравниваются на group/time split. Macro-F1, per-class recall, confusion matrix, calibration и truncation slices дополняют accuracy. Long-text topic head не исправит отсутствие evidence, отрезанного tokenizer.

## 2. Multi-label, hierarchy и selective prediction

При независимых labels sigmoid/BCE выдаёт probability каждого события. Global threshold проще, per-label thresholds позволяют учитывать prevalence/cost, но требуют достаточно validation data. Missing labels mask-ятся, а не превращаются автоматически в negatives.

Toy truth двух examples: $(1,0,1)$ и $(0,1,0)$. Scores $(0.8,0.4,0.3)$ и $(0.2,0.6,0.1)$ при threshold 0.5 дают TP=2, FP=0, FN=1. Micro-F1=0.8, macro-F1=$2/3$: редкий третий label полностью пропущен. Threshold 0.25 только для третьего label исправляет именно этот toy set, но выбирать его по итоговому test нельзя.

Hierarchy child→parent может требовать constrained decoding, conditional probabilities или penalty на inconsistencies. Таксономия должна действительно задавать включение; «связанные labels» не всегда parent-child. Изменение taxonomy version меняет semantics output и требует migration.

Abstention откладывает решение при недостаточной надёжности. Report содержит coverage и risk на accepted subset. Система, отвечающая только на одном лёгком примере, не становится общей 100%-accurate моделью. Confidence threshold выбирается на calibration/validation и проверяется на relevant OOD cases.

## 3. NER/POS/slots: word, subword и char span

Token projection даёт logits $B\times L\times C$. В BIO B-TYPE начинает entity, I-TYPE продолжает, O вне entities; BILOU отдельно обозначает single-token и boundaries. Word labels переносятся на pieces по выбранной policy: supervision first piece only либо coherent labels всех pieces. При втором варианте continuation исходного B-PER обычно становится I-PER, а не начинает новую entity.

Для «Иванова-Петрова» token strings могут быть несколькими pieces. `word_ids` и offsets связывают их с исходным text; special/padding labels ignored. Нельзя восстанавливать entities простым concat `##` strings, теряя исходные пробелы/Unicode. [Alignment guide](https://huggingface.co/docs/transformers/main/tasks/token_classification).

Entity-level strict F1 считает correct entity только при совпадении type и boundaries. Если из четырёх gold entities предсказаны четыре, но одна имеет wrong boundary, TP=3, FP=1, FN=1, F1=0.75. Token accuracy может быть высокой из-за множества O. POS обычно оценивает word labels; slot filling дополнительно требует целостности нужных slots в task.

CRF добавляет sequence score emissions+transitions:

$$
P(y\mid x)=\frac{\exp\left(\sum_t e_t(y_t)+\sum_t A_{y_{t-1},y_t}\right)}{Z(x)}.
$$

Forward algorithm считает normalizer, Viterbi — лучшую path, обычно cost порядка $LC^2$. Constraints могут запретить незаконные BIO transitions, но правильная syntax labels не гарантирует semantic entity correctness. CRF стоит добавлять по diagnosis, а не считать обязательной головой BERT.

## 4. Nested entities, relations, events и coreference

Flat BIO даёт один label на token и не выражает произвольные nested/overlapping spans. Span classifier рассматривает candidates $(i,j)$ и representation endpoints/pooling, затем type/NA. Всего contiguous spans $L(L+1)/2$; при maximum width $M\le L$ их $ML-M(M-1)/2$. Для $L=5,M=2$ — девять вместо пятнадцати.

Пример «Университет города Москвы» может содержать organization целиком и location внутри по guideline. Один flat sequence не кодирует оба произвольных overlapping entities одновременно. Candidate pruning уменьшает compute, но вводит recall ceiling: удалённый gold span head уже не вернёт.

Relation extraction получает pair entity representations и context; negatives включают NA и трудные похожие pairs. Gold-entity evaluation показывает head quality условно на идеальном NER; predicted-entity pipeline включает upstream losses. Фраза «Анна не работает в компании Б» не подтверждает employment relation только по совместному упоминанию.

Event extraction разделяет trigger, arguments и roles: «Компания купила завод» требует buyer/asset и event type. Coreference связывает mentions одного referent через antecedent scoring и document context; одинаковая surface string не гарантирует один entity. Staged против joint models сравниваются по total outcome и supervision budget, с split по documents/entities.

## 5. NLI, paraphrase, STS и relevance

Pair encoding позволяет tokens двух texts взаимодействовать в attention. Premise «Анна живёт в Москве», hypothesis «Анна живёт в России» может быть entailment при заданных знаниях; reverse direction не обязательно entailment. «Анна не живёт в Москве» lexical близко, но contradiction. Общая тема без ответа query не означает relevance.

NLI three-class labels, paraphrase equivalence, STS continuous similarity и retrieval relevance различаются. Для STS выбирают scale, MSE/Huber и correlations; для duplicate detection — identity policy и errors ложного merge. Нельзя брать любой NLI classifier и без проверки называть score universal similarity.

Cross-encoder sees pair jointly, bi-encoder encodes independently и сравнивает vectors. Truncation second segment может удалить нужный passage. Сравнение проводится на same held-out queries/pairs и budget; candidate order/asymmetry учитываются.

## 6. Extractive QA: start/end и ограничения

Question+context encoder получает two scalar projections:

$$
s_i=w_s^\top h_i+b_s,\qquad e_j=w_e^\top h_j+b_e,
\qquad L=\operatorname{CE}(s,i^*)+\operatorname{CE}(e,j^*).
$$

Некоторые implementations усредняют two terms; scale convention проверяют. Decoding максимизирует $s_i+e_j$ при valid context positions, $i\le j$ и maximum answer length. Mask question/SEP/padding задаётся явно. [QA offsets/windowing](https://huggingface.co/docs/transformers/main/tasks/question_answering).

Для three context tokens start $(0,5,2)$, end $(4,0,2)$ independent maxima дают invalid start=2,end=1. При max length 2 лучший valid span (2,3), score 7; span (1,1) score 4. Argmax двух heads независимо не решает constrained span task.

Char offsets half-open позволяют вернуть исходный substring. В «Анна в Москве» «Москве» занимает [7,13). Нормализация/tokenization не должна потерять mapping original characters. При overlapping windows gold answer может быть целиком только в одном; labels других windows должны соответствовать no-answer/window protocol, а не указывать обрезанный fragment как full truth.

Unanswerable handling сравнивает null score с лучшим valid span по calibrated rule. Например, null=8 против best=7 даёт difference 1; выберется null при threshold 0 в такой convention. Смена sign/threshold меняет decision. Несколько gold spans, boundary windows и cross-window score aggregation требуют explicit rules; они не решаются автоматически одним head.

Оцениваются EM/token-F1, no-answer precision/recall и source correctness. Retrieval+QA отдельно проверяется на gold и retrieved context: плохой retrieval не следует смешивать с head failure на достаточном evidence. Extractive model выбирает fragment, а не доказывает, что в corpus существовал ответ.

## 7. Few-shot и zero-shot

NLI zero-shot переводит candidate label в hypothesis template. Его phrasing влияет на entailment score; mutually-exclusive и multi-label decisions имеют разные normalization/threshold rules. Zero-shot относится к новой task label set, а не отсутствию pretraining/NLI supervision вообще.

MLM verbalizer сопоставляет class словам в MASK positions. Multi-token label не равна одному vocabulary logit; нужно определить aggregation и prompt calibration. Это специализированный classifier recipe, а не general chat interface.

[SetFit](https://arxiv.org/abs/2209.11055) адаптирует sentence representations по pairs и затем обучает classifier при малом количестве labels. Тысячи pairs из 16 texts не создают тысячи independent исходных examples. Нужны несколько few-shot samples/seeds, frozen baseline и ordinary tuning с comparable budget.

## 8. Контроль и практика модулей 15–21

1. Почему high micro-F1 скрывает rare label? **Ответ:** суммарные counts доминирующих labels могут перевесить отсутствие редкого.
2. Почему strict NER F1 ниже token accuracy? **Ответ:** entity требует одновременно правильных boundaries/type, а O tokens многочисленны.
3. Что не выражается flat BIO? **Ответ:** произвольные nested/overlapping entities при одном label на token.
4. Может ли NLI entailment score заменить relevance без оценки? **Ответ:** нет; это different supervision.
5. Почему separate start/end argmax ошибочен? **Ответ:** нарушает constraints порядка/длины/context positions.

**Практика:** выполните multi-label confusion counts, разметьте nested entities и relation с negation, восстановите char spans. Выберите valid QA span и null decision в примере. Для проектов classification/NER/QA определите baseline, annotation guideline, metric и один controlled intervention; различайте specification и фактический run.
