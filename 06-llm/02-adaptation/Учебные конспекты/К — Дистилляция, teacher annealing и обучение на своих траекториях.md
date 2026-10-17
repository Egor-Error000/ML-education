# К. Дистилляция: чему student учится у teacher

[Оглавление](README.md). Пункт 12 плана; перенесённая тема старого inference A.5.

**Основа:** S4, строки 3445–4767. Сохранены вопросы: обязательно ли student меньше teacher; можно ли в начале обучения дать мягкий сигнал учителя и постепенно его ослабить; что можно сказать о training новых поколений моделей. **Исправления C13:** гипотезы о закрытых компаниях не факты, произвольные schedules/температуры не общие нормы, экономия total compute не следует из меньшего числа student steps. **Углубление:** KL, T², tokenizer compatibility, on-policy shift и честное сравнение стоимости.

## 1. Передаётся выбранный сигнал, а не «знание целиком»

Дистилляция обучает student имитировать выбранные outputs/representations teacher. Частая цель — компактная модель, но student может иметь тот же или больший размер. Термины обозначают источник и получателя supervision, а не обязательный порядок parameter counts.

Teacher может дать полное распределение logits, готовые ответы, embeddings, промежуточные hidden states или проверенные траектории действий. Каждый вариант переносит разные свойства и требует своего objective. При response distillation через API обычно доступны тексты, а не все vocabulary logits.

Soft targets полезны тем, что различают альтернативы. Например, hard target $\left(1,\,0,\,0\right)$ говорит «первый класс», а teacher $\left(0.7,\,0.25,\,0.05\right)$ дополнительно задаёт относительное сходство/неуверенность. Однако это **модельные** альтернативы: teacher может быть неправильно уверен. Их имитация не гарантирует правильное причинное понимание.

Квантизация меняет представление чисел той же модели; pruning удаляет структуру/веса; distillation обучает другую функцию. Их можно сочетать, но они не являются одной операцией с разными названиями.

## 2. Logit distillation и temperature

Пусть teacher logits z_T, student z_S определены над одними и теми же классами. При температуре T>0:

$$
\begin{gathered}p_T^{(T)}=\operatorname{softmax}\!\left(\frac{z_T}{T}\right),\qquad p_S^{(T)}=\operatorname{softmax}\!\left(\frac{z_S}{T}\right),\\\mathcal{L}_{\mathrm{soft}}=T^2D_{\mathrm{KL}}\!\left(p_T^{(T)}\,\Vert\,p_S^{(T)}\right).\end{gathered}
$$

Teacher фиксирован; KL направлен от teacher distribution к student. С точностью до константы по student это cross-entropy $-\sum_jp_{T,j}^{(T)}\log p_{S,j}^{(T)}$. Типичная смесь:

$$
\mathcal{L}=(1-\alpha)\mathcal{L}_{\mathrm{task}}+\alpha\mathcal{L}_{\mathrm{soft}},\qquad0\le\alpha\le1.
$$

Ориентация alpha зависит от convention: в другом тексте alpha может стоять у hard loss. Поэтому прежде чем обсуждать «ослабление alpha», нужно прочесть формулу.

T увеличивает мягкость: при больших T различия logits делятся на большее число. Например, logits $\left(2,\,0\right)$ дают p≈(0.8808,0.1192) при $T=1$ и $\left(0.7311,\,0.2689\right)$ при T=2. Но очень высокая температура может стереть полезные различия; её не выбирают только из желания «дать больше знаний».

Производная soft KL без T² по student logits имеет вид $\frac{p_S^{(T)}-p_T^{(T)}}{T}$. При больших T сама разность probabilities тоже часто масштаба 1/T, поэтому T² компенсирует типичное ослабление gradient. Это мотивация standard convention, а не обязательный множитель для любого distillation objective/normalization. [Классическая дистилляция](https://arxiv.org/abs/1503.02531).

### Ручной KL-пример

Teacher distribution $\left(0.8,\,0.2\right)$, student $\left(0.6,\,0.4\right)$, $T=1$:

$$
D_{\mathrm{KL}}=0.8\ln\frac{0.8}{0.6}+0.2\ln\frac{0.2}{0.4}\approx0.091516.
$$

Это не cross-entropy 0.591919: CE включает entropy teacher≈0.50040. По student logits gradient soft CE равен $\left(-0.2,\,+0.2\right)$. Совместный hard-target loss для первого класса имеет gradient $\left(-0.4,\,+0.4\right)$. Их смесь задаёт иной training signal; не следует путать величину KL и CE, хотя сами функции потерь отличаются на константу teacher entropy и имеют одинаковый gradient по student.

## 3. Совместимость и виды дистилляции

Для token-level KL должны совпадать смысл классов и conditioning. Если tokenizers разные, «координата 100» может быть другим токеном, а длины сегментации не совпадут. Прямой поэлементный KL такого вида некорректен без специального alignment/метода. Текстовые response targets допускают разные tokenizers: student токенизирует полученный текст своим способом.

Hidden-state distillation требует согласовать shapes. Для teacher hidden width 1024 и student 512 иногда используют проекцию/выбор слоёв. Просто вычесть tensors несовместимой ширины нельзя; даже одинаковая ширина не означает одинаковые координаты.

Response distillation обучает на teacher answers через SFT. Нужны разнообразие prompts, проверка answers и независимый holdout. Если teacher всегда пишет длинную вступительную фразу, student может скопировать её вместе с полезными свойствами. Filtering только по внешней «красоте» укрепляет style bias.

Reasoning/trajectory distillation использует доступные проверяемые промежуточные шаги/действия. Верный final answer не делает каждый шаг правильным. Для tools проверяются arguments, observations, permissions и достигнутое состояние; текст «я вызвал поиск» не равен записанному выполнению инструмента.

## 4. Teacher annealing: идея из вашего вопроса

Пусть student учится и на hard data, и на teacher. Можно задать убывающую $\alpha (t)$:

$$
\begin{aligned}\mathcal{L}_t&=(1-\alpha(t))\mathcal{L}_{\mathrm{hard}}+\alpha(t)\mathcal{L}_{\mathrm{teacher}},\\\alpha(t)&=\alpha_0\max\!\left(0,1-\frac{t}{T_{\mathrm{end}}}\right).\end{aligned}
$$

Это иллюстрация schedule. Она объясняет предложенный в чате механизм: сначала использовать дополнительные smooth targets, затем дать student больше свободы оптимизировать собственный objective. Ни первые «20–30%», ни $T=10$–20, ни $\alpha=0.9$ не являются универсальной нормой из этого рассуждения.

Почему это может помочь: teacher задаёт полезное распределение альтернатив и уменьшает сложность раннего поиска. Почему может мешать: teacher ошибается, ограничивает достижимую функцию, даёт слишком слабый/сильный сигнал или требует слишком дорогого forward. Больший student не обязан во всём превосходить меньшего teacher в начале; но ограничивать его teacher forever тоже не всегда разумно.

Гипотеза проверяется ablation: hard-only, постоянная alpha, убывающая alpha на одном data/evaluation протоколе. Помимо steps до заданной метрики считать total FLOPs/time teacher+student, generation/logits storage и фильтрацию. Если student стал сходиться вдвое быстрее, а teacher сделал десятикратно больше работы, итоговая экономия не установлена.

## 5. On-policy distillation и ошибка распределения

В обычном SFT student видит teacher prefixes. При generation собственная ранняя ошибка уводит его на состояния, которых в training почти не было. On-policy вариант собирает собственные student continuations и получает feedback teacher на этих состояниях. Это может уменьшить mismatch, но требует новых rollout/teacher расходов и careful labeling.

Teacher должен оценивать именно то продолжение/состояние, которое доступно student, без скрытой информации, которой не будет на inference. Нельзя train на объяснении с подсмотренным ответом и считать это честной demonstration обычного решения.

Self-distillation использует свою другую версию, ensemble или более ранний checkpoint как источник targets. Предыдущий checkpoint не автоматически «умнее текущего», а EMA target — не ground truth. Нужно определить, какой сигнал он стабилизирует и чем проверяется эффект.

На вопрос «так ли обучают конкретную новую модель компании» ответ должен опираться на опубликованный report. Из полезности teacher annealing не следует, что именно этот recipe использован в закрытом production. Чат содержал предположения, поэтому здесь они оставлены как исследовательские идеи, а не атрибуция компаниям.

## 6. Вопросы и практика

1. Обязан ли student быть меньше? **Ответ:** нет; размер — типичный мотив, а не определение.
2. Можно ли вычислять прямой vocab KL разных tokenizers? **Ответ:** без согласования class semantics/sequence alignment нельзя.
3. Доказана ли экономия, если student training занял меньше steps? **Ответ:** нет, нужно учесть teacher и весь pipeline.
4. Чем teacher answer отличается от AI preference? **Ответ:** первый даёт demonstration target, второй — оценку/сравнение; это разные supervision.

**Практика:** пересчитать KL-пример, затем заменить student на $\left(0.8,\,0.2\right)$ и получить нулевой KL. Спроектировать сравнение трёх alpha schedules с независимым holdout. Выписать отдельно teacher data generation cost, student training cost и inference cost успешного решения.

<!-- latex-display-v1 -->
