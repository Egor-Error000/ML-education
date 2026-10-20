# З. Регуляризация дискретных последовательностей и policy distributions

[Оглавление](README.md). Объединены пункты 9–10. **Основа:** S2 — embeddings/attention, S3 — SFT/RLHF; остальное отмечено как новое углубление. Полный RL-course находится в [lifecycle](../../06-llm/05-lifecycle/Учебные%20конспекты/README.md).

**Уровень:** основа branches NLP/RL; details can be skipped otherwise. **Предпосылки:** R1, R2; sequence masking/conditional probability; RL objective for RL portion. **Используют:** LLM adaptation/lifecycle.

## 1. Weight tying

Если input embedding matrix $E\in\mathbb R^{V\times d}$ и hidden dimension совпадает, logits могут использовать ту же matrix:

$$
z_t=E h_t+b,
\qquad h_t\in\mathbb R^d,\quad z_t\in\mathbb R^V.
$$

Для row-oriented batch запись эквивалентна $Z=HE^\top+b$. Число независимых parameters уменьшается примерно на $Vd$ относительно отдельного output matrix. Gradient в $E$ приходит и через input lookup, и через output projection. Это constraint на параметры, а не утверждение, что каждый embedding является готовым human-readable «значением слова».

Если dimensions различаются, требуется projection или другая architecture. Не все pretrained models используют tying. Веса нельзя просто связать после training без проверки поведения и contract checkpoint.

## 2. Шум для RNN, tokens и embeddings

Variational dropout для recurrent networks может использовать одну mask на все time steps sequence. Независимые masks каждый step вносят другой temporal noise и могут сильнее разрушать перенос информации. Нужно указать, какие connections маскируются: inputs, recurrent state или outputs.

Word/token dropout удаляет или заменяет tokens по объявленному protocol. `[MASK]` существует в BERT MLM, но не является универсальным meaningful token любой decoder model. MLM corruption — часть pretraining objective, а не произвольный preset на inference.

Noisy embeddings задаёт $\widetilde e=e+\xi$, где $\xi\sim\mathcal N(0,\sigma^2I)$. Scale относительно norm embeddings важен. Такой noise не создаёт дискретно корректную перефразировку и не заменяет проверку semantic invariance. Если target зависит от одной цифры/отрицания, сильный noise может повредить обучению.

## 3. Entropy attention и R-Drop

Для attention distribution $a$ entropy:

$$
H(a)=-\sum_j a_j\log a_j.
$$

Если минимизируемый loss содержит $-\lambda H(a)$, он поощряет более высокую entropy; term $+\lambda H(a)$, наоборот, поощряет концентрацию. Sign принципиален. Masked positions исключаются, $0\log0$ трактуется как 0. У causal tokens разное число доступных positions; сравнение raw entropy требует учитывать эту разницу.

Сосредоточенное attention не обязательно ошибка: точное копирование нужного token может требовать именно этого. Entropy prior не доказывает улучшение reasoning и не является direct measure объяснимости.

R-Drop делает два stochastic forward для одного input с разными dropout masks. Для выходных distributions $p^{(1)},p^{(2)}$:

$$
L=\frac{L_{\mathrm{CE}}^{(1)}+L_{\mathrm{CE}}^{(2)}}2
+\frac\alpha2\left[
D_{\mathrm{KL}}(p^{(1)}\Vert p^{(2)})+
D_{\mathrm{KL}}(p^{(2)}\Vert p^{(1)})\right].
$$

Это consistency penalty между stochastic predictions, не требующий отдельного EMA teacher. В sequence tasks masks и нормировка по active tokens должны совпадать. Два forward увеличивают compute; comparison учитывает budget. [R-Drop](https://arxiv.org/abs/2106.14448).

## 4. Sequence-level smoothing и decoding

Token-level smoothing независимо смягчает target в каждом шаге. Sequence-level objective может распределять массу по альтернативным допустимым целым sequences, например translations. Нужно определить, как генерируются alternatives, чем они взвешены и сохраняют ли смысл. Нельзя назвать любой штраф за длину sequence-level label smoothing.

Length penalty в beam search меняет ranking уже генерируемых кандидатов без gradient update. Training regularization меняет weights. Их effects могут пересекаться по длине outputs, но это разные вмешательства и разные места подбора hyperparameters.

## 5. Entropy, KL и PPO в RL

При maximization expected return entropy bonus имеет sign плюс:

$$
J=\mathbb E[R]+\beta H(\pi)
-\lambda D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}}).
$$

Entropy поощряет diversity/exploration относительно самой distribution. KL ограничивает отклонение от заданной reference, которая может быть далеко не uniform. Для $\pi=(0.9,0.1)$ entropy положительна; KL к такой же reference ноль. Они не взаимозаменяемы.

PPO clipping ограничивает incentive больших изменений probability ratios по собранным rollouts, но не является тем же KL penalty. Target network в value-based RL стабилизирует bootstrap targets через медленное обновление; её назначение тоже отличается от entropy regularization. Общий термин «стабилизация» не делает механизмы эквивалентными.

## 6. Контроль и практика

1. Где gradient weight tying? **Ответ:** shared matrix получает суммы вкладов input и output путей.
2. Какой sign повышает attention entropy при minimization? **Ответ:** отрицательный перед $\lambda H$.
3. Почему beam length penalty не training regularizer? **Ответ:** он меняет decoding ranking без изменения parameters.

**Практика:** выпишите shapes tying для $V=1000,d=64$. Вычислите symmetric KL двух distributions $(0.8,0.2)$ и $(0.6,0.4)$. Затем сравните entropy и KL к reference для двух policies; объясните, какой objective поддерживает разнообразие, а какой — близость к прежнему поведению.



**Схема курса:** same input/sequence → two stochastic forwards or policy rollout → consistency/KL or constrained update → evaluate task success, diversity and policy violations

<!-- source-depth-regularization-З -->

## 7. Weight tying: один параметр, два пути gradient

### 7.1. Shapes и реальное разделение параметров

При V=1000 и d=64 input embedding имеет 64 000 coefficients. Отдельная output matrix той же формы добавляет ещё 64 000. Tying оставляет одну общую матрицу с 64 000 независимыми coefficients, экономя 64 000; output bias, если он есть, считается отдельно. Экономятся также связанные состояния обучаемого параметра, но точный объём памяти зависит от optimizer и precision.

Равенство начальных чисел двух матриц не означает tying. Если это две независимые копии, их gradients и последующие updates могут разойтись. Нужно действительно разделяемое состояние параметра с совместимым checkpoint contract. Нельзя независимо заморозить input E и обучать ту же самую E как head без изменения постановки. Возможности tying и его влияние исследуются в [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859).

### 7.2. Как складываются вклады

В output path $z_k=E_k^\top h+b_k$ softmax CE даёт вклад в строку E_k, равный $(p_k-q_k)h^\top$. Он обычно ненулевой для многих vocabulary rows. Через input lookup gradient приходит к встреченным token rows, а его величина зависит от всего последующего Transformer. В shared E эти вклады суммируются; tying является ограничением общего обучения, а не просто сокращением файла weights.

Например, E имеет строки (1,0), (0,1), (1,1), hidden h=(2,1), bias=0. Logits равны (2,1,3), prediction примерно (0.24473,0.09003,0.66524). При target первого token output-path gradients строк равны $-0.75527h^\top$, $0.09003h^\top$ и $0.66524h^\top$. Если первая строка участвовала и во входе, её полный gradient включает ещё input-path contribution; приведённые три vectors не являются полным gradient всей сети.

Tying не делает nearest neighbors embedding готовыми синонимами и не доказывает хорошего обобщения. Оно уменьшает свободу независимой настройки input и output представлений. Если hidden dimension не совпадает с embedding dimension, нужна совместимая projection; просто транспонировать произвольную matrix недостаточно.

## 8. Шум в дискретных и непрерывных representations

### 8.1. Locked masks во времени

В recurrent постановках временно общая dropout mask может сохранять одно решение для feature на протяжении sequence. При q=0.5 и трёх шагах вероятность сохранить feature на всех шагах равна 0.5 для общей маски и 0.125 для трёх независимых. Ожидаемое число сохранённых позиций одинаково — 1.5, но возможность непрерывно передавать сигнал разная.

Надо указать, маскируются ли input features, recurrent connections или outputs. Термин variational dropout используется в нескольких связанных постановках и не заменяет конкретного описания. Один из теоретически мотивированных recurrent вариантов изложен в [работе Gal и Ghahramani](https://arxiv.org/abs/1512.05287). Независимый elementwise dropout в Transformer имеет другую структуру зависимостей.

### 8.2. Удаление token, replacement и MLM

Удалить token означает изменить длину и позиции последующих элементов. Заменить его UNK или существующим special token означает сохранить позицию, но изменить содержание и training distribution. Случайный обычный token может добавить ложное слово. Эти операции не равны Gaussian noise в embedding space и не должны называться одной безопасной «аугментацией текста».

В BERT MLM masked positions получают специальные targets исходных tokens согласно pretraining-протоколу. Для decoder SFT удаление отрицания из user instruction при сохранении прежнего ответа может создать неправильную пару. Наличие строкового `[MASK]` в тексте не гарантирует наличие meaningful special ID в tokenizer модели. Перед обучением проверяют сериализованные tokens, masks и target-позиции, как в курсе данных.

Для additive noise $\xi\sim N(0,\sigma^2I_d)$ имеем $E\|\xi\|^2=d\sigma^2$. При d=4 и sigma=0.1 это 0.04; квадратный корень среднего квадрата norm равен 0.2. Это не тождественно среднему norm. При фиксированном sigma рост dimension увеличивает суммарный noise; scale сопоставляют с нормами embeddings. Вектор между embeddings не обязан соответствовать допустимому слову или перефразировке.

## 9. Attention entropy и R-Drop

### 9.1. Знак, число доступных позиций и gradient

Entropy сосредоточенной строки a=(0.9,0.1) равна примерно 0.32508 nats, uniform по двум позициям — log2≈0.69315. Uniform по восьми имеет entropy log8≈2.07944. Поэтому рост raw entropy вдоль causal sequence может объясняться только увеличением числа разрешённых keys, а не изменением качества attention.

Иногда рассматривают H/log m для m>1, где m — число доступных позиций. При m=1 знаменатель нулевой: такую строку исключают или задают отдельное соглашение. Padding и causal-forbidden entries не включают в m как доступные alternatives. Равномерное attention также не всегда полезно: задача точного копирования может требовать сильной концентрации.

Для a=softmax(z) derivative entropy равна

$$
\frac{\partial H}{\partial z_j}=-a_j(\log a_j+H).
$$

У минимизируемого loss $-\lambda H$ gradient имеет противоположный знак: $\lambda a_j(\log a_j+H)$. Для (0.9,0.1) при lambda=1 это примерно (0.19775,−0.19775); gradient descent уменьшит более высокий logit и поднимет меньший. Положительный penalty +lambda H, напротив, способствует концентрации. Entropy prior не является доказательством объяснимости и не сообщает, на правильный ли token смотрит модель.

### 9.2. Полный R-Drop loss двух проходов

Для одного input два stochastic forwards дают p=(0.8,0.2) и q=(0.6,0.4), true class первый. CE равны −log0.8≈0.22314 и −log0.6≈0.51083, среднее≈0.36698. KL(p||q)≈0.09152, KL(q||p)≈0.10465; их полусумма≈0.09808. При alpha=2 в соглашении раздела 3 итоговый loss≈0.56315.

Супервизия требует правильного ответа, а symmetric consistency уменьшает расхождение stochastic predictions. Нужны обе части: два одинаково неправильных distributions имеют нулевой KL. Separate teacher не требуется, а gradient выбранного R-Drop objective может проходить через оба outputs. Если одну сторону detached, численное значение forward loss может совпасть, но gradient-процедура будет иной. Исходный метод описан в [R-Drop](https://arxiv.org/abs/2106.14448).

При dropout p=0 и отсутствии другой stochasticity оба forwards совпадают и KL=0. Это полезная проверка механизма, а не аргумент бесплатно делать два одинаковых прохода. R-Drop увеличивает объём вычислений; exact время и память зависят от batching, autograd и checkpointing. На обычном inference не требуется автоматически усреднять два stochastic outputs: это отдельное решение о decoding.

### 9.3. Sequence alignment и loss masks

В teacher-forced sequence training оба passes получают один и тот же context на каждой target-позиции. Сравниваются distributions одного следующего token при одинаковой истории; padding и excluded targets не должны попадать в token-average по ошибке. CE и KL reductions задают явно, иначе coefficient alpha меняет смысл с длиной ответа и размером vocabulary.

Если вместо этого независимо сгенерировать два полных ответа, на позиции t у них уже могут быть разные предыдущие tokens. Это разные conditional contexts, и прямое сопоставление distributions по одному индексу t не реализует прежний objective автоматически. Независимые histories требуют другого выравнивания или sequence-level критерия.

## 10. Распределение по sequences и decoding score

Sequence-level soft target Q(s|x) задаёт предпочтения целым допустимым ответам. Objective имеет вид $-\sum_sQ(s\mid x)\log P_\theta(s\mid x)$, где autoregressive log probability каждой sequence складывается по её собственной истории, включая объявленное правило EOS. Если alternatives различаются длиной, сумма token-loss и среднее по длине задают разные веса; это часть objective.

Две альтернативы перевода могут сохранять смысл, а две строки с разными числами или именами — нет. Происхождение, веса и проверка alternatives поэтому важнее одного названия smoothing. Нельзя независимо смешать marginal tokens разных ответов и считать, что получилось то же распределение корректных sequences.

Beam length penalty действует на ранжирование уже оцениваемых кандидатов. Например, для условного score logP/L кандидат с logP=−2 и L=2 получает −1, а с logP=−3 и L=4 — −0.75. Без нормировки первый имеет больше probability, с такой нормировкой предпочтут второй. Параметры модели не изменились; поменялось правило выбора. Это не training regularizer, хотя output lengths и итоговая метрика могут измениться.

## 11. RL: три механизма, которые нельзя смешивать

### 11.1. Entropy и KL предпочитают разные distributions

Для fixed action set uniform u имеем $KL(\pi\Vert u)=\log|A|-H(\pi)$. Поэтому entropy maximization эквивалентна приближению к uniform при этих условиях. Но reference policy обычно не uniform. Если все rewards одинаковы, entropy-only objective предпочитает uniform, а objective с penalty KL(pi||reference) — саму reference.

При reference=(0.9,0.1) её entropy≈0.32508 и KL к себе=0. Uniform имеет большую entropy≈0.69315, но KL(uniform||reference)≈0.51083. Поддерживать разнообразие и сохранять прежнюю неравномерную policy — разные цели. Sign проверяют с учётом того, максимизируется J или минимизируется −J.

В простом bandit objective $\sum_a\pi_aR_a-\lambda KL(\pi\Vert\rho)$ с положительной reference rho и lambda>0 стационарность с ограничением суммы probabilities даёт

$$
\pi_a^*=\frac{\rho_a\exp(R_a/\lambda)}
{\sum_b\rho_b\exp(R_b/\lambda)}.
$$

При равных rewards остаётся rho. При rho=(0.9,0.1), lambda=1 и rewards=(0,log9) получаем uniform: повышенная награда второго action компенсировала его меньшую reference probability. Это условная аналитическая модель одного состояния, не формула готового обучения всей LLM. Если reference запрещает action с нулевой probability, KL support требует отдельного внимания.

### 11.2. PPO clipping не является жёсткой границей новой policy

Пусть $r=\pi_\theta(a\mid s)/\pi_{old}(a\mid s)$ и advantage A получен по rollout. PPO максимизирует среднее

$$
\min\left(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A\right).
$$

При epsilon=0.2, A=2 и r=1.5 обычный term равен 3, clipped surrogate — 2.4. При A=2 и r=0.5 результат остаётся 1: ухудшение вероятности полезного действия не скрывается нижней clipping-границей. При A=−2 и r=0.5 берётся −1.6 вместо −1; при A=−2 и r=1.5 остаётся −3. Знак advantage определяет, какая сторона перестаёт давать дополнительное поощрение.

Это изменение surrogate incentive на собранных actions. Оно не проецирует все probabilities новой policy в разрешённый диапазон и не гарантирует ограничение KL на всех состояниях. Общие параметры, другие actions и дополнительные terms могут продолжать изменять ratio. Reference для KL и old policy для ratio имеют разные роли: reference может оставаться фиксированной SFT-моделью, old — обновляться при новом сборе rollout. Основной алгоритм приведён в [PPO paper](https://arxiv.org/abs/1707.06347), а полный trajectory-разбор находится в lifecycle-курсе.

### 11.3. Token masks, reward scale и target networks

В LLM policy entropy и KL обычно усредняют или суммируют по выбранным action tokens, отделяя их от user/tool context. Сумма per-token KL создаёт больший общий cost для более длинного ответа при том же среднем расхождении. Поэтому sum, token mean и sequence mean не взаимозаменяемы. Увеличение reward scale при прежнем KL coefficient меняет компромисс даже без изменения optimizer.

Target network value-based RL служит другой цели: вычисляет более медленно меняющийся bootstrap target $y=r+\gamma\max_aQ_{target}(s',a)$, который фиксируют при соответствующем regression update. При r=1, gamma=0.9 и target values (2,4) получаем 4.6. Если online values уже (10,20), подстановка их в target дала бы 19. Отдельное состояние ограничивает движение этой цели, а не напрямую повышает entropy policy.

EMA target network и EMA teacher похожи способом обновления весов, но отличаются назначением target: reward bootstrap против consistency prediction. Стабильность сама по себе не доказывает правильность обоих. Они требуют data protocol, update schedule и независимой оценки именно нужной задачи.

## 12. Контроль и практика

1. **Почему tied E получает gradients даже для token, которого не было во входе?** Его row может участвовать в output softmax, хотя input lookup этой row отсутствовал.
2. **Почему R-Drop не требует EMA teacher?** Согласуются два stochastic outputs самой модели на одном условии.
3. **Почему raw attention entropy нельзя сравнивать без числа доступных keys?** Максимальная entropy растёт как log m.
4. **Почему PPO ratio и reference KL нельзя считать одним ограничением?** Они относятся к разным distributions и по-разному воздействуют на objective.

Практика: выписать tying shapes и два gradient paths, проверить R-Drop toy loss и активные sequence masks. Затем вычислить entropy/reference KL и четыре случая PPO clipping, объяснив направление каждого стимула. Все policies и rewards главы учебные; реальное RL/SFT-обучение пользователя и результаты его модели здесь не заявляются.
