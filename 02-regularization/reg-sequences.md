# Регуляризация последовательностей, NLP и RL

**Уровень:** углубление.

**Предпосылки:** [Обобщение, диагностика и границы теории](<reg-diagnosis.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Чем различаются шум представлений, entropy и ограничение обновления политики?





## Карта вычисления и границы этапов

![Регуляризация последовательностей, NLP и RL: входы, преобразования и выходы](<../оформление/course/reg-sequences.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:b7804212c23b450144 -->
<a id="u-b7804212c23b450144"></a>

## Weight tying и общие параметры

<!-- /unit:b7804212c23b450144 -->

<!-- unit:e0ede342977597cee1 -->
<a id="u-e0ede342977597cee1"></a>

Если input embedding matrix $E\in\mathbb R^{V\times d}$ и hidden dimension совпадает, logits могут использовать ту же matrix:

<!-- /unit:e0ede342977597cee1 -->

<!-- unit:c425868541a9860b46 -->
<a id="u-c425868541a9860b46"></a>

$$
z_t=E h_t+b,
\qquad h_t\in\mathbb R^d,\quad z_t\in\mathbb R^V.
$$

<!-- /unit:c425868541a9860b46 -->

<!-- unit:2b7b67e0d94de31009 -->
<a id="u-2b7b67e0d94de31009"></a>

Для row-oriented batch запись эквивалентна $Z=HE^\top+b$. Число независимых parameters уменьшается примерно на $Vd$ относительно отдельного output matrix. Gradient в $E$ приходит и через input lookup, и через output projection. Это constraint на параметры, а не утверждение, что каждый embedding является готовым human-readable «значением слова».

<!-- /unit:2b7b67e0d94de31009 -->

<!-- unit:853da017c1de0c61c5 -->
<a id="u-853da017c1de0c61c5"></a>

Если dimensions различаются, требуется projection или другая architecture. Не все pretrained models используют tying. Веса нельзя просто связать после training без проверки поведения и contract checkpoint.

<!-- /unit:853da017c1de0c61c5 -->

<!-- unit:885d13acf16f2896a5 -->
<a id="u-885d13acf16f2896a5"></a>

### Shapes и реальное разделение параметров

<!-- /unit:885d13acf16f2896a5 -->

<!-- unit:cc11cd29a9e2318f50 -->
<a id="u-cc11cd29a9e2318f50"></a>

При V=1000 и d=64 input embedding имеет 64 000 coefficients. Отдельная output matrix той же формы добавляет ещё 64 000. Tying оставляет одну общую матрицу с 64 000 независимыми coefficients, экономя 64 000; output bias, если он есть, считается отдельно. Экономятся также связанные состояния обучаемого параметра, но точный объём памяти зависит от optimizer и precision.

<!-- /unit:cc11cd29a9e2318f50 -->

<!-- unit:2b142bf309884910a5 -->
<a id="u-2b142bf309884910a5"></a>

Равенство начальных чисел двух матриц не означает tying. Если это две независимые копии, их gradients и последующие updates могут разойтись. Нужно действительно разделяемое состояние параметра с совместимым checkpoint contract. Нельзя независимо заморозить input E и обучать ту же самую E как head без изменения постановки. Возможности tying и его влияние исследуются в [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859).

<!-- /unit:2b142bf309884910a5 -->

<!-- unit:31aadd2d9d0def3b76 -->
<a id="u-31aadd2d9d0def3b76"></a>

### Как складываются вклады

<!-- /unit:31aadd2d9d0def3b76 -->

<!-- unit:c7b6e3e9eb94bafaf9 -->
<a id="u-c7b6e3e9eb94bafaf9"></a>

В output path $z_k=E_k^\top h+b_k$ softmax CE даёт вклад в строку E_k, равный $(p_k-q_k)h^\top$. Он обычно ненулевой для многих vocabulary rows. Через input lookup gradient приходит к встреченным token rows, а его величина зависит от всего последующего Transformer. В shared E эти вклады суммируются; tying является ограничением общего обучения, а не просто сокращением файла weights.

<!-- /unit:c7b6e3e9eb94bafaf9 -->

<!-- unit:8adc3311db9f27b913 -->
<a id="u-8adc3311db9f27b913"></a>

Например, E имеет строки (1,0), (0,1), (1,1), hidden h=(2,1), bias=0. Logits равны (2,1,3), prediction примерно (0.24473,0.09003,0.66524). При target первого token output-path gradients строк равны $-0.75527h^\top$, $0.09003h^\top$ и $0.66524h^\top$. Если первая строка участвовала и во входе, её полный gradient включает ещё input-path contribution; приведённые три vectors не являются полным gradient всей сети.

<!-- /unit:8adc3311db9f27b913 -->

<!-- unit:f6ae511639b7b8a758 -->
<a id="u-f6ae511639b7b8a758"></a>

Tying не делает nearest neighbors embedding готовыми синонимами и не доказывает хорошего обобщения. Оно уменьшает свободу независимой настройки input и output представлений. Если hidden dimension не совпадает с embedding dimension, нужна совместимая projection; просто транспонировать произвольную matrix недостаточно.

<!-- /unit:f6ae511639b7b8a758 -->

<!-- unit:0ce45a19d277135d32 -->
<a id="u-0ce45a19d277135d32"></a>

## Шум последовательностей и embeddings

<!-- /unit:0ce45a19d277135d32 -->

<!-- unit:e9d20627db03feb6d8 -->
<a id="u-e9d20627db03feb6d8"></a>

Variational dropout для recurrent networks может использовать одну mask на все time steps sequence. Независимые masks каждый step вносят другой temporal noise и могут сильнее разрушать перенос информации. Нужно указать, какие connections маскируются: inputs, recurrent state или outputs.

<!-- /unit:e9d20627db03feb6d8 -->

<!-- unit:5d8e7f32a48368d414 -->
<a id="u-5d8e7f32a48368d414"></a>

Word/token dropout удаляет или заменяет tokens по объявленному protocol. `[MASK]` существует в BERT MLM, но не является универсальным meaningful token любой decoder model. MLM corruption — часть pretraining objective, а не произвольный preset на inference.

<!-- /unit:5d8e7f32a48368d414 -->

<!-- unit:4e8c616d5504e53a29 -->
<a id="u-4e8c616d5504e53a29"></a>

Noisy embeddings задаёт $\widetilde e=e+\xi$, где $\xi\sim\mathcal N(0,\sigma^2I)$. Scale относительно norm embeddings важен. Такой noise не создаёт дискретно корректную перефразировку и не заменяет проверку semantic invariance. Если target зависит от одной цифры/отрицания, сильный noise может повредить обучению.

<!-- /unit:4e8c616d5504e53a29 -->

<!-- unit:75a8116bcb09aab167 -->
<a id="u-75a8116bcb09aab167"></a>

### Locked masks во времени

<!-- /unit:75a8116bcb09aab167 -->

<!-- unit:3e7887f200c27ee024 -->
<a id="u-3e7887f200c27ee024"></a>

В recurrent постановках временно общая dropout mask может сохранять одно решение для feature на протяжении sequence. При q=0.5 и трёх шагах вероятность сохранить feature на всех шагах равна 0.5 для общей маски и 0.125 для трёх независимых. Ожидаемое число сохранённых позиций одинаково — 1.5, но возможность непрерывно передавать сигнал разная.

<!-- /unit:3e7887f200c27ee024 -->

<!-- unit:92a4c80d9f33a2fba8 -->
<a id="u-92a4c80d9f33a2fba8"></a>

Надо указать, маскируются ли input features, recurrent connections или outputs. Термин variational dropout используется в нескольких связанных постановках и не заменяет конкретного описания. Один из теоретически мотивированных recurrent вариантов изложен в [работе Gal и Ghahramani](https://arxiv.org/abs/1512.05287). Независимый elementwise dropout в Transformer имеет другую структуру зависимостей.

<!-- /unit:92a4c80d9f33a2fba8 -->

<!-- unit:35ecea0237ec030ec6 -->
<a id="u-35ecea0237ec030ec6"></a>

### Удаление token, replacement и MLM

<!-- /unit:35ecea0237ec030ec6 -->

<!-- unit:ad0f202eb8ae3dbafc -->
<a id="u-ad0f202eb8ae3dbafc"></a>

Удалить token означает изменить длину и позиции последующих элементов. Заменить его UNK или существующим special token означает сохранить позицию, но изменить содержание и training distribution. Случайный обычный token может добавить ложное слово. Эти операции не равны Gaussian noise в embedding space и не должны называться одной безопасной «аугментацией текста».

<!-- /unit:ad0f202eb8ae3dbafc -->

<!-- unit:d32124d503de64091e -->
<a id="u-d32124d503de64091e"></a>

В BERT MLM masked positions получают специальные targets исходных tokens согласно pretraining-протоколу. Для decoder SFT удаление отрицания из user instruction при сохранении прежнего ответа может создать неправильную пару. Наличие строкового `[MASK]` в тексте не гарантирует наличие meaningful special ID в tokenizer модели. Перед обучением проверяют сериализованные tokens, masks и target-позиции, как в курсе данных.

<!-- /unit:d32124d503de64091e -->

<!-- unit:2bad4755f1b7ad729f -->
<a id="u-2bad4755f1b7ad729f"></a>

Для additive noise $\xi\sim N(0,\sigma^2I_d)$ имеем $E\|\xi\|^2=d\sigma^2$. При d=4 и sigma=0.1 это 0.04; квадратный корень среднего квадрата norm равен 0.2. Это не тождественно среднему norm. При фиксированном sigma рост dimension увеличивает суммарный noise; scale сопоставляют с нормами embeddings. Вектор между embeddings не обязан соответствовать допустимому слову или перефразировке.

<!-- /unit:2bad4755f1b7ad729f -->

<!-- unit:121723f3118da3d13a -->
<a id="u-121723f3118da3d13a"></a>

## Attention entropy и R-Drop

<!-- /unit:121723f3118da3d13a -->

<!-- unit:df7e5d5db3d2d1acd8 -->
<a id="u-df7e5d5db3d2d1acd8"></a>

Для attention distribution $a$ entropy:

<!-- /unit:df7e5d5db3d2d1acd8 -->

<!-- unit:9f2e8712d2f5ddd898 -->
<a id="u-9f2e8712d2f5ddd898"></a>

$$
H(a)=-\sum_j a_j\log a_j.
$$

<!-- /unit:9f2e8712d2f5ddd898 -->

<!-- unit:d093602e17b353e2d3 -->
<a id="u-d093602e17b353e2d3"></a>

Если минимизируемый loss содержит $-\lambda H(a)$, он поощряет более высокую entropy; term $+\lambda H(a)$, наоборот, поощряет концентрацию. Sign принципиален. Masked positions исключаются, $0\log0$ трактуется как 0. У causal tokens разное число доступных positions; сравнение raw entropy требует учитывать эту разницу.

<!-- /unit:d093602e17b353e2d3 -->

<!-- unit:4b4524989e0727f242 -->
<a id="u-4b4524989e0727f242"></a>

Сосредоточенное attention не обязательно ошибка: точное копирование нужного token может требовать именно этого. Entropy prior не доказывает улучшение reasoning и не является direct measure объяснимости.

<!-- /unit:4b4524989e0727f242 -->

<!-- unit:da9ee96086844e7759 -->
<a id="u-da9ee96086844e7759"></a>

R-Drop делает два stochastic forward для одного input с разными dropout masks. Для выходных distributions $p^{(1)},p^{(2)}$:

<!-- /unit:da9ee96086844e7759 -->

<!-- unit:4d8e96e42993671117 -->
<a id="u-4d8e96e42993671117"></a>

$$
L=\frac{L_{\mathrm{CE}}^{(1)}+L_{\mathrm{CE}}^{(2)}}2
+\frac\alpha2\left[
D_{\mathrm{KL}}(p^{(1)}\Vert p^{(2)})+
D_{\mathrm{KL}}(p^{(2)}\Vert p^{(1)})\right].
$$

<!-- /unit:4d8e96e42993671117 -->

<!-- unit:1552e0eeb4b6a74ebd -->
<a id="u-1552e0eeb4b6a74ebd"></a>

Это consistency penalty между stochastic predictions, не требующий отдельного EMA teacher. В sequence tasks masks и нормировка по active tokens должны совпадать. Два forward увеличивают compute; comparison учитывает budget. [R-Drop](https://arxiv.org/abs/2106.14448).

<!-- /unit:1552e0eeb4b6a74ebd -->

<!-- unit:ffb94a06427dcb08a7 -->
<a id="u-ffb94a06427dcb08a7"></a>

### Знак, число доступных позиций и gradient

<!-- /unit:ffb94a06427dcb08a7 -->

<!-- unit:fd8196fa6c3ad82998 -->
<a id="u-fd8196fa6c3ad82998"></a>

Entropy сосредоточенной строки a=(0.9,0.1) равна примерно 0.32508 nats, uniform по двум позициям — log2≈0.69315. Uniform по восьми имеет entropy log8≈2.07944. Поэтому рост raw entropy вдоль causal sequence может объясняться только увеличением числа разрешённых keys, а не изменением качества attention.

<!-- /unit:fd8196fa6c3ad82998 -->

<!-- unit:2b29bdc3fd3d5835a4 -->
<a id="u-2b29bdc3fd3d5835a4"></a>

Иногда рассматривают H/log m для m>1, где m — число доступных позиций. При m=1 знаменатель нулевой: такую строку исключают или задают отдельное соглашение. Padding и causal-forbidden entries не включают в m как доступные alternatives. Равномерное attention также не всегда полезно: задача точного копирования может требовать сильной концентрации.

<!-- /unit:2b29bdc3fd3d5835a4 -->

<!-- unit:59d473012a60345f8f -->
<a id="u-59d473012a60345f8f"></a>

Для a=softmax(z) derivative entropy равна

<!-- /unit:59d473012a60345f8f -->

<!-- unit:dc924b015ad8507e98 -->
<a id="u-dc924b015ad8507e98"></a>

$$
\frac{\partial H}{\partial z_j}=-a_j(\log a_j+H).
$$

<!-- /unit:dc924b015ad8507e98 -->

<!-- unit:3c8a5d35888c26777e -->
<a id="u-3c8a5d35888c26777e"></a>

У минимизируемого loss $-\lambda H$ gradient имеет противоположный знак: $\lambda a_j(\log a_j+H)$. Для (0.9,0.1) при lambda=1 это примерно (0.19775,−0.19775); gradient descent уменьшит более высокий logit и поднимет меньший. Положительный penalty +lambda H, напротив, способствует концентрации. Entropy prior не является доказательством объяснимости и не сообщает, на правильный ли token смотрит модель.

<!-- /unit:3c8a5d35888c26777e -->

<!-- unit:2b700a67fee4f58813 -->
<a id="u-2b700a67fee4f58813"></a>

### Полный R-Drop loss двух проходов

<!-- /unit:2b700a67fee4f58813 -->

<!-- unit:045f9bb9a9806acd21 -->
<a id="u-045f9bb9a9806acd21"></a>

Для одного input два stochastic forwards дают p=(0.8,0.2) и q=(0.6,0.4), true class первый. CE равны −log0.8≈0.22314 и −log0.6≈0.51083, среднее≈0.36698. KL(p||q)≈0.09152, KL(q||p)≈0.10465; их полусумма≈0.09808. При alpha=2 в соглашении раздела 3 итоговый loss≈0.56315.

<!-- /unit:045f9bb9a9806acd21 -->

<!-- unit:4b160ae5b11d6ce59b -->
<a id="u-4b160ae5b11d6ce59b"></a>

Супервизия требует правильного ответа, а symmetric consistency уменьшает расхождение stochastic predictions. Нужны обе части: два одинаково неправильных distributions имеют нулевой KL. Separate teacher не требуется, а gradient выбранного R-Drop objective может проходить через оба outputs. Если одну сторону detached, численное значение forward loss может совпасть, но gradient-процедура будет иной. Исходный метод описан в [R-Drop](https://arxiv.org/abs/2106.14448).

<!-- /unit:4b160ae5b11d6ce59b -->

<!-- unit:a15cbfe8f2a18e8304 -->
<a id="u-a15cbfe8f2a18e8304"></a>

При dropout p=0 и отсутствии другой stochasticity оба forwards совпадают и KL=0. Это полезная проверка механизма, а не аргумент бесплатно делать два одинаковых прохода. R-Drop увеличивает объём вычислений; exact время и память зависят от batching, autograd и checkpointing. На обычном inference не требуется автоматически усреднять два stochastic outputs: это отдельное решение о decoding.

<!-- /unit:a15cbfe8f2a18e8304 -->

<!-- unit:71c6fbf690c9f69812 -->
<a id="u-71c6fbf690c9f69812"></a>

### Sequence alignment и loss masks

<!-- /unit:71c6fbf690c9f69812 -->

<!-- unit:2c79db616e18d80b48 -->
<a id="u-2c79db616e18d80b48"></a>

В teacher-forced sequence training оба passes получают один и тот же context на каждой target-позиции. Сравниваются distributions одного следующего token при одинаковой истории; padding и excluded targets не должны попадать в token-average по ошибке. CE и KL reductions задают явно, иначе coefficient alpha меняет смысл с длиной ответа и размером vocabulary.

<!-- /unit:2c79db616e18d80b48 -->

<!-- unit:cd5e8b3f4d92a9fc8d -->
<a id="u-cd5e8b3f4d92a9fc8d"></a>

Если вместо этого независимо сгенерировать два полных ответа, на позиции t у них уже могут быть разные предыдущие tokens. Это разные conditional contexts, и прямое сопоставление distributions по одному индексу t не реализует прежний objective автоматически. Независимые histories требуют другого выравнивания или sequence-level критерия.

<!-- /unit:cd5e8b3f4d92a9fc8d -->

<!-- unit:a53e56b6d341af296a -->
<a id="u-a53e56b6d341af296a"></a>

## Sequence objectives и decoding

<!-- /unit:a53e56b6d341af296a -->

<!-- unit:671b138d26bc9733e9 -->
<a id="u-671b138d26bc9733e9"></a>

Token-level smoothing независимо смягчает target в каждом шаге. Sequence-level objective может распределять массу по альтернативным допустимым целым sequences, например translations. Нужно определить, как генерируются alternatives, чем они взвешены и сохраняют ли смысл. Нельзя назвать любой штраф за длину sequence-level label smoothing.

<!-- /unit:671b138d26bc9733e9 -->

<!-- unit:609ad0cbccbc3cfdd3 -->
<a id="u-609ad0cbccbc3cfdd3"></a>

Length penalty в beam search меняет ranking уже генерируемых кандидатов без gradient update. Training regularization меняет weights. Их effects могут пересекаться по длине outputs, но это разные вмешательства и разные места подбора hyperparameters.

<!-- /unit:609ad0cbccbc3cfdd3 -->

<!-- unit:7fe17cadbb0f1ed160 -->
<a id="u-7fe17cadbb0f1ed160"></a>

Sequence-level soft target Q(s|x) задаёт предпочтения целым допустимым ответам. Objective имеет вид $-\sum_sQ(s\mid x)\log P_\theta(s\mid x)$, где autoregressive log probability каждой sequence складывается по её собственной истории, включая объявленное правило EOS. Если alternatives различаются длиной, сумма token-loss и среднее по длине задают разные веса; это часть objective.

<!-- /unit:7fe17cadbb0f1ed160 -->

<!-- unit:8d8a595fc52c0cbdf9 -->
<a id="u-8d8a595fc52c0cbdf9"></a>

Две альтернативы перевода могут сохранять смысл, а две строки с разными числами или именами — нет. Происхождение, веса и проверка alternatives поэтому важнее одного названия smoothing. Нельзя независимо смешать marginal tokens разных ответов и считать, что получилось то же распределение корректных sequences.

<!-- /unit:8d8a595fc52c0cbdf9 -->

<!-- unit:aa6133117aa8ea8deb -->
<a id="u-aa6133117aa8ea8deb"></a>

Beam length penalty действует на ранжирование уже оцениваемых кандидатов. Например, для условного score logP/L кандидат с logP=−2 и L=2 получает −1, а с logP=−3 и L=4 — −0.75. Без нормировки первый имеет больше probability, с такой нормировкой предпочтут второй. Параметры модели не изменились; поменялось правило выбора. Это не training regularizer, хотя output lengths и итоговая метрика могут измениться.

<!-- /unit:aa6133117aa8ea8deb -->

<!-- unit:5e9b529127843c1110 -->
<a id="u-5e9b529127843c1110"></a>

## Entropy, KL и PPO

<!-- /unit:5e9b529127843c1110 -->

<!-- unit:4a31e9e078cbf67747 -->
<a id="u-4a31e9e078cbf67747"></a>

При maximization expected return entropy bonus имеет sign плюс:

<!-- /unit:4a31e9e078cbf67747 -->

<!-- unit:6feab56e60f23c0071 -->
<a id="u-6feab56e60f23c0071"></a>

$$
J=\mathbb E[R]+\beta H(\pi)
-\lambda D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}}).
$$

<!-- /unit:6feab56e60f23c0071 -->

<!-- unit:1a692ed7edfda3edd9 -->
<a id="u-1a692ed7edfda3edd9"></a>

Entropy поощряет diversity/exploration относительно самой distribution. KL ограничивает отклонение от заданной reference, которая может быть далеко не uniform. Для $\pi=(0.9,0.1)$ entropy положительна; KL к такой же reference ноль. Они не взаимозаменяемы.

<!-- /unit:1a692ed7edfda3edd9 -->

<!-- unit:60044cf00705154fe6 -->
<a id="u-60044cf00705154fe6"></a>

PPO clipping ограничивает incentive больших изменений probability ratios по собранным rollouts, но не является тем же KL penalty. Target network в value-based RL стабилизирует bootstrap targets через медленное обновление; её назначение тоже отличается от entropy regularization. Общий термин «стабилизация» не делает механизмы эквивалентными.

<!-- /unit:60044cf00705154fe6 -->

<!-- unit:73009fea403119a5da -->
<a id="u-73009fea403119a5da"></a>

### Entropy и KL предпочитают разные distributions

<!-- /unit:73009fea403119a5da -->

<!-- unit:5abccda71b2b4bcae9 -->
<a id="u-5abccda71b2b4bcae9"></a>

Для fixed action set uniform u имеем $KL(\pi\Vert u)=\log|A|-H(\pi)$. Поэтому entropy maximization эквивалентна приближению к uniform при этих условиях. Но reference policy обычно не uniform. Если все rewards одинаковы, entropy-only objective предпочитает uniform, а objective с penalty KL(pi||reference) — саму reference.

<!-- /unit:5abccda71b2b4bcae9 -->

<!-- unit:55bff31f0f5cf789a2 -->
<a id="u-55bff31f0f5cf789a2"></a>

При reference=(0.9,0.1) её entropy≈0.32508 и KL к себе=0. Uniform имеет большую entropy≈0.69315, но KL(uniform||reference)≈0.51083. Поддерживать разнообразие и сохранять прежнюю неравномерную policy — разные цели. Sign проверяют с учётом того, максимизируется J или минимизируется −J.

<!-- /unit:55bff31f0f5cf789a2 -->

<!-- unit:111999f6cc375fcb0d -->
<a id="u-111999f6cc375fcb0d"></a>

В простом bandit objective $\sum_a\pi_aR_a-\lambda KL(\pi\Vert\rho)$ с положительной reference rho и lambda>0 стационарность с ограничением суммы probabilities даёт

<!-- /unit:111999f6cc375fcb0d -->

<!-- unit:3e012a08749487e835 -->
<a id="u-3e012a08749487e835"></a>

$$
\pi_a^*=\frac{\rho_a\exp(R_a/\lambda)}
{\sum_b\rho_b\exp(R_b/\lambda)}.
$$

<!-- /unit:3e012a08749487e835 -->

<!-- unit:cd45392f21fa236d5a -->
<a id="u-cd45392f21fa236d5a"></a>

При равных rewards остаётся rho. При rho=(0.9,0.1), lambda=1 и rewards=(0,log9) получаем uniform: повышенная награда второго action компенсировала его меньшую reference probability. Это условная аналитическая модель одного состояния, не формула готового обучения всей LLM. Если reference запрещает action с нулевой probability, KL support требует отдельного внимания.

<!-- /unit:cd45392f21fa236d5a -->

<!-- unit:e4bd9b1ea16a8850ca -->
<a id="u-e4bd9b1ea16a8850ca"></a>

### PPO clipping не является жёсткой границей новой policy

<!-- /unit:e4bd9b1ea16a8850ca -->

<!-- unit:d2ae84a74d3e11a83d -->
<a id="u-d2ae84a74d3e11a83d"></a>

Пусть $r=\pi_\theta(a\mid s)/\pi_{old}(a\mid s)$ и advantage A получен по rollout. PPO максимизирует среднее

<!-- /unit:d2ae84a74d3e11a83d -->

<!-- unit:797c377708b03c0f2d -->
<a id="u-797c377708b03c0f2d"></a>

$$
\min\left(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A\right).
$$

<!-- /unit:797c377708b03c0f2d -->

<!-- unit:59f6ea74b8367d7741 -->
<a id="u-59f6ea74b8367d7741"></a>

При epsilon=0.2, A=2 и r=1.5 обычный term равен 3, clipped surrogate — 2.4. При A=2 и r=0.5 результат остаётся 1: ухудшение вероятности полезного действия не скрывается нижней clipping-границей. При A=−2 и r=0.5 берётся −1.6 вместо −1; при A=−2 и r=1.5 остаётся −3. Знак advantage определяет, какая сторона перестаёт давать дополнительное поощрение.

<!-- /unit:59f6ea74b8367d7741 -->

<!-- unit:203657c0c86d844bd7 -->
<a id="u-203657c0c86d844bd7"></a>

Это изменение surrogate incentive на собранных actions. Оно не проецирует все probabilities новой policy в разрешённый диапазон и не гарантирует ограничение KL на всех состояниях. Общие параметры, другие actions и дополнительные terms могут продолжать изменять ratio. Reference для KL и old policy для ratio имеют разные роли: reference может оставаться фиксированной SFT-моделью, old — обновляться при новом сборе rollout. Основной алгоритм приведён в [PPO paper](https://arxiv.org/abs/1707.06347), а полный trajectory-разбор находится в lifecycle-курсе.

<!-- /unit:203657c0c86d844bd7 -->

<!-- unit:fe0555d6963cce1011 -->
<a id="u-fe0555d6963cce1011"></a>

### Token masks, reward scale и target networks

<!-- /unit:fe0555d6963cce1011 -->

<!-- unit:154665a00f0bc76bc4 -->
<a id="u-154665a00f0bc76bc4"></a>

В LLM policy entropy и KL обычно усредняют или суммируют по выбранным action tokens, отделяя их от user/tool context. Сумма per-token KL создаёт больший общий cost для более длинного ответа при том же среднем расхождении. Поэтому sum, token mean и sequence mean не взаимозаменяемы. Увеличение reward scale при прежнем KL coefficient меняет компромисс даже без изменения optimizer.

<!-- /unit:154665a00f0bc76bc4 -->

<!-- unit:8eccf935b8dfb5ed60 -->
<a id="u-8eccf935b8dfb5ed60"></a>

Target network value-based RL служит другой цели: вычисляет более медленно меняющийся bootstrap target $y=r+\gamma\max_aQ_{target}(s',a)$, который фиксируют при соответствующем regression update. При r=1, gamma=0.9 и target values (2,4) получаем 4.6. Если online values уже (10,20), подстановка их в target дала бы 19. Отдельное состояние ограничивает движение этой цели, а не напрямую повышает entropy policy.

<!-- /unit:8eccf935b8dfb5ed60 -->

<!-- unit:4f2de91945e88c40b1 -->
<a id="u-4f2de91945e88c40b1"></a>

EMA target network и EMA teacher похожи способом обновления весов, но отличаются назначением target: reward bootstrap против consistency prediction. Стабильность сама по себе не доказывает правильность обоих. Они требуют data protocol, update schedule и независимой оценки именно нужной задачи.

<!-- /unit:4f2de91945e88c40b1 -->

<!-- unit:09cb25d71621dcaff7 -->
<a id="u-09cb25d71621dcaff7"></a>

## Контроль и практика

<!-- /unit:09cb25d71621dcaff7 -->

<!-- unit:349a5d0937e588b4ac -->
<a id="u-349a5d0937e588b4ac"></a>

1. Где gradient weight tying? **Ответ:** shared matrix получает суммы вкладов input и output путей.
2. Какой sign повышает attention entropy при minimization? **Ответ:** отрицательный перед $\lambda H$.
3. Почему beam length penalty не training regularizer? **Ответ:** он меняет decoding ranking без изменения parameters.

<!-- /unit:349a5d0937e588b4ac -->

<!-- unit:1fa8a641da8ee52923 -->
<a id="u-1fa8a641da8ee52923"></a>

**Практика:** выпишите shapes tying для $V=1000,d=64$. Вычислите symmetric KL двух distributions $(0.8,0.2)$ и $(0.6,0.4)$. Затем сравните entropy и KL к reference для двух policies; объясните, какой objective поддерживает разнообразие, а какой — близость к прежнему поведению.

<!-- /unit:1fa8a641da8ee52923 -->

<!-- unit:88b1a7f97207196158 -->
<a id="u-88b1a7f97207196158"></a>

**Схема курса:** same input/sequence → two stochastic forwards or policy rollout → consistency/KL or constrained update → evaluate task success, diversity and policy violations

<!-- /unit:88b1a7f97207196158 -->

<!-- unit:e6a2ea020879c01b65 -->
<a id="u-e6a2ea020879c01b65"></a>

<!-- source-depth-regularization-З -->

<!-- /unit:e6a2ea020879c01b65 -->

<!-- unit:38d3b1e8292ebed727 -->
<a id="u-38d3b1e8292ebed727"></a>

1. **Почему tied E получает gradients даже для token, которого не было во входе?** Его row может участвовать в output softmax, хотя input lookup этой row отсутствовал.
2. **Почему R-Drop не требует EMA teacher?** Согласуются два stochastic outputs самой модели на одном условии.
3. **Почему raw attention entropy нельзя сравнивать без числа доступных keys?** Максимальная entropy растёт как log m.
4. **Почему PPO ratio и reference KL нельзя считать одним ограничением?** Они относятся к разным distributions и по-разному воздействуют на objective.

<!-- /unit:38d3b1e8292ebed727 -->

<!-- unit:1d3262c2e71fd4cf60 -->
<a id="u-1d3262c2e71fd4cf60"></a>

Практика: выписать tying shapes и два gradient paths, проверить R-Drop toy loss и активные sequence masks. Затем вычислить entropy/reference KL и четыре случая PPO clipping, объяснив направление каждого стимула. Все policies и rewards главы учебные; реальное RL/SFT-обучение пользователя и результаты его модели здесь не заявляются.

<!-- /unit:1d3262c2e71fd4cf60 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Чем различаются шум представлений, entropy и ограничение обновления политики?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
