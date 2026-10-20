# Г. PEFT за пределами LoRA: что именно меняется

[Оглавление](README.md). Пункт 3 плана.

**Основа:** S3, строки 49–224. Сохранён вопрос: «лучше раскрыть суть Prefix Tuning и BitFit: как и почему работают, что и куда добавляем». **Исправление:** soft prompt на входе и layerwise prefix K/V разделены; исходное описание P-Tuning v2 «только первый слой» не переносится. Универсальные проценты качества/latency удалены. **Углубление:** формы и один attention-пример с prefix.

**Уровень:** Основа PEFT-ветки. **Предпосылки:** C1, A3–A5. **Используют:** C8/C10 and adapter serving.

## 1. Adapter layer добавляет обучаемую нелинейную функцию

LoRA меняет linear operation через low-rank update без обязательной новой нелинейности. Классический bottleneck adapter добавляет небольшой модуль к hidden representation:

$$
\begin{gathered}h\in\mathbb{R}^{D},\quad W_{\mathrm{down}}\in\mathbb{R}^{r\times D},\quad W_{\mathrm{up}}\in\mathbb{R}^{D\times r},\\u=\varphi(W_{\mathrm{down}}h+b_{\mathrm{down}})\in\mathbb{R}^{r},\qquad h_{\mathrm{new}}=h+W_{\mathrm{up}}u+b_{\mathrm{up}}.\end{gathered}
$$

Точное место — после attention/MLP или в другой ветви — определяется архитектурой adapter. Главная идея: исходная база заморожена, новая небольшая функция обучается менять её representations. Down/up обычно дают $2Dr$ весов плюс biases. При $D=768$,$r=64$ это 98 304 весов на такой module без biases.

Пример $D=2$,$r=1$: $W_{\mathrm{down}}=\left(1,\,-1\right)$, $W_{\mathrm{up}}=[0.5,1]^\top $, φ=ReLU. Для h=(3,1) bottleneck равен 2, поправка (1,2), результат (4,3). Для h=(1,3) bottleneck 0, результат прежний. Из-за ReLU нельзя в общем случае свернуть всю функцию в одну постоянную матрицу, одинаковую для всех h. Поэтому обычный nonlinear adapter не merge-ится так же, как линейная LoRA.

Это не доказательство, что adapters всегда медленнее любой LoRA-конфигурации. Добавочные операции и serial dependency создают потенциальную цену, но measured latency зависит от runtime, batch, kernels и способа реализации.

### Почему это отдельный способ адаптации, хотя формы похожи на LoRA

В S3 adapter описан как down→nonlinearity→up. Его bottleneck напоминает факторы LoRA, но объект входа и место residual принципиальны. LoRA меняет выбранный linear W0x внутри блока. Nonlinear adapter получает representation h и возвращает h+g(h). В зависимости от design его ставят после attention, после MLP или в другой residual branch. Нельзя по одной паре down/up заключить, что эти методы имеют одинаковую функцию.

Если φ — identity и biases нулевые, adapter действительно становится $(I+W_up W_down)h$, то есть линейной residual correction. При ReLU область активных channels зависит от h. В примере [3,1] и [1,3] один и тот же module в первом случае меняет обе координаты, во втором выключается. Одна постоянная ΔW не воспроизведёт все такие переключения, если нет дополнительных ограничений.

При D=768,r=64 count без biases98304. С двумя bias vectors добавляется64+768=832, итог99136. Если в каждом из12 blocks стоит по два adapters, count=2379264; если по одному —1189632. Процент модели зависит от placements и точного P_base. Поэтому таблица «adapter всегда1–3%, LoRA всегда0.5–2%» из чата не задаёт метод и не заменяет подсчёт.

Nonlinear adapter после training обычно остаётся частью graph. Это потенциальная дополнительная serial работа, но её стоимость зависит от batch и kernels. Возможность держать несколько задач с shared frozen base есть не только у prefixes: adapters и LoRA тоже допускают разные task-specific параметры при поддерживаемом runtime. [Adapters](https://arxiv.org/abs/1902.00751).

## 2. Soft prompt: обучаемые входные векторы

Обычный текстовый prompt сначала разбивается tokenizer. Soft prompt — набор свободных вещественных векторов $P:m\times D$, которые оптимизируются gradient descent. Они могут не соответствовать ни одному token ID и не обязаны декодироваться в осмысленные слова.

При входе X длины L строится $\begin{bmatrix}P\\X\end{bmatrix}$ длины m+L. Через frozen Transformer проходят и prefix, и реальные токены. Loss считается по целевым позициям задачи; gradient идёт через frozen операции к P. Весовые матрицы модели не меняются, но P учится создавать полезные внутренние состояния и управлять дальнейшим attention.

Если $D=768$,$m=20$, обучается 15 360 входных чисел. Это небольшой parameter state, но вычислительная длина увеличивается. Влияние на context budget/positions зависит от реализации. При $m=0$ возвращается обычный вход без trainable prompt.

На inference soft prompt нужно снова подать в той же конфигурации. Его обучение — отдельный offline процесс; обычный пользовательский текст не становится soft prompt автоматически и не обновляется при каждом сообщении.

### Вопрос чата: почему нельзя просто оптимизировать слова prompt

Token ID дискретен: между индексами «кот» и «собака» нет содержательного промежуточного token0.3. Обычный gradient descent не выбирает автоматически соседнее слово по производной ID. Soft prompt вместо этого оптимизирует D вещественных координат каждого virtual token. Градиент language loss по этим координатам определён через differentiable backbone.

Пусть модель — фиксированная функция fθ(P,X), θ frozen. Тогда задача $\min_P\mathcal L(f_\theta(P,X),Y)$ имеет обычный gradient по P. Frozen означает θ не обновляется optimizer; производные по входу P всё ещё вычисляются. Если обернуть fθ в no-grad целиком, этот путь исчезнет. Поэтому «не обучаем базу» не означает «обучение prompt не имеет backward через базу».

Входной P — параметр, общий для примеров выбранной задачи, а contextual states виртуальных позиций меняются в зависимости от architecture и разрешённого внимания. Нельзя считать soft prompt произвольным человеческим текстом, который обязательно можно распечатать. Поиск ближайших token embeddings может дать читаемые ID, но изменит векторы и не гарантирует сохранения поведения.

Инициализировать P можно по определённым token embeddings или иначе; scale и расположение влияют на optimization. Успех на одной task/model не доказывает тезис чата «continuous prompt всегда лучше любого человеческого текста». Сравнивать нужно с настроенным text prompt и тем же evaluation set. [Prompt tuning](https://arxiv.org/abs/2104.08691).

## 3. Prefix tuning: дополнительные K/V в слоях

В prefix tuning обучаемые состояния добавляются к keys/values attention, часто для нескольких/всех слоёв. Для одной головы:

$$
\begin{gathered}Q\in\mathbb{R}^{L\times d_k},\quad K_{\mathrm{real}}\in\mathbb{R}^{L\times d_k},\quad V_{\mathrm{real}}\in\mathbb{R}^{L\times d_v},\\K_{\mathrm{prefix}}\in\mathbb{R}^{m\times d_k},\quad V_{\mathrm{prefix}}\in\mathbb{R}^{m\times d_v},\\K_{\mathrm{all}}=\begin{bmatrix}K_{\mathrm{prefix}}\\K_{\mathrm{real}}\end{bmatrix},\quad V_{\mathrm{all}}=\begin{bmatrix}V_{\mathrm{prefix}}\\V_{\mathrm{real}}\end{bmatrix},\quad QK_{\mathrm{all}}^\top\in\mathbb{R}^{L\times(m+L)}.\end{gathered}
$$

Реальные queries получают дополнительные обучаемые источники. Векторы prefix K/V могут параметризоваться непосредственно или через небольшую reparameterization network. Для N слоёв и общей ширины KV D упрощённый прямой budget порядка $2NmD$; в GQA он зависит от KV width. При $N=12$,$m=10$,$D=768$ это 184 320 чисел до возможной дополнительной сети.

Числовая иллюстрация одной query q=1: реальный key $k=0$, value $v=2$; prefix key k_p=1, value v_p=4. Без prefix единственный source даёт output 2. С prefix scores $\left(1,\,0\right)$, веса примерно $\left(0.7311,\,0.2689\right)$, output $0.7311\cdot 4+0.2689\cdot 2\approx 3.4621$. Можно изменить output без изменения исходных W_Q/W_K/W_V, научив prefix влиять на веса и передаваемое содержимое.

Это объяснение механизма, а не обещание, что prefix всегда найдёт нужное решение. Fixed backbone ограничивает класс функций; длина prefix, инициализация, данные и размер базы влияют на качество. Deep prompt tuning, P-Tuning и P-Tuning v2 имеют свои схемы расположения параметров; их нельзя сводить к одному правилу «виртуальные токены только на первом слое». При реализации выбирают конкретный метод и сверяют config.

### Что exactly добавлено в attention и какие query остаются

У прямого layerwise prefix реальные query Q остаются L×d_k. Добавлены m key rows и m value rows; scores становятся L×(m+L). Новые prefix rows не обязаны иметь свои query outputs, которые прогоняются далее как обычные input positions. Это отличается от входного soft prompt, где весь набор m+L tokens проходит обычные projections и layers.

Для L=3,m=2 causal доступ у real query1 включает два prefix keys и первый real key, у query2 — prefix плюс первые два real keys, у query3 — все пять. Запрещённые будущие real keys всё ещё маскируются. Дополнительные learned sources не дают права читать будущие targets. При multi-head prefix хранится по соответствующим head/channel axes, а при GQA ширина KV может быть меньше ширины Q.

В числовом примере q=1,kp=1,kr=0,vp=4,vr=2 prefix weight $a=\sigma(1)\approx0.731059$ и output $o=2+2a\approx3.462117$. Локально $\partial o/\partial v_p=a$, а $\partial o/\partial k_p=2a(1-a)\approx0.393224$. Обучение может менять и содержимое virtual value, и то, сколько веса получает prefix key. Вот точный смысл метафоры из чата «шпаргалка направляет модель».

Если для demonstration target o*=3, squared loss даёт gradient o−3≈0.462117. По vp gradient≈0.337835, по kp≈0.181715: gradient descent уменьшает output через оба доступных рычага. Фиксированные W_Q/W_K/W_V при этом не обновляются. Prefix не обязан хранить отдельный дословный ответ на каждый prompt; он учится shared conditioning, полезному на training distribution. [Prefix tuning](https://arxiv.org/abs/2101.00190).

### Почему deep prefix нельзя без оговорок свести к input prompt

Входной soft prompt создаёт layerwise K/V через связанный frozen graph. Прямой prefix задаёт их отдельно или через reparameterization. Не каждый произвольный набор independently chosen K/V всех layers достижим одним и тем же input P. Поэтому методы имеют разное пространство interventions даже при похожей фразе «виртуальные токены».

P-Tuning, prompt tuning и P-Tuning v2 надо различать по конкретной работе; исходная оговорка S3 про «v2 только первый слой» неверна. Deep prompt tuning использует interventions на разных глубинах. Название не задаёт автоматически одинаковые placements во всех architectures. [P-Tuning v2](https://arxiv.org/abs/2110.07602).

### Цена prefix на inference

Для N=12,m=10,D_KV=768 прямые K/V содержат184320 values. BF16 raw prefix state≈360 KiB; для N=4,m=2,D_KV=16 —256 values. Но каждый real query может обращаться к дополнительным m sources; input soft prompt также проходит собственные layers. Возможность кэшировать часть prefix вычислений зависит от режима и architecture.

Увеличение prefix length добавляет attention/key positions; лимит context/позиционные индексы и supported cache contract нужно проверять у конкретного implementation. Это не всегда буквальная одинаковая потеря m текстовых tokens из пользовательского API limit. Также нельзя считать смену prefix совместимой с K/V прежнего диалога: representations реальных токенов могли уже зависеть от старого conditioning.

## 4. BitFit: меняются только bias-параметры

Линейный слой $y=Wx+b$ при frozen W может менять b. Это сдвигает пороги активаций/признаки, используя уже выученные направления. В одномерной ReLU $y=\operatorname{max}\left(0,\,wx+b\right)$ значение b двигает границу $x=\frac{-b}{w}$ при ненулевом w. Поэтому bias tuning может заметно изменить поведение глубокой сети, хотя не поворачивает непосредственно W.

Например, $y=ReLU(2x+b)$. При $b=0$ активация включается после x=0. При $b=-2$ — после x=1. Для $x=0.5$ output меняется с 1 на 0. Это небольшой, но содержательный рычаг адаптации.

Ограничение зависит от архитектуры: в сети без нужных biases обучаемых параметров для буквального BitFit может оказаться мало или не быть. Нельзя обещать определённый процент от P для любой LLM. BitFit также не означает «bias в смысле социальной предвзятости» — речь об additive parameters.

Почему столь ограниченный метод иногда работает? Представления предобученной модели уже содержат полезные признаки; downstream задаче бывает достаточно изменить их активацию/сочетание. Это эмпирический сценарий, а не доказательство, что все сложные задачи решаются bias-only update.

### Что делает bias в глубокой сети, кроме последнего сдвига

Для $y=W_2\operatorname{ReLU}(W_1x+b_1)+b_2$ производная по b1 проходит через ReLU mask и W2. Если сдвинуть b1 так, что hidden unit пересечёт activation threshold, изменится кусочная функция по x. Поэтому bias-only update может менять input-dependent решения, хотя один отдельный linear y=Wx+b меняет только additive offset.

В одномерном примере ReLU(2x+b) b=−2 выключает inputs x≤1. Уменьшение b не поворачивает W, но меняет, какой диапазон входов использует выученный признак. В глубоком стеке таких переключений много. Это более точное объяснение полезности BitFit, чем утверждение «мы лишь усиливаем нужные концепции, не меняя знания вообще».

Если выбранный checkpoint почти не имеет biases, literal BitFit не создаст их из воздуха. Добавить новые b — уже изменение parameterization, его нужно объявить. Небольшое число trainable values не гарантирует маленькой norm update или отсутствие regression. Заявления S3 «не умеет JSON» и «всегда проваливается на reasoning» без задания/эксперимента слишком категоричны: scope empirical результатов надо сохранять. [BitFit](https://arxiv.org/abs/2106.10199).

Для сравнения фиксируют данные и token budget, измеряют target quality, regression tasks, raw trainable state и full runtime. LoRA с merged weights, nonlinear adapter и long prefix имеют разные memory/compute trade-offs; подбирать один метод без tuning, а другому дать большой search unfair. Метафоры эксперта, шпаргалки и регуляторов из чата помогают запомнить intervention, но итог доказывают graph и evaluation.

## 5. Как сравнивать методы

Сначала определить доступный механизм: новые nonlinear functions, входные vectors, layerwise KV или часть исходных параметров. Затем посчитать trainable state и измерить полный runtime. У prefix меняется длина attention, у adapters — дополнительные операции, у BitFit — исходные biases, у LoRA — low-rank branch с возможным merge.

На одном split/наборе prompts сравнивать baseline и один метод при одинаковом data budget; подобрать параметры честно, не дать одному методу в десять раз больше tuning. Не переносить из старой таблицы «BitFit даёт 90% качества LoRA»: нет универсальной метрики и условий для такого процента.

1. Почему frozen backbone не мешает обучению soft prompt? **Ответ:** gradient по входным P проходит через фиксированную differentiable функцию.
2. Что отличает prefix K/V от входного soft prompt? **Ответ:** место вмешательства; K/V могут задаваться напрямую в слоях, а входные vectors сначала проходят обычный стек.
3. Почему nonlinear adapter обычно нельзя merge как LoRA? **Ответ:** его функция зависит от h через нелинейность и не сводится к одному постоянному linear update.

**Практика:** нарисовать для $L=3$,$m=2$ форму attention scores $3\times 5$ с prefix K/V. Отдельно выписать форму soft-prompt входа $5\times D$. Посчитать budget обеих схем для $N=4$,$D=16$ без biases/reparameterization.



**Схема курса:** замороженный Transformer → adapter внутри блока | prefix K/V в attention | soft prompt перед token embeddings | обучаемые bias → задача и serving interface

<!-- latex-display-v1 -->

<!-- source-depth-adaptation-Г -->
