# Б. LoRA: куда добавляются A/B и как они обучаются

[Оглавление](README.md). Пункт 2 и протокол 14C плана; варианты семейства — в главе В.

**Основа:** S2, строки 169–500, 1817–3404 и 3413–3487; S1, страницы 2–3, 8–9 и 15–17. Ключевой запрос пользователя: «я так и не понял, куда мы вставляем эти матрицы». **Исправления:** единая ориентация матриц, r не число параметров, точный loss первого шага, target modules не отдельные головы. **Углубление:** второй шаг SGD и прямое сравнение с merged weight.

## 1. Добавляется ветвь к конкретному linear layer

Линейная проекция для одного токена при соглашении о столбцах:

$$
x\in\mathbb{R}^{d_{\mathrm{in}}},\qquad W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},\qquad y_0=W_0x.
$$

Full fine-tuning изменял бы W0. LoRA сохраняет её frozen и задаёт обновление через произведение маленьких матриц:

$$
\begin{gathered}A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\quad B\in\mathbb{R}^{d_{\mathrm{out}}\times r},\quad s=\frac{\alpha}{r},\\\boxed{y=W_0x+sB(Ax)}.\end{gathered}
$$

Сначала A переводит вход в r чисел, затем B переводит их в d_out чисел, результат масштабируется и складывается с исходной проекцией. До суммы оба пути имеют форму $d_{\mathrm{out}}\times 1$. Вычислять и хранить плотную $BA$ на каждом training forward обычно не требуется: выгоднее два небольших умножения.

```text
x ─→ W0 (frozen) ──────────────→ сумма → y
└──→ A (trainable) → B (trainable) → ×s ─┘
```

Это не вставка дополнительных токенов и не отдельная сеть после всего Transformer. Обычно выбранный linear module оборачивается таким путём внутри блока. В row-vector batch convention формула становится $Y=XW_0^\top+sXA^\top B^\top$. Нельзя брать shapes из одного соглашения, а порядок произведения из другого: именно это противоречие было в исходном PDF.

Ранг $BA$ не превосходит r. Однако не утверждается, что любое желаемое обновление любой задачи точно низкого ранга: это ограничение пространства доступных изменений, полезность которого проверяется. $r=8$ означает восемь промежуточных координат, а не восемь параметров.

## 2. Что означают target_modules и как считать параметры

`q_proj`, `k_proj`, `v_proj` создают attention Q/K/V, `o_proj` смешивает объединённые heads. `gate_proj`, `up_proj`, `down_proj` принадлежат gated MLP. Их место разобрано в [основах](<../../01-foundations/Учебные конспекты/В — Transformer, causal mask и устройство блока.md>).

Список имён в конфигурации сообщает библиотеке, какие модули адаптировать. Конкретные правила matching, исключения и ограничения слоёв зависят от config/версии. Поэтому фраза из чата «абсолютно ко всем блокам всегда» нуждается в проверке фактических matched module names. Обычно одинаковый suffix находит множество проекций разных блоков; у каждой своя пара A/B.

Если Q projection реализована общей матрицей, выход затем reshaped на heads. LoRA на `q_proj` относится к этой матрице; отдельные adapters для каждой головы автоматически не создаются. Если архитектура использует fused QKV или GQA, имена и выходные размеры меняются.

Число весов adapter одной проекции без bias:

$$
P_{\mathrm{LoRA}}=r\,d_{\mathrm{in}}+d_{\mathrm{out}}r=r(d_{\mathrm{in}}+d_{\mathrm{out}}).
$$

Для квадратной $4096\times 4096$ и $r=8$: $P=65 536$, у полной матрицы $16 777 216$; отношение около 0.390625%. Для двух таких проекций в каждом из 32 блоков: $32\cdot 2\cdot 65 536=4 194 304$. Это учебный MHA-пример, а не универсальный count каждой модели с «7B» в названии.

Для MLP $4096\to11008$, $r=8$: $8(4096+11008)=120 832$. Если добавить несколько MLP modules, adapter существенно вырастет. Если менять r и targets одновременно, нельзя понять, какой фактор дал улучшение.

## 3. Почему A случайная, B нулевая

Желательно начать с $\Delta W=sBA=0$, чтобы initial forward сохранял base behavior. Распространённый вариант — ненулевая случайная A и нулевая B. Распределение A зависит от реализации; «всегда Gaussian» неверно. Для deterministic сравнения следует также учесть dropout и режим train/eval.

Пусть $g=\frac{\partial\mathcal{L}}{\partial y}$, $h=Ax$. Тогда:

$$
\begin{aligned}\frac{\partial\mathcal{L}}{\partial B}&=sgh^\top\in\mathbb{R}^{d_{\mathrm{out}}\times r},\\\frac{\partial\mathcal{L}}{\partial A}&=sB^\top gx^\top\in\mathbb{R}^{r\times d_{\mathrm{in}}},\\\frac{\partial\mathcal{L}}{\partial x}&=W_0^\top g+sA^\top B^\top g.\end{aligned}
$$

При $B=0$ на первом backward градиент A нулевой, B обычно ненулевой. После обновления B в следующем backward A уже может обучаться. Если обе A/B нулевые, обе производные равны нулю и такая ветвь не начнёт двигаться обычным gradient descent по task loss.

Frozen W0 не получает weight update, но её транспонированное умножение участвует в градиенте входа. Это объясняет, почему обучение adapters внутри глубокой frozen базы всё равно требует части backward compute.

## 4. Два шага на исходном числовом примере

Используем пример из PDF с исправленной арифметикой:

$$
\begin{gathered}x=\begin{bmatrix}1\\2\end{bmatrix},\quad W_0=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad A=\begin{bmatrix}0.1&0.2\end{bmatrix},\quad B=\begin{bmatrix}0\\0\end{bmatrix},\\y^*=\begin{bmatrix}6\\10\end{bmatrix},\quad s=1,\quad\eta=0.1,\quad\mathcal{L}=\frac12\lVert y-y^*\rVert_2^2.\end{gathered}
$$

**Первый forward:** $W_0x=(5,11)$, $h=Ax=0.5$, adapter output нулевой. $g=\left(-1,\,1\right)$, $\mathcal{L}=1$.

$$
\begin{gathered}\nabla_B\mathcal{L}=\begin{bmatrix}-0.5\\0.5\end{bmatrix},\quad\nabla_A\mathcal{L}=\begin{bmatrix}0&0\end{bmatrix},\\B^{(1)}=\begin{bmatrix}0.05\\-0.05\end{bmatrix},\quad A^{(1)}=\begin{bmatrix}0.1&0.2\end{bmatrix}.\end{gathered}
$$

Проверка после шага: $y=\left(5.025,\,10.975\right)$ и $\mathcal{L}=0.950625$. В PDF было около 0.9503; точное значение другое.

**Второй backward** считается при этих новых параметрах, до их следующего одновременного обновления. $g=\left(-0.975,\,0.975\right)$, $h=0.5$.

$$
\begin{gathered}\nabla_B\mathcal{L}=\begin{bmatrix}-0.4875\\0.4875\end{bmatrix},\quad B^\top g=0.05(-0.975)-0.05(0.975)=-0.0975,\\\nabla_A\mathcal{L}=\begin{bmatrix}-0.0975&-0.195\end{bmatrix},\\B^{(2)}=\begin{bmatrix}0.09875\\-0.09875\end{bmatrix},\qquad A^{(2)}=\begin{bmatrix}0.10975&0.2195\end{bmatrix}.\end{gathered}
$$

Теперь $h_{\mathrm{next}}=0.54875$, $B_{\mathrm{next}} h_{\mathrm{next}}=(0.0541890625,-0.0541890625)$. Значит:

$$
y^{(2)}=\begin{bmatrix}5.0541890625\\10.9458109375\end{bmatrix},\qquad\mathcal{L}^{(2)}=(0.9458109375)^2\approx0.89455833.
$$

A действительно начала двигаться на втором шаге. Важно вычислить оба gradients из одного набора параметров, а затем обновить A/B одновременно; если подставить уже обновлённую B в gradient A того же шага, получится другой алгоритм.

## 5. Merge и ограничения интуиции

После обучения в совместимой конфигурации можно материализовать $W_{\mathrm{merge}}=W_0+sBA$. Для первого шага:

$$
BA=\begin{bmatrix}0.005&0.010\\-0.005&-0.010\end{bmatrix},\qquad W_{\mathrm{merge}}=\begin{bmatrix}1.005&2.010\\2.995&3.990\end{bmatrix}.
$$

Умножение на x снова даёт $\left(5.025,\,10.975\right)$. Отдельные adapter matmuls больше не нужны. Но точное численное равенство зависит от dtype/rounding; quantized base может потребовать dequantization/requantization. После merge нужно повторить task evaluation. Обещание «ровно 0 ms дополнительной задержки на любом runtime» слишком сильное.

Alpha и r не отменяют подбор LR. Стандартный scaling alpha/r задаёт масштаб ветви; другие схемы могут использовать другой scaling. Из изменения r автоматически не следует неизменная динамика оптимизации.

**Ошибки:** считать A/B «новыми attention heads»; объявлять Q/V универсально лучшим набором; считать frozen base отсутствием forgetting; путать маленький rank с малой нормой обновления; забывать s в backward; merge-ить в другую revision базы.

## 6. Самопроверка

1. $d_{\mathrm{in}}=3,\qquad d_{\mathrm{out}}=5,\qquad r=2$. Shapes и число параметров? **Ответ:** A $2\times 3$, B $5\times 2$; 16 весов против 15 в W0. При таких маленьких размерах LoRA не обязана экономить параметры.
2. Почему первый grad_A нулевой? **Ответ:** в формуле стоит Bᵀg, B=0.
3. Можно ли отдельно усреднить A/B двух adapters и получить среднее их ΔW? **Ответ:** в общем случае нет: возникают cross terms; подробнее в главе Ж.

**Практика:** повторить два шага, затем задать $s=2$ и заново вычислить gradients. Не удваивать только forward: s входит и в derivatives. Основа метода: [LoRA](https://arxiv.org/abs/2106.09685); параметры/режимы реализации: [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora).

<!-- latex-display-v1 -->
