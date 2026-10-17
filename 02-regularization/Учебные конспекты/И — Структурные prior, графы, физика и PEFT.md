# И. Ограничение должно соответствовать структуре задачи

[Оглавление](README.md). Объединены пункты 11 и 13. **Основа:** S1–S3 подробно объясняют PEFT/LoRA; временные, графовые и физические penalties — дополнение по плану. Ссылка на FRESNEL/NICA в исходном плане не используется как доказательство вашего конкретного опыта или готового эксперимента.

## 1. Гладкость по времени и seasonality

Если параметры или predictions меняются во времени, можно штрафовать резкие изменения:

$$
\Omega_{\mathrm{time}}=
\lambda\sum_{t=2}^T\|w_t-w_{t-1}\|_2^2.
$$

Это prior о локальной плавности. Seasonal penalty связывает одинаковые phases периода $P$:

$$
\Omega_{\mathrm{season}}=
\lambda_s\sum_{t=P+1}^T\|w_t-w_{t-P}\|_2^2.
$$

Локальная гладкость и seasonality различаются: сигнал может резко меняться каждый день и повторять pattern каждую неделю. Если произошёл настоящий regime change, чрезмерный smoothing сгладит важный переход. Period и penalty выбирают по доступному прошлому, без fit по будущему test.

Штрафовать можно weights, latent states или outputs, но это разные assumptions. Небольшое изменение weights не гарантирует небольшое изменение output для всех inputs, а smooth outputs не означают simple parameters.

## 2. Graph Laplacian

Пусть $A$ — симметричная неотрицательная adjacency, $D_{ii}=\sum_j A_{ij}$ и $L=D-A$. Для scalar predictions $f_i$:

$$
f^\top Lf=\frac12\sum_{i,j}A_{ij}(f_i-f_j)^2.
$$

Penalty делает predictions соседних nodes похожими. Для цепочки из трёх nodes с unit edges и $f=(0,1,1)$ значение равно 1: первый edge даёт разность 1, второй — 0. Factor $1/2$ устраняет двойной учёт symmetric edges.

Это полезно при homophily: близость графа соответствует сходству target. При heterophily соседние nodes могут систематически иметь разные labels; тот же prior ухудшит решение. Построение graph по full data может быть допустимо в явно transductive setup, но отличается от inductive проверки на новых nodes. Если graph построен по test labels или недоступным будущим связям, это leakage.

Для vector-valued outputs используют $\operatorname{tr}(F^\top LF)$. Normalized Laplacian меняет weighting degree; нельзя подменять одну convention другой без изменения objective.

## 3. Physics-informed constraints

Если известное поле $u_\theta(t,x)$ должно удовлетворять operator equation $\mathcal N[u]=0$, добавляют residual loss на collocation points:

$$
L=L_{\mathrm{data}}+
\lambda_r\frac1M\sum_{j=1}^M
\|\mathcal N[u_\theta](z_j)\|^2+
\lambda_b L_{\mathrm{boundary}}.
$$

Например, для затухания $du/dt+ku=0$ residual равен $r(t)=\partial_tu_\theta(t)+ku_\theta(t)$. Начальное условие $u(0)=u_0$ задаётся отдельным term или architecture. Малый data loss не гарантирует соблюдение equation между measured points; малый residual на конечной сетке не гарантирует глобальную точность решения.

Units и scales важны: data error и differential residual имеют разные размерности. Nondimensionalization/normalization и weights определяют compromise. Неточный physical law или неизвестные boundary conditions могут внести bias. «Физический» loss не является автоматической гарантией истины модели.

## 4. Ограничение числа trainable parameters

При LoRA weight update:

$$
W=W_0+sBA,
\quad B\in\mathbb R^{d_{\mathrm{out}}\times r},
\quad A\in\mathbb R^{r\times d_{\mathrm{in}}},
\quad\operatorname{rank}(BA)\le r.
$$

Trainable parameters — $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ вместо $d_{\mathrm{in}}d_{\mathrm{out}}$ в выбранном matrix. Для $1024\times1024,r=8$ это 16 384 вместо 1 048 576, без учёта других modules. Но rank не ограничивает norm: умножив B на большое число, можно получить большой update и при rank 1. Поэтому low rank не гарантирует близость к pretrained behaviour и отсутствие forgetting.

Подробный forward/backward LoRA, DoRA и AdaLoRA уже разобран в [адаптации](../../06-llm/02-adaptation/Учебные%20конспекты/README.md). Здесь важен смысл constraint: доступное множество обновлений уже, но его соответствие задаче проверяется на held-out data.

Bottleneck $h\mapsto g(f(h))$ с промежуточной dimension $r<d$ ограничивает representation. В linear случае rank композиции не выше $r$; nonlinear bottleneck может выражать более сложный manifold и не равен буквально rank-r linear map. Слишком узкий bottleneck теряет информацию, а autoencoder может научиться nuisance details вместо полезного semantics. Weight tying из предыдущей главы — другой способ связать parameters без такого промежуточного слоя.

## 5. Контроль и практика

1. Когда graph smoothing вреден? **Ответ:** когда adjacency не соответствует сходству target либо содержит ошибочную/недоступную информацию.
2. Гарантирует ли low rank малое update? **Ответ:** нет; rank и norm — разные свойства.
3. Почему physical residual требует scaling? **Ответ:** его units/численная magnitude отличаются от data loss и определяют effective balance.

**Практика:** вычислите Laplacian penalty для $(0,1,1)$ и $(0,0,0)$ на цепочке. Постройте rank-1 matrix с произвольно большой norm. Для простой decay equation выпишите data/boundary/residual losses и обсудите, какие measurements нужны для оценки неизвестного $k$.
