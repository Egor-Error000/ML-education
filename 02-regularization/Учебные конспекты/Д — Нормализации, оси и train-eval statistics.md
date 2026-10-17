# Д. Нормализация: что усредняется и почему это важно

[Оглавление](README.md). Пункт 5 плана. **Основа:** блоки Transformer из S2 и backbones CV из S7; систематическое сравнение семейств — дополнение. Нормализация прежде всего меняет оптимизацию, а шум статистик может дополнительно регуляризовать.

## 1. Общая формула и разные множества

Для выбранного множества элементов $\mathcal S$:

$$
\mu=\frac1{|\mathcal S|}\sum_{j\in\mathcal S}x_j,
\quad
v=\frac1{|\mathcal S|}\sum_{j\in\mathcal S}(x_j-\mu)^2,
\quad
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+\beta_i.
$$

Главное различие — какие элементы принадлежат $\mathcal S$ и как разделяются affine parameters. $\epsilon>0$ стабилизирует деление, поэтому variance нормированного output не обязана быть точно 1. Affine layer может затем намеренно изменить mean и scale.

Для тензора $N,C,H,W$ **BatchNorm** считает statistics по $N,H,W$ отдельно для каждого channel. **InstanceNorm** — по $H,W$ для каждого example/channel. **GroupNorm** — по пространству и channels выбранной группы внутри одного example. **LayerNorm** — по объявленным normalized dimensions; в Transformer $N,T,D$ обычно только по $D$ отдельно для каждой пары example/token. Нельзя без config считать, что LN всегда нормирует всё изображение.

Пример с одним channel и двумя изображениями, каждый из двух pixels: $(1,3)$ и $(5,7)$. BN использует все четыре значения, mean 4 и variance 5. IN для первого использует mean 2, variance 1, для второго mean 6, variance 1. Без epsilon/affine IN даёт $(-1,1)$ в обоих examples, стирая различие общей яркости. BN сохраняет его в normalized values. Это показывает, почему выбор norm меняет доступную информацию.

## 2. BatchNorm на training и inference

На training BN использует mini-batch statistics и обновляет running estimates. На eval типичный режим использует накопленные estimates, а не statistics текущего входного batch. Конкретные variance estimators и momentum convention зависят от library; не переносите число momentum из optimizer в BN по сходству названий.

Малый или сильно correlated batch даёт шумные statistics. Gradient accumulation не эквивалентен большому BN batch: несколько forward по microbatches не объединяют их statistics автоматически. SyncBatchNorm может синхронизировать statistics между devices, но не решает все проблемы не-i.i.d. data и имеет communication cost.

Шум batch statistics способен давать regularizing effect, но историческое объяснение через internal covariate shift не является единственным установленным механизмом. Исследуется также сглаживание optimization landscape; см. [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604). Нельзя считать BN простым заменителем L2 или dropout.

После freezing backbone нужно отдельно решить режим BN. Замороженные weights не обязательно означают замороженные running statistics. При fine-tuning с маленьким domain-specific batch их неосторожное обновление может ухудшить pretrained representation.

## 3. Batch Renormalization

Batch Renorm вводит corrections между текущими batch statistics и running estimates. Упрощённо:

$$
\widehat x_i=\frac{x_i-\mu_B}{\sigma_B},\quad
r=\operatorname{clip}\left(\frac{\sigma_B}{\sigma},\frac1{r_{\max}},r_{\max}\right),
\quad
d=\operatorname{clip}\left(\frac{\mu_B-\mu}{\sigma},-d_{\max},d_{\max}\right),
$$
$$
y_i=\gamma(r\widehat x_i+d)+\beta.
$$

Здесь standard deviations включают stabilizer по реализации, а $r,d$ используются с остановленным gradient; clipping limits могут меняться по schedule. Это не просто замена batch mean running mean во всех derivatives. Идея — уменьшить mismatch training/inference при сохранении управляемой оптимизации. [Batch Renormalization](https://arxiv.org/abs/1702.03275).

## 4. RMSNorm и нормализация weights

RMSNorm не центрирует:

$$
\operatorname{RMS}(x)=\sqrt{\frac1D\sum_{j=1}^D x_j^2+\epsilon},
\qquad
y_i=\gamma_i\frac{x_i}{\operatorname{RMS}(x)}.
$$

Для $(1,3)$ без epsilon LN даёт $(-1,1)$, RMSNorm — $(1/\sqrt5,3/\sqrt5)\approx(0.4472,1.3416)$. Mean не удалена. Это изменение преобразования, а не лишь сокращённая запись LN. Применимость оценивается для конкретной architecture/training recipe.

Weight Normalization параметризует vector weights как $w=g\,v/\|v\|_2$: scale и direction оптимизируются отдельно. Здесь нормируются parameters, а не batch activations. Weight Standardization центрирует и масштабирует coefficients каждого filter, например по input-channel и kernel axes. Его часто сочетают с activation normalization; WS и GN решают разные части преобразования.

Нормирование weights не гарантирует, что вся network имеет Lipschitz constant 1. Для этого нужны bounds operator norms и учёт nonlinearities, residual sums и других layers. Spectral norm разобрана в следующей главе.

## 5. Контроль и практика

1. Почему accumulation не увеличивает BN batch автоматически? **Ответ:** statistics считаются при каждом forward microbatch отдельно.
2. Равна ли RMSNorm LayerNorm без bias? **Ответ:** нет; RMSNorm также не вычитает mean.
3. Что может меняться при frozen weights? **Ответ:** state вроде BN running statistics, если оставлен training mode.

**Практика:** на shape $2\times4\times2\times2$ выпишите axes для BN, IN, GN с двумя groups и LN с двумя разными `normalized_shape`. Вручную нормируйте приведённые pixels. Проверьте train/eval outputs и запишите, какие statistics сохраняются в checkpoint.
