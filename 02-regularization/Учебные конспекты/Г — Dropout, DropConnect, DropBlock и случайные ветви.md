# Г. Шум в forward pass: какие части сети временно выключаются

[Оглавление](README.md). Пункт 4 плана и практика dropout. **Происхождение:** дополнение; в чатах LoRA/Transformer dropout встречается контекстно, но семейство методов там не разобрано.

## 1. Inverted dropout

Пусть activation $h_j$ оставляется с вероятностью $q=1-p$. Для mask $m_j\sim\operatorname{Bernoulli}(q)$:

$$
\widetilde h_j=\frac{m_j}{q}h_j,
\qquad
\mathbb E[\widetilde h_j\mid h_j]=h_j,
\qquad
\operatorname{Var}(\widetilde h_j\mid h_j)=\frac{p}{1-p}h_j^2.
$$

Это inverted scaling: training activations усиливаются на $1/q$, а на eval dropout становится identity. При $p=0.5,h=2$ выход равен 0 или 4 с равной вероятностью, mean 2 и variance 4. Вероятность $p=1$ недопустима для этой формулы; нулевой keep probability требует отдельного поведения реализации.

Сохраняется expected activation, а не обязательно expected output всей nonlinear network. Поэтому выражение «точно усредняем все подсети» — слишком сильное. Dropout напоминает shared-weight ensemble, но eval identity не является точным average predictions всех возможных masks.

Train/eval mode важен. Если забыть eval, predictions становятся стохастическими; если забыть train, regularizing noise пропадает. MC-dropout намеренно оставляет stochastic passes для оценки variation, но не превращает её автоматически в калиброванную полную Bayesian uncertainty.

## 2. Разная зернистость masks

**DropConnect** маскирует weights: $\widetilde W=(M\odot W)/q$, затем применяет слой. Dropout маскирует outputs/inputs units, так что несколько связей получают общую mask. Это разные структуры noise, даже если marginal keep probability одинакова.

В feature maps соседние activations коррелируют; независимые одиночные нули иногда легко компенсируются соседями. **DropBlock** удаляет пространственные блоки feature map. Размер и вероятность seed-blocks выбирают с учётом desired drop fraction и границ; поставить dropout probability напрямую как block-centre probability обычно неверно.

Channel/spatial dropout может выключать весь channel, сохраняя одну mask по пространственным координатам. Это ещё один granularity choice, а не синоним произвольного DropBlock. Метод выбирают по структуре representation, не по общему названию «обнуление».

## 3. Stochastic depth

Для residual block $x+F(x)$ training может использовать:

$$
y=x+\frac bqF(x),\qquad b\sim\operatorname{Bernoulli}(q).
$$

На eval используется $x+F(x)$ в данной inverted convention. Исходные/конкретные implementations могут применять другую scaling convention. Identity path сохраняет передачу сигнала, а случайная residual branch меняет effective depth. Часто keep probability зависит от глубины; это hyperparameter schedule.

Нельзя удалить arbitrary слой с несовместимыми input/output shapes и считать его identity. Projection/downsampling blocks требуют отдельной обработки. Если implementation всё равно считает $F(x)$ и лишь затем умножает на ноль, экономии compute на этом step может не быть, хотя stochastic regularization сохраняется.

## 4. Shake-Shake и ShakeDrop

В двух residual branches Shake-Shake использует случайную смесь:

$$
y=x+\alpha F_1(x)+(1-\alpha)F_2(x).
$$

В training коэффициент обратного распространения может быть другим, $\beta$. Поэтому заданный backward является специальным surrogate rule, а не обычной производной forward с fixed $\alpha$. Важно не выводить градиент как будто coefficients совпадают. На eval применяется deterministic combination по выбранной convention. См. [Shake-Shake](https://arxiv.org/abs/1705.07485).

ShakeDrop сочетает Bernoulli gate и случайное масштабирование residual contribution, включая разные forward/backward coefficients, чтобы распространить идею на другие residual architectures. Это не просто другое название stochastic depth. Конкретные coefficient ranges и layer schedules берутся из [описания метода](https://arxiv.org/abs/1802.02375), а не произвольно переносятся между architectures.

## 5. Как проверять эффект

Нужен baseline с $p=0$, несколько вероятностей, один data split и comparable steps/LR. Слишком сильный noise затрудняет fit и может ухудшить generalization. Несколько noises одновременно затрудняют диагностику. Не переносите «обычно полезный dropout» на каждый pretrained backbone без validation: training regime и already existing regularization важны.

1. Зачем division на keep probability? **Ответ:** сохраняется conditional mean activation при training.
2. Точен ли ensemble interpretation для всей nonlinear network? **Ответ:** нет; равенство mean activation не переносится через любую нелинейность.
3. Чем DropConnect отличается от dropout? **Ответ:** mask относится к отдельным weights, а не к outputs units.

**Практика:** вычислите все masks для двух activations при $p=0.5$ и их expected linear output. Затем добавьте nonlinear activation и проверьте, совпадает ли expected result с eval. Нарисуйте masks Dropout/Channel/DropBlock для одной feature map и объясните различия.
