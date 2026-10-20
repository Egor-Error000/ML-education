# Адаптация LLM: PEFT, supervised tuning, preferences и оценка

В этой заметке собраны темы fine-tuning и перенесённая из старого inference тема distillation. Объединены повторы; обсуждение не означает запуск обучения.

## 1. Цели адаптации и PEFT

Pretraining учит общие закономерности на больших корпусах. Continued pretraining (CPT) продолжает language-model objective на доменных данных. SFT оптимизирует примеры под выбранную инструкционную или иную supervised-цель. Preference optimization использует предпочтения/сравнения или отдельные desirability labels. Выбор данных определяется целью: сырой текст может подходить CPT; для SFT важны формат и target objective. Книжный текст сам по себе не становится instruction-датасетом.

PEFT замораживает большую часть base model и обучает небольшое число параметров или добавочных модулей. Это уменьшает обучаемый state, но не отменяет память на frozen weights, activations, optimizer states, gradients, временные buffers и runtime overhead. Для 7B параметров только веса в FP16/BF16 занимают приблизительно `7e9 × 2 bytes ≈ 14 GB` (десятичные GB), до overhead. Универсальное «14 bytes на параметр» некорректно без явного перечисления параметров учёта.

LoRA добавляет низкоранговое обновление матрицы. При column-vector convention:

```text
W0: d_out × d_in       x: d_in × 1
A:  r × d_in           B: d_out × r
y = W0 x + s B A x     s = alpha / r
trainable params = r(d_in + d_out)
```

Это факторизует обновление `ΔW=sBA`, ранг которого не выше `r`. На старте типичная реализация инициализирует `A` случайно (часто Kaiming-uniform) и `B=0`, поэтому исходная функция слоя сохраняется. Это не универсальное правило для всех вариантов реализации. Merge вычисляет эффективные веса `W0+ΔW`; он может убрать отдельный adapter path в подходящем inference, но не гарантирует фиксированное ускорение или нулевую задержку.

## 2. LoRA: проверенный первый шаг

Микро-пример из исходного PDF: `W0=[[1,2],[3,4]]`, `x=[1,2]ᵀ`, `A=[0.1,0.2]`, начальный `B=[0,0]ᵀ`, target `t=[6,10]ᵀ`, scale `s=1`, loss `L=½||y−t||²`.

```text
W0x = [5,11]ᵀ; Ax = 0.5; BAx = [0,0]ᵀ
y = [5,11]ᵀ; e=y−t=[−1,1]ᵀ; L=1
∂L/∂B = e(Ax) = [−0.5, 0.5]ᵀ
∂L/∂A = Bᵀ e xᵀ = 0
```

При SGD с `η=0.1`: `B_new=[0.05,−0.05]ᵀ`, `A_new=A`, `BAx=[0.025,−0.025]ᵀ`, `y_new=[5.025,10.975]ᵀ`, `L_new=½(0.975²+0.975²)=0.950625`. Следовательно, в первом шаге `A` не получает градиент, потому что `B=0`; после обновления `B` это обычно меняется. Число исправлено вручную по размерностям и арифметике (C01).

DoRA разделяет направление и magnitude весов; magnitude применяется в соответствии с принятой нормировкой по выходному каналу. Ось нормы и broadcasting нужно выводить из формы матрицы, не переносить вслепую формулу для другой convention. AdaLoRA перераспределяет бюджет ранга по матрицам/слоям и сокращает менее полезные компоненты; «создание ранга из ничего» и гарантированный прирост качества — неверные обещания. QLoRA хранит frozen base в квантованном виде (часто NF4), вычисления и trainable adapter имеют собственные dtype. NF4 — формат представления весов, не обещание 4-bit для всех activations и состояний. Double quantization дополнительно экономит память на quantization constants; paged optimizer помогает пиковому управлению памятью.

Другие PEFT: adapters добавляют модульные преобразования; prefix tuning вводит обучаемые K/V-представления по слоям; prompt tuning добавляет обучаемые soft vectors к входу; BitFit обновляет bias там, где он есть. Термины и размещение параметров зависят от архитектуры. LoRA+ обсуждён только обзорно.

## 3. SFT, masking и preference learning

В causal LM labels обычно сдвинуты относительно входа для предсказания следующего токена. Assistant-only loss mask — распространённая конфигурация, но не единственный корректный objective. Проверяют сдвиг labels, EOS, padding mask, границы packing и то, какие позиции входят в loss. Размеры вроде «1000 SFT и 500 DPO» в обсуждениях — иллюстрации, не минимумы и не гарантии.

В классическом RLHF могут участвовать policy, reference policy, reward model и value function — логические роли PPO-схемы, а не обязательные четыре независимые полные копии на каждом этапе. PPO оптимизирует policy с контролем обновления/регуляризацией; точное устройство зависит от реализации.

DPO оптимизирует предпочтительные/непредпочтительные пары с reference-relative objective. IPO меняет целевую формулировку и не гарантирует отсутствия переобучения. KTO использует отдельные desirable/undesirable labels и reference-relative оценку, а не просто «среднее качество ответов». ORPO сочетает preference-сигнал с supervised objective в собственной схеме. Эти методы нельзя сводить к утверждению, что все требуют одинаковый pairwise dataset. (C06–C07.)

## 4. Distillation, merging и оценка

Distillation переносит поведение teacher в student через выбранные targets/objective. Нужно учитывать направление KL, температуру и её квадратный коэффициент в типичных soft-target формулах, объём/разнообразие данных и capacity student. Teacher-generated data и RLAIF относятся к источнику/обработке supervision, а не являются синонимом distillation.

Model soup усредняет совместимые checkpoints; эффект зависит от совместимости и проверки. TIES trim-ит малые task deltas, агрегирует знаки и сливает согласованные изменения; sign election стандартного метода опирается на агрегированные значения после trimming, а не всегда на простое голосование моделей. DARE drop/rescale-ит элементы task delta и не является универсальным решателем конфликтов. Усреднение полных `BA`-дельт и отдельных факторов `A`/`B` неэквивалентно. Hot-swapping adapter, merge одного adapter в base и объединение нескольких task models — разные операции.

Evaluation должна сочетать метрики под задачу: correctness, instruction following, factuality, format validity, retrieval и task success. Perplexity полезна для language-model objective на соответствующем корпусе, но не измеряет всю полезность assistant. Human/judge preference требует rubric, blind/order-swapped comparisons, tie option и человеческой калибровки. Высокий benchmark score сам по себе не доказывает contamination; JSON validity не доказывает семантическую правильность. Не переносить общие проценты alignment tax или неподтверждённые benchmark-цифры как факты. (C08–C09, C13.)

## Прогресс и происхождение

Вопросы пользователя, сохранённые по смыслу: почему в первом шаге LoRA градиент `A` равен нулю; как согласовать shapes `A/B` и ось magnitude в DoRA; чем различаются NF4 storage и compute dtype; когда уместны CPT, SFT и preference learning.

- S2, `chat-План изучения темы.txt`: строки 4–165 (PEFT/LoRA+), 169–500 и 1817–3404 (LoRA/LLM), 3413–4587 (target modules/DoRA), 4591–5453 (AdaLoRA), 5457–5549 (QLoRA). Длинный вставленный блок USER 501–1816 не считать самостоятельным прохождением. [Архив](../../../archive/study-sources/chat-%D0%9F%D0%BB%D0%B0%D0%BD%20%D0%B8%D0%B7%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F%20%D1%82%D0%B5%D0%BC%D1%8B.txt).
- S3, `chat-Изучение Fine-tuning LLM.txt`: строки 49–224 (прочие PEFT), 228–339 (SFT/masking), 343–423 (RLHF/PPO), 428–505 (CPT/SFT context), 509–673 (DPO/IPO/KTO/ORPO), 719–816 (merging), 820–918 (evaluation). [Архив](../../../archive/study-sources/chat-%D0%98%D0%B7%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5%20Fine-tuning%20LLM.txt).
- S4, `chat-План изучения темы(1).txt`: строки 3445–4767 (distillation и связанные продолжения; детали о synthetic feedback см. lifecycle/application note). [Архив](<../../../archive/study-sources/chat-%D0%9F%D0%BB%D0%B0%D0%BD%20%D0%B8%D0%B7%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F%20%D1%82%D0%B5%D0%BC%D1%8B%281%29.txt>).
- S1, `blank.pdf`: страницы 2–3 и 8–9 (LoRA placement/merge), 15–17 (backprop и числовой шаг). [Архив PDF](../../../archive/study-sources/blank.pdf).

Обсуждение девяти пунктов старого fine-tuning маршрута зафиксировано, но запуск SFT/DPO и самостоятельный расчёт не подтверждены. Проверить себя: вывести формы `BA`, посчитать первый шаг выше без подсказки и объяснить, когда CPT предпочтительнее SFT.
