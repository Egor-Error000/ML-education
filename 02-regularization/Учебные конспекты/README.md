# Регуляризация: подробный учебный курс

[Версия для чтения с набранными формулами](<../../Читать конспекты.html#course-2-0>) · работает без интернета.

[План](<../План — регуляризация.md>) · [Общее оглавление](<../../Учебные конспекты.md>)

14 пунктов плана объединены в 10 глав. Группировка сохраняет все методы и практику; близкие доменные темы раскрыты вместе. Отдельного полного чата по регуляризации нет: это дополнение по плану, с точечными связями с Transformer/CV/PEFT/RL из источников S1–S7.

1. [А — Обобщение и теория](<А — Обобщение, bias-variance и ограничения теории.md>). Пункт 1: risk, bias–variance, MAP, Gaussian/Laplace priors, VC/Rademacher/PAC-Bayes и ограничения выводов.
2. [Б — Штрафы и AdamW](<Б — L1, L2, группы и AdamW.md>). Пункт 2: ridge/lasso/Elastic Net/Group Lasso, geometry, proximal step, scaling и decoupled decay.
3. [В — Training trajectory](<В — Early stopping, learning rate и усреднение весов.md>). Пункт 3: stopping, spectral filtering, warmup/decay, SWA/EMA и BN state.
4. [Г — Стохастические методы](<Г — Dropout, DropConnect, DropBlock и случайные ветви.md>). Пункт 4: inverted scaling, ensemble limits, masks, stochastic depth, Shake-Shake/ShakeDrop.
5. [Д — Нормализации](<Д — Нормализации, оси и train-eval statistics.md>). Пункт 5: BN/LN/IN/GN, running state, Batch Renorm, RMSNorm, Weight Norm/Standardization.
6. [Е — Устойчивость и consistency](<Е — Adversarial training, consistency и spectral norm.md>). Пункт 6: FGSM/PGD, VAT, Π-model/Mean Teacher, spatial alignment и operator norm.
7. [Ж — Data-level и CV](<Ж — Label smoothing, MixUp и CV-аугментации.md>). Пункты 7–8: soft labels, MixUp/Manifold, CutMix/Cutout/Erasing, policies и TV.
8. [З — NLP и RL](<З — NLP, R-Drop и ограничения политики в RL.md>). Пункты 9–10: tying, variational/token/embedding noise, attention entropy/R-Drop, sequence smoothing, entropy/KL/clipping.
9. [И — Структурные ограничения](<И — Структурные prior, графы, физика и PEFT.md>). Пункты 11 и 13: temporal/seasonal, Laplacian/physics, rank/bottleneck и PEFT boundaries.
10. [К — Оптимизация и выбор](<К — SAM, Lookahead и протокол выбора регуляризации.md>). Пункты 12 и 14: SAM/flatness, Lookahead, diagnosis, strength sweep, interactions и итоговый protocol.

## Порядок и практические результаты

Ядро — А–Д, базовая часть Ж и протокол К. Е, продвинутые noise methods, PAC-Bayes и structural constraints — углубление. Читать сначала идею и objective, затем конкретный update, числовой пример и ограничения. Каждая глава содержит контрольные вопросы с ответами и самостоятельную практику.

Семь обязательных практик исходного плана покрыты так: penalties/soft-threshold — Б; AdamW — Б; stopping — В; dropout scaling/train-eval — Г; axes/running statistics — Д; MixUp/CutMix/R-Drop — Ж/З; итоговая диагностика — К. Все математические выражения оформлены LaTeX.

Используйте один [data split и pipeline](../../01-data/Учебные%20конспекты/README.md), baseline, одинаковый budget и отдельный test. Итог — объяснение выбранного механизма и проверенное сравнение, включая отсутствие выигрыша, если оно наблюдается. Тексты не означают, что вы уже выполнили experiments.

Подробные смежные механизмы: [LoRA/PEFT и alignment](../../06-llm/02-adaptation/Учебные%20конспекты/README.md), [agentic RL](../../06-llm/05-lifecycle/Учебные%20конспекты/README.md), [CV](../../05-computer-vision/Учебные%20конспекты/README.md).
