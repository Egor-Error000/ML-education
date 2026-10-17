# А. Lifecycle: как связать данные, модель, приложение и обучение

[Оглавление](README.md). Пункт1 плана.

**Связь с чатами:** S3:228–918 обсуждает SFT/preferences/evaluation; S5/S6 — runtime, RAG и deployment. Эта глава связывает уже обсуждённые механизмы и дополняет их диагностикой agentic tasks. Самостоятельное обучение агента в источниках не подтверждено.

## 1. Цикл начинается с задачи и оценки

Pretrained checkpoint — начальная точка маршрута, а не конечное приложение. Требуются данные, спецификация поведения, evaluation и интерфейсы. После deployment появляются новые inputs/errors, которые могут привести к обновлению prompt, corpus, runtime или параметров. Такое обновление снова проходит оценку.

Useful lifecycle: задача → независимые evaluation cases → baseline → diagnosis ошибок → одна интервенция → сравнение → export/integration → нагрузочная проверка → rollout/rollback → мониторинг и новые данные. Это не обязательная цепочка «каждый раз SFT→DPO→RL»: intervention выбирается по evidence.

Например, assistant неверно отвечает на новую внутреннюю инструкцию. Если инструкция отсутствует в доступных данных, RAG/обновление corpus — естественная гипотеза. Если документ найден, но его version неверна, исправляется ingestion/filtering. Если evidence правильный, но модель систематически игнорирует отрицания, рассматриваются prompt/data/model interventions. Один final failure ещё не определяет метод.

## 2. Что меняет каждое вмешательство

Prompting меняет вход и доступные instructions. Structured output ограничивает форму. Retrieval меняет доступные evidence. Tool execution добавляет внешние operations/observations. Graph меняет ordering/state/termination. SFT/PEFT меняют параметры на demonstrations. Preferences дают сигнал сравнительного качества, RL — feedback из rollouts/reward.

Compression/serving меняют representation/runtime и quality-cost trade-off, но сами по себе не объясняют, как выбирать sequence действий. Continual learning обновляет поведение по новым данным, требуя проверки прежних способностей. У каждого layer собственные failure modes и evaluation artifacts.

Задачу агентного поведения нужно отличать от single-turn formatting. Если одна tool request имеет correct syntax, но после её результата policy выбирает неверный следующий шаг, schema уже не основной bottleneck. Если tools возвращают ошибочные данные, обучение policy может закрепить работу вокруг сломанного интерфейса вместо исправления среды.

## 3. Сопоставимый baseline и стоимость

Сначала зафиксировать simple prompting/tool loop без нового training. Дать ему те же tasks, tools, context и budgets, которые будут у candidate. Если candidate использует в пять раз больше попыток, часть improvement может объясняться test-time compute, а не лучшей policy.

Общая полезность может быть сформулирована как constrained optimisation:

$$
\max_{\text{system}}\;\mathbb{E}[\mathrm{success}]
\quad\text{при}\quad
\mathbb{E}[C]\le B_C,\qquad P(T> B_T)\le\varepsilon_T.
$$

Здесь $C$ — cost, $T$ — latency, budgets задаются приложением. Такая запись помогает не скрыть failure, купленный бесконечными retries. Она не определяет автоматически численные thresholds.

Training cost также входит в выбор. Улучшение на десяти запросах может не окупать expensive tuning, если тот же effect даёт небольшой deterministic fix. Для больших repeated workloads расходы распределяются иначе; сравнивать надо в выбранном горизонте эксплуатации.

## 4. Какие артефакты связывают этапы

Data manifest, split IDs и annotation guideline задают происхождение. Model/base revision, tokenizer/template, adapters и configs определяют computation. Tool/environment schemas и versions определяют действия. Evaluation report содержит outputs, failures и cost. Deployment bundle/rollback связывает tested candidate с реально исполняемой системой.

Если новая model evaluation прошла в одном template, а production использует другой, lifecycle разорван. Если rollout переключает weights, но оставляет несовместимый retrieval space, нельзя считать это проверенной версией. Если в training попали evaluation trajectories, улучшение не доказывает перенос на новые задачи.

В текущем курсе подробный training objective находится в адаптации, serving — в инференсе, tool/RAG contracts — в приложениях. Следующие главы добавляют именно agentic постановку: environment, trajectory data, distribution shift, delayed reward и horizon.

## 5. Самопроверка и упражнение

1. Почему graph improvement не обязательно training? **Ответ:** sequencing/state может меняться при тех же weights.
2. Как понять, что нужен agentic dataset? **Ответ:** failures относятся к выбору последовательности действий/восстановлению при корректных отдельных interfaces.
3. Зачем evaluation до training? **Ответ:** задаёт independent критерий, который нельзя подменить снижением выбранного train loss.

**Практика:** взять пять условных failures и назначить layer/hypothesis: missing knowledge, wrong source version, malformed args, повтор после timeout, неверный вывод при верном evidence. Для каждого выбрать одну проверяемую intervention и результат, который опроверг бы гипотезу. Не назначать SFT/RL автоматически всем случаям.
