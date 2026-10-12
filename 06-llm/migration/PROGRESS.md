# Реестр переноса прогресса

Срез: 12 октября 2026. Источники/карта переноса — в [плане миграции](MIGRATION_PLAN.md). Это подтверждение **обсуждения**, не сертификат компетенции. Для всех тем ниже практический запуск пользователем не подтверждён предоставленными экспортами; snippets/числа ассистента не считаются его результатами.

Перенесённый материал: [основы](../01-foundations/notes/llm-foundations.md), [адаптация](../02-adaptation/notes/llm-adaptation.md), [инференс](../03-inference/notes/llm-inference.md), [применения](../04-applications/notes/llm-applications.md), [DSpark case study](../03-inference/notes/llm-inference-research.md). Для каждой темы фиксируются три независимых оси: **обсуждение** (уровень покрытия, приведённый ниже), **понимание** (не проверено / частично подтверждено конкретным ответом пользователя / проверено самостоятельным объяснением), **практика** (не подтверждена / есть пользовательский артефакт / воспроизведена пользователем). По этим экспортам ни одна тема не подтверждена как полностью освоенная самостоятельным объяснением; отдельные корректные вопросы/замечания — свидетельство частичного понимания, а не mastery. Практика не подтверждена: ответы ассистента, вставленные примеры и фрагменты кода не считаются запуском, обучением, deployment или созданием агента.

## LLM foundations

- **FOUND-RNN:** подробно обсуждены recurrence, hidden state, shapes и attention. S1, стр. 4–6. Следующее: объяснить, что recurrent state не хранит всю историю без потерь; C04.
- **FOUND-ATTENTION:** подробно обсуждены Transformer, Q/K/V, multi-head, causal/cross attention. S1, стр. 6–13; S2, строки 1817–3404. Следующее: согласовать shapes и корректный softmax-backprop; C02/C04.
- **FOUND-HIDDEN:** подробно обсуждены три блока, hidden states и LM head. S1, стр. 13–15. Следующее: различать token-ID, hidden state и logits; не фиксировать «семантическую роль» каждого слоя заранее.

## Адаптация

- **FT-PEFT** (новый §1): подробный разбор. S2, 4–72; повтор в S3, 1–48. Проверить accounting памяти без универсальных 14 bytes/parameter; C05.
- **FT-LORA** (§2/§14C): подробный разбор и показанный ассистентом числовой шаг. S2, 169–500, 1817–3404, 3413–3488; S1, 15–17. Самостоятельный расчёт не подтверждён. Следующее: C01 и повтор первого gradient step.
- **FT-QLORA** (§2/§14C): подробный разбор. S2, 5457–5549; более раннее объяснение было длинным, и пользователь сообщил, что не дочитал часть. Поздний отдельный разбор имеет приоритет. Следующее: storage vs compute dtype, C03/C05.
- **FT-DORA:** подробный разбор с пользовательским вопросом о dimensions и magnitude. S2, 3489–4587. Следующее: per-output magnitude/broadcasting и norm axis; C01.
- **FT-ADALORA:** подробный разбор. S2, 4591–5453. Следующее: бюджет и pruning singular directions без обещания произвольного «роста ранга из ничего».
- **FT-LORA-PLUS:** обзор в S2, 76–165; отдельное углубление/эксперимент не найдены. Не включать в «детально завершено» по одной сводке.
- **FT-OTHER-PEFT** (§3): теория разобрана. S3, 49–224; prefix/BitFit уточнялись по вопросу пользователя. Следующее: различать prompt/prefix и наличие bias в конкретной архитектуре.
- **FT-SFT** (§4/§14B): теория разобрана. S3, 228–339; данные/CPT дополнены в 428–505. Следующее: manual labels/mask/EOS; C06.
- **FT-RLHF** (§5/§14D): теоретическая схема разобрана. S3, 343–423. Полный вывод PPO и обучение не подтверждены. Следующее: reward/KL/value roles; C07.
- **FT-DPO** (§6/§14D): теория DPO/ORPO; IPO/KTO разобраны подробнее. S3, 509–673. Следующее: числовой preference-loss и правильный reference KTO; C07.
- **FT-MERGING** (§8/§14E): теория разобрана. S3, 719–816. Следующее: task-vector merging, sign election, adapter vs full model; C08.
- **FT-EVAL** (§9/§14E): теория разобрана. S3, 820–918. Следующее: C09, ties/position bias, task success и PPL как разные метрики.
- **FT-DISTILL** (§12, бывший inference A.5): подробная теория. S4, 3445–3913, 4396–4767. Следующее: KL direction/T² и teacher-student compatibility; отдельные industry claims — C13.
- **FT-RLAIF** (§10): упоминания/обзор Constitutional AI и self-improvement в S4, 3917–4395. Отдельный систематический разбор нового раздела не найден. Следующее: источник feedback, rubric, bias и проверка.
- **FT-MULTIOBJECTIVE** (§11): отдельное прохождение не найдено.
- **FT-CONTINUAL** (§13): forgetting обсуждался в PEFT/industry-вопросах; replay/EWC и последовательный эксперимент нового блока не подтверждены.

## Инференс

- **INF-A1–A3:** quantization formats/PTQ/QAT/GPTQ/AWQ подробно обсуждены. S4, 4–2649; refresh S3, 674–715. Не повторять как новые темы; исправить C03/C05/C10.
- **INF-A4:** pruning подробно обсуждён. S4, 2653–3441. Не считать гипотезы о закрытом production фактами; C13.
- **INF-A5 → FT-DISTILL:** перенести прогресс к новому месту; в inference оставить alias и ссылку, не вторую запись освоения.
- **INF-A6:** KV-cache/MHA/MQA/GQA разобраны. S4, 4771–4922. Следующее: вручную посчитать bytes/cache и влияние длины, C10.
- **INF-B1–B4:** теоретически разобраны batching, PagedAttention, engines, ONNX/PyTorch/Triton. S4, 4926–6435. Показанные config/benchmarks не подтверждают выполненный deployment. Следующее: отделить concept от API version.
- **INF-C1:** speculative decoding подробно разобран, включая beam search и draft-quality вопрос. S4, 6439–8230. Следующее: acceptance/residual sampling, budget, C12; DSpark — research alias, C13.
- **INF-C2:** теория structured output разобрана. S5, 120–242. Следующее: C11 и отказ/прерывание/schema support.
- **INF-C3:** sampling подробно разобран, пользователь верно заметил проблему greedy-temperature. S5, 246–413. Следующее: окончательная точная формулировка C12.
- **INF-C4:** long context/prefix caching подробно разобраны. S5, 417–1034. Следующее: механизм отдельно от TTL/цен/доступности API; C14.
- **INF-D1–D3:** orchestration/LangChain/LangGraph и локальные runtimes разобраны. S5, 1038–2822. Следующее: C14; не присваивать agent training.
- **INF-D4:** начало в S5, 2826–3041; завершённый теоретический разбор S6, 4–178. Следующее: schema validation vs semantic correctness и execution lifecycle; C11.
- **INF-D5:** RAG подробно разобран, включая BM25, retrieval embeddings, hybrid и multi-vector. S6, 182–1373. Следующее: отделить retrieval/generation metrics, уточнить неподтверждённые числа C13.
- **INF-D6:** framework concepts/coordination/termination разобраны. S6, 1377–2197. Следующее: runtime role vs learned policy; code/API — C14.
- **INF-E1:** deployment/cold-start теория разобрана. S6, 2201–3009. Следующее: отличать readiness/liveness/startup и модель в volume от прогрева.
- **INF-E2:** autoscaling/queue/latency теория разобрана. S6, 3013–3904. Это последняя основная тема inference в предоставленной цепочке. Следующее: не принимать примерные thresholds за универсальные настройки.
- **INF-E3:** prompt cache и краткое различение semantic cache уже упоминались в S5; отдельного подробного прохождения E.3 не найдено. Продолжить с answer/semantic cache, ключей, invalidation и корректности повторного использования.
- **INF-E4:** оценка ответов есть в FT-EVAL; отдельного мониторинга production, drift, alerting/regression loop не найдено. Частичный prerequisite, а не «нулевые знания».

## Agentic training

Orchestration, роли и loops имеют прогресс INF-D1–D6. Введение обучающей среды, trajectory SFT, DAgger, agentic RL, PRM/ORM, memory training и multi-agent training как отдельные разделы нового lifecycle не подтверждены. Упоминания RL/GRPO в S4 не означают прохождение этой ветки.

Начать с постановки задачи и оценки траекторий. Если нужен только теоретический маршрут, использовать игрушечные среды/траектории на бумаге; не приписывать создание проекта.

## Остальные планы

Данные, категории, кластеризация, CV и регуляризация были переработаны, но предоставленные шесть документов не дают достаточного evidence для переноса их тематического прогресса. Упоминания embeddings, clusters или BN не означают завершённый курс. Статус: **не установлен по этим источникам**.

## Следующий checkpoint

Обязательные короткие поправки C01/C02/C09/C11/C12, затем E.3 → E.4. Альтернатива для ветки обучения: RLAIF → multi-objective; agentic training начать отдельно. Заново проходить A–D inference и §1–9 старого fine-tuning не требуется.
