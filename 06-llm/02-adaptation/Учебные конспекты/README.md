# Адаптация LLM — подробные конспекты по вашим чатам

[Версия для чтения с набранными формулами](<../../../Читать конспекты.html#course-7-0>) · работает без интернета.

[Актуальный план](<../План — адаптация LLM.md>) · [Прежняя сводная заметка](../notes/llm-adaptation.md) · [Основы LLM](../../01-foundations/Учебные%20конспекты/README.md)

14 пунктов плана объединены в **10 учебных глав**. Главы А–З и К основаны на обсуждениях; глава И явно дополняет темы, которых в чатах нет в систематическом виде. Каждая глава указывает исходный диапазон, исправления и углубления. Порядок букв пропускает Й; последняя глава обозначена К.

- [А. Выбор адаптации, PEFT и память](<А — Выбор адаптации, PEFT и память обучения.md>) — 1, 14A.
- [Б. LoRA: placement, shapes, target modules, два шага](<Б — LoRA, target modules и два шага обучения.md>) — 2, 14C.
- [В. DoRA, AdaLoRA, LoRA+, QLoRA](<В — DoRA, AdaLoRA, LoRA+ и QLoRA.md>) — 2, 7, 14C.
- [Г. Adapters, soft prompt, prefix K/V, BitFit](<Г — Adapters, prefix tuning, soft prompts и BitFit.md>) — 3.
- [Д. SFT, книги как источник данных, masks и packing](<Д — SFT, данные из книг, masks и training objective.md>) — 4, 14B.
- [Е. RLHF/PPO, DPO, IPO, KTO, ORPO](<Е — RLHF, PPO, DPO, IPO, KTO и ORPO.md>) — 5–6, 14D.
- [Ж. Model soup, TIES, DARE, merge и switching](<Ж — Model soup, TIES, DARE, merge и switching.md>) — 8, 14E.
- [З. Evaluation, PPL, judge и экспорт](<З — Evaluation, perplexity, judge и экспорт.md>) — 9, 14E.
- [И. AI feedback, несколько целей и continual learning](<И — AI feedback, несколько целей и continual learning.md>) — 10–11, 13; дополнительный материал.
- [К. Дистилляция и teacher annealing](<К — Дистилляция, teacher annealing и обучение на своих траекториях.md>) — 12; перенесённая тема inference A.5.

## Исходники

- **S1:** [blank.pdf](../../../archive/study-sources/blank.pdf), страницы 2–3, 8–9, 15–17: LoRA placement и числовой шаг. Исправлены транспонированные формы и arithmetic.
- **S2:** [chat-План изучения темы.txt](<../../../archive/study-sources/chat-План изучения темы.txt>), строки 4–165, 169–500, 1817–3404, 3413–4587, 4591–5453, 5457–5549. Большая вставка USER 501–1816 не засчитывается как новое объяснение.
- **S3:** [chat-Изучение Fine-tuning LLM.txt](<../../../archive/study-sources/chat-Изучение Fine-tuning LLM.txt>), 49–224: прочие PEFT; 228–339: SFT; 343–423: RLHF; 428–505: CPT/SFT; 509–673: preferences; 719–816: merging; 820–918: evaluation.
- **S4:** [chat-План изучения темы(1).txt](<../../../archive/study-sources/chat-План изучения темы(1).txt>), 3445–4767: distillation и ваши вопросы о teacher на ранних этапах обучения.

Точные origin ranges и история обсуждений — в [карте миграции](../../migration/MIGRATION_PLAN.md). [Поправки](../../migration/CORRECTIONS.md) C01/C03/C05–C09/C13 применены к учебному изложению.

## Как читать и проверять

Для основного маршрута: А → Б → В (сначала QLoRA, остальные варианты по вопросу) → Д → Е/DPO → З. Г и Ж полезны для сравнения механизмов; И — новая ветка, К — подробное продолжение реально заданных вопросов. Это способ чтения, а не сокращение содержания плана.

Особые самостоятельные проверки: формы $W_0+sBA$, второй шаг с ненулевым grad_A, broadcasting m в DoRA, один DPO loss, контрпример factor averaging и masked SFT batch. Затем выбранная практика: training config → independent evaluation → export/reload → повторная оценка.

Ни один демонстрационный пример не объявлен результатом вашего запуска. Созданные материалы не повышают отметки «понимание» и «практика» в историческом реестре.
