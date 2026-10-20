# Контекст для продолжения обучения

Я изучаю материал по обновлённым планам в `ML edu plans`, а не обязательно создаю проект. Прошу меньше глав в одном ответе, но подробнее механизм: одна тема, 2–4 смысловых раздела, конкретные shapes/маленькие матрицы и возможность задать вопрос. Не переходить автоматически к следующей теме по одному длинному объяснению.

Актуальные заметки: [основы LLM](../01-foundations/notes/llm-foundations.md), [адаптация](../02-adaptation/notes/llm-adaptation.md), [инференс](../03-inference/notes/llm-inference.md), [приложения и RAG](../04-applications/notes/llm-applications.md), [DSpark](../03-inference/notes/llm-inference-research.md). Статусы и происхождение — в [реестре](PROGRESS.md).

## Что уже обсуждали

- PEFT, LoRA/QLoRA, DoRA и AdaLoRA подробно; LoRA+ только краткий обзор.
- Other PEFT, SFT/masking, RLHF, DPO/IPO/KTO/ORPO, merging и evaluation — теория старых девяти пунктов fine-tuning пройдена по обсуждению.
- Inference A/B, C.1–C.4, D.1–D.6 и E.1–E.2 — теоретически разобраны.
- Distillation перемещена из старого inference A.5 в adaptation §12; generic quantization описана один раз в inference, QLoRA остаётся в adaptation.
- RNN/Transformer/QKV/hidden states повторялись; не начинать их с нуля без точечного вопроса.
- Есть исследовательские обсуждения DSpark, speculative beam search, синтетики/teacher annealing; не все утверждения чатов проверены.

Обсуждение и показанный ассистентом код не означают, что я запускал обучение, serving или создавал агента. Практика и самостоятельное понимание требуют отдельного подтверждения; не писать «мы обучили/задеплоили» без моего результата.

## Что исправить коротко

Сверяться с [поправками](CORRECTIONS.md): shapes и арифметика LoRA, softmax-backprop, memory accounting, greedy-temperature, schema vs semantic tool correctness, merging/switching и значение perplexity. Не использовать выдуманные проценты качества и универсальное правило Q5/Q6 для tools.

## Откуда продолжать

Основная inference-цепочка закончилась на **E.2**. После коротких поправок продолжать **E.3: cache готовых ответов, semantic caching, keys/invalidation/correctness**, затем **E.4: production monitoring/drift/regression evaluation**. Prompt caching уже обсуждён.

Для adaptation следующий новый блок — **RLAIF/Constitutional AI**, затем multi-objective alignment; имеются ранние упоминания, но систематического прохождения нет. LoRA+ можно углубить отдельно.

Agentic training не засчитывать по знакомству с LangGraph/AutoGen. Начать с среды, action/observation/reward/termination и оценки траекторий, если выбрана эта ветка.

Источники и точные уровни покрытия: [реестр](PROGRESS.md) и [план миграции](MIGRATION_PLAN.md). При расхождении стартовой старой сводки и более поздних сообщений использовать поздний подтверждённый материал. Статус других ML/CV-планов по этим экспортам не установлен.
