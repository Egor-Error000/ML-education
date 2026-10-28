# Обучение агентов

[Модуль «LLM и языковые модели»](<../README.md>) · [Сценарии глав](<План.md>)

- [Постановка обучения агента и baseline](<agent-task.md>) — основа ветки. Как выбрать вмешательство по измеренному типу ошибки?
- [Среда, observations, actions и horizon](<agent-environment.md>) — основа ветки. Как описать доступную агенту информацию и последствия действия?
- [Оценка траекторий и воспроизводимость](<agent-evaluation.md>) — основа ветки. Что считать успехом целой попытки и как учитывать retries и стоимость?
- [Behavior cloning и SFT на демонстрациях](<agent-demonstrations.md>) — основа ветки. Как из траектории получить targets для обучения политики?
- [Distribution shift, recovery и DAgger](<agent-recovery.md>) — основа ветки. Как научиться действовать в состояниях, порождённых собственными ошибками?
- [Обучение через взаимодействие: PPO и GRPO](<agent-rl.md>) — основа ветки. Как отложенный reward меняет вероятность последовательности действий?
- [Углубление: reasoning и верификация](<agent-reasoning.md>) — углубление. Когда дополнительный поиск и проверка улучшают результат?
- [Обучение рассуждению: данные, RL и инфраструктура](<agent-reasoning-training.md>) — углубление. Как построить и проверить обучающий цикл для многошаговых задач?
- [Углубление: память и длинный горизонт](<agent-memory.md>) — углубление. Какую информацию сохранить и как измерить её полезность?
- [Углубление: предпочтения траекторий](<agent-preferences.md>) — углубление. Как сравнивать полные попытки с разным бюджетом и исходом?
- [Углубление: роли и обучение координации](<agent-coordination.md>) — углубление. При каких условиях несколько участников дают измеримый выигрыш?

**Оценка агента перед обучением:** [Постановка обучения агента и baseline](<agent-task.md>) → [Среда, observations, actions и horizon](<agent-environment.md>) → [Оценка траекторий и воспроизводимость](<agent-evaluation.md>) → [Behavior cloning и SFT на демонстрациях](<agent-demonstrations.md>) → [Distribution shift, recovery и DAgger](<agent-recovery.md>) → [Обучение через взаимодействие: PPO и GRPO](<agent-rl.md>)

Вход в маршрут: [Состояние, orchestration и граф управления](<../05-applications/app-state.md>), [Оценка retrieval, ответов и task success](<../05-applications/app-evaluation.md>), [SFT: демонстрации, шаблоны и обучающие цели](<../03-adaptation/adapt-sft.md>).

**Обучение рассуждению и инфраструктура:** [SFT: демонстрации, шаблоны и обучающие цели](<../03-adaptation/adapt-sft.md>) → [Оценка траекторий и воспроизводимость](<agent-evaluation.md>) → [Обучение через взаимодействие: PPO и GRPO](<agent-rl.md>) → [Углубление: reasoning и верификация](<agent-reasoning.md>) → [Обучение рассуждению: данные, RL и инфраструктура](<agent-reasoning-training.md>)

Вход в маршрут: [Обучение: targets, loss и обновление весов](<../01-foundations/lm-training.md>), [Оценка retrieval, ответов и task success](<../05-applications/app-evaluation.md>), [Адаптация: задача, базовая система и бюджет](<../03-adaptation/adapt-choice.md>), [Среда, observations, actions и horizon](<agent-environment.md>), [Distribution shift, recovery и DAgger](<agent-recovery.md>).
