# План: основы LLM

Цель — уверенно читать формы и объяснять путь от последовательности токенов до распределения следующего токена. Подробный исторический конспект — в [notes](notes/README.md).

## Темы

1. Токены, embeddings, позиции и next-token objective.
2. RNN: recurrent state, градиенты и ограничения последовательного вычисления.
3. Transformer: self-attention, causal mask, encoder-decoder и decoder-only.
4. Q/K/V, multi-head attention и размерности; вывод forward/backward на маленьких матрицах.
5. Hidden states, LM head и logits; отделение архитектурных фактов от интерпретаций слоёв.

**Checkpoint:** самостоятельно вывести shapes одного attention слоя и объяснить различия token ID, hidden state и logits. Практическую реализацию отмечать отдельно от теоретического ответа.
