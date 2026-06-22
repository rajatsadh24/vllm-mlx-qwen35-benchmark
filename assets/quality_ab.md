# 4-bit KV cache — quality A/B (greedy decode)

- prompts: **10** · identical outputs: **10/10** · mean similarity: **1.000** · mean prefix-agreement: **1.000**
- outputs with similarity < 0.85: **0/10**

| # | prompt | identical | similarity | prefix | tok fp16/4bit |
|---|--------|-----------|-----------|--------|----------------|
| 0 | What are the three laws of thermodynamics? Explain each one briefly. | yes | 1.000 | 1.00 | 493/493 |
| 1 | Write a Python function that checks if a string is a palindrome. Inclu | yes | 1.000 | 1.00 | 196/196 |
| 2 | Explain the difference between a process and a thread, with one exampl | yes | 1.000 | 1.00 | 768/768 |
| 3 | A farmer has 17 sheep. All but 9 run away. How many are left? Explain  | yes | 1.000 | 1.00 | 312/312 |
| 4 | Summarize the main causes of World War I in five bullet points. | yes | 1.000 | 1.00 | 277/277 |
| 5 | Implement binary search in Python and explain its time complexity. | yes | 1.000 | 1.00 | 768/768 |
| 6 | Explain how HTTPS establishes a secure connection, step by step. | yes | 1.000 | 1.00 | 768/768 |
| 7 | Write a professional email declining a meeting invite, then explain yo | yes | 1.000 | 1.00 | 598/598 |
| 8 | Review this Python service for bugs, security, and performance:  impor | yes | 1.000 | 1.00 | 768/768 |
| 9 | You are an SRE. Analyze these production logs and write a short post-m | yes | 1.000 | 1.00 | 356/356 |