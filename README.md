
## Retrieval Evaluation

Built a small evaluation harness (7 test questions with known-correct source documents) to measure retrieval quality rather than relying on manual spot-checks.

- Precision@1: 0.86 (6/7 questions retrieved the correct source as the top result)
- Precision@3: 1.00 (all questions found the correct source within the top 3 results)

The one Precision@1 miss involved a question with overlapping content across two related documents, correctly resolved within the top 3.

## Guardrail

The system checks retrieval confidence before generating an answer. Below a similarity threshold, it responds that it does not have enough information rather than generating an unsupported answer.
