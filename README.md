\# ML Projects RAG Assistant



A retrieval-augmented generation system that answers questions over my own ML project documentation. Uses local sentence-transformer embeddings, a ChromaDB vector store, and paragraph-based chunking, with a confidence-threshold guardrail to avoid answering out-of-scope questions.



\## Quickstart



```bash

git clone https://github.com/VarshaVK-beep/ml-projects-rag-assistant.git

cd ml-projects-rag-assistant

pip install -r requirements.txt



python src/build\_vector\_store.py

python src/query\_rag.py "your question here"

```



\## Example



Run `python src/query\_rag.py "What tuning method was used for the churn prediction model?"` and paste the actual output here once you've run it.

