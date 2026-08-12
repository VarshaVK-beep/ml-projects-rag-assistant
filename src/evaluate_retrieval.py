import json
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='data/chroma_db')
collection = client.get_collection(name='ml_projects_docs')

eval_set = [
    {'question': 'What accuracy did the churn prediction model achieve?', 'expected_source': 'churn_readme.md'},
    {'question': 'What hyperparameter tuning method was used for XGBoost?', 'expected_source': 'churn_readme.md'},
    {'question': 'How was SHAP used in the churn project?', 'expected_source': 'churn_readme.md'},
    {'question': 'What transformer model was fine-tuned for ticket classification?', 'expected_source': 'ticket_classifier_readme.md'},
    {'question': 'How is the churn model deployed?', 'expected_source': 'churn_readme.md'},
    {'question': 'What tokenizer was used for the support ticket model?', 'expected_source': 'ticket_finetune.py'},
    {'question': 'How does the churn preprocessing handle missing TotalCharges values?', 'expected_source': 'churn_preprocessing.py'},
]

def retrieve(question, top_k=3):
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    sources = [m['source'] for m in results['metadatas'][0]]
    return sources

def precision_at_k(retrieved_sources, expected_source, k):
    top_k_sources = retrieved_sources[:k]
    return 1.0 if expected_source in top_k_sources else 0.0

results_log = []
for item in eval_set:
    q = item['question']
    expected = item['expected_source']
    retrieved = retrieve(q, top_k=3)
    p_at_1 = precision_at_k(retrieved, expected, k=1)
    p_at_3 = precision_at_k(retrieved, expected, k=3)
    results_log.append({'question': q, 'expected_source': expected, 'retrieved_sources': retrieved, 'precision_at_1': p_at_1, 'precision_at_3': p_at_3})
    if p_at_1 == 1.0:
        status = 'PASS'
    elif p_at_3 == 1.0:
        status = 'PARTIAL'
    else:
        status = 'FAIL'
    print('[' + status + '] ' + q)
    print('  Expected: ' + expected + ' | Retrieved top-3: ' + str(retrieved))

avg_p1 = sum(r['precision_at_1'] for r in results_log) / len(results_log)
avg_p3 = sum(r['precision_at_3'] for r in results_log) / len(results_log)

print()
print('Precision@1: ' + str(round(avg_p1, 2)))
print('Precision@3: ' + str(round(avg_p3, 2)))

with open('data/retrieval_eval_results.json', 'w') as f:
    json.dump({'results': results_log, 'precision_at_1': avg_p1, 'precision_at_3': avg_p3}, f, indent=2)

print('Saved: data/retrieval_eval_results.json')
