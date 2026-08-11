import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='data/chroma_db')
collection = client.get_collection(name='ml_projects_docs')

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

NO_ANSWER = 'I do not have enough information in my knowledge base to answer that.'

def ask(question, top_k=3):
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    docs = results['documents'][0]
    distances = results['distances'][0]
    sources = [m['source'] for m in results['metadatas'][0]]
    best_similarity = 1 - distances[0]
    if best_similarity < 0.35:
        return {'answer': NO_ANSWER, 'sources': [], 'confidence': best_similarity}
    context = chr(10).join(docs)
    prompt = 'Answer the question using only the context below.' + chr(10) + chr(10) + 'Context:' + chr(10) + context + chr(10) + chr(10) + 'Question: ' + question + chr(10) + 'Answer:'
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {'answer': answer, 'sources': list(set(sources)), 'confidence': best_similarity}

questions = ['What accuracy did the churn prediction model achieve?', 'What model was used for support ticket classification?', 'What is the capital of France?']

for q in questions:
    r = ask(q)
    print('Q: ' + q)
    print('A: ' + r['answer'])
    print('Sources: ' + str(r['sources']) + ', Confidence: ' + str(round(r['confidence'], 2)))
    print()
