import os
import glob
import re
import chromadb
from sentence_transformers import SentenceTransformer

doc_paths = glob.glob('data/source_docs/*')
print(f'Found {len(doc_paths)} source documents')

def clean_text(text):
    lines = text.split(chr(10))
    lines = [l for l in lines if not l.strip().startswith('![')]
    return chr(10).join(lines)

def chunk_by_paragraph(text, min_length=50):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    return chunks

all_chunks = []
all_metadata = []
for path in doc_paths:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = clean_text(text)
    filename = os.path.basename(path)
    for i, chunk in enumerate(chunk_by_paragraph(text)):
        all_chunks.append(chunk)
        all_metadata.append({'source': filename, 'chunk_index': i})

print(f'Created {len(all_chunks)} paragraph-based chunks')
for c in all_chunks:
    print('---')
    print(c[:150])

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(all_chunks, show_progress_bar=True)

client = chromadb.PersistentClient(path='data/chroma_db')
try:
    client.delete_collection('ml_projects_docs')
except Exception:
    pass
collection = client.get_or_create_collection(name='ml_projects_docs', metadata={'hnsw:space': 'cosine'})

ids = [f'chunk_{i}' for i in range(len(all_chunks))]
collection.add(ids=ids, embeddings=embeddings.tolist(), documents=all_chunks, metadatas=all_metadata)

print(f'Vector store rebuilt: {collection.count()} chunks indexed')
