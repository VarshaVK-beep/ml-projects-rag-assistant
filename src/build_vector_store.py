import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

# Load all source documents
doc_paths = glob.glob('data/source_docs/*')
print(f'Found {len(doc_paths)} source documents')

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Build chunks with metadata (which file each chunk came from)
all_chunks = []
all_metadata = []
for path in doc_paths:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    filename = os.path.basename(path)
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({'source': filename, 'chunk_index': i})

print(f'Created {len(all_chunks)} chunks from {len(doc_paths)} documents')

# Load embedding model (small, fast, local, free)
print('Loading embedding model...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all chunks
print('Generating embeddings...')
embeddings = embedder.encode(all_chunks, show_progress_bar=True)

# Set up ChromaDB (persistent local vector store)
client = chromadb.PersistentClient(path='data/chroma_db')
collection = client.get_or_create_collection(name='ml_projects_docs')

# Add chunks to the vector store
ids = [f'chunk_{i}' for i in range(len(all_chunks))]
collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=all_chunks,
    metadatas=all_metadata
)

print(f'Vector store built: {collection.count()} chunks indexed')
print('Saved to data/chroma_db/')
