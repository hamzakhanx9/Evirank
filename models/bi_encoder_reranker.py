import torch
from sentence_transformers import SentenceTransformer

def rerank(claim, top_bm25_docs):
    model = SentenceTransformer("all-MiniLM-L6-v2") 
    claim_embedding = model.encode(claim, convert_to_tensor=True) 
    document_embeddings = model.encode(top_bm25_docs, convert_to_tensor=True)  

    similarities = torch.cosine_similarity(claim_embedding, document_embeddings)

    ranked_docs = sorted(zip(top_bm25_docs, similarities.tolist()), key=lambda x: x[1], reverse=True)

    return ranked_docs