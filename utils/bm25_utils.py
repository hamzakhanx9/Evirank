from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

def calculate_bm25_scores(query, documents):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    return list(zip(documents, scores))

def get_top_k_unique(bm25_results, k=200):
    seen = set()
    unique_results = []
    for doc, score in sorted(bm25_results, key=lambda x: x[1], reverse=True):
        if doc not in seen:
            seen.add(doc)
            unique_results.append((doc, score))
        if len(unique_results) == k:
            break
    return unique_results