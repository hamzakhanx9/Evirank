import json
import os
from models.bi_encoder_reranker import rerank
from models.cross_encoder_reranker import CrossEncoderReranker
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    return data

def calculate_bm25_scores(claim, evidence_list):
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in evidence_list]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_claim = word_tokenize(claim.lower())
    scores = bm25.get_scores(tokenized_claim)
    return list(zip(evidence_list, scores))

def run_full_pipeline(claim_index: int):
    dataset = load_dataset('data/climatefever.jsonl')
    claim_data = dataset[claim_index]
    claim = claim_data['claim']

    print(f"\nClaim: {claim}")

    evidences = claim_data.get('evidences', [])
    all_evidence = [e['evidence'] for c in dataset for e in c.get('evidences', [])]

    bm25_results = calculate_bm25_scores(claim, all_evidence)
    top_k_bm25 = 200
    seen = set()
    unique_top_bm25 = []
    for doc, score in sorted(bm25_results, key=lambda x: x[1], reverse=True):
        if doc not in seen:
            seen.add(doc)
            unique_top_bm25.append((doc, score))
        if len(unique_top_bm25) == top_k_bm25:
            break

    print(f"\nBM25 Results (Top {top_k_bm25} Unique):")
    for evidence, score in unique_top_bm25[:5]:
        print(f"  - {evidence} ({score:.2f})")

    top_bm25_docs = [doc for doc, _ in unique_top_bm25]
    bi_reranked_results = rerank(claim, top_bm25_docs)

    top_k_biencoder = 100
    top_biencoder_docs = bi_reranked_results[:top_k_biencoder]

    print(f"\nInitial Reranked Results (Bi-Encoder, Top 5 of {top_k_biencoder}):")
    for evidence, score in top_biencoder_docs[:5]:
        print(f"  - {evidence} ({score:.4f})")

    top_k_cross = 20
    cross_encoder = CrossEncoderReranker()
    top_biencoder_texts = [doc for doc, _ in top_biencoder_docs]
    cross_reranked_results = cross_encoder.rerank(claim, top_biencoder_texts, top_n=top_k_cross)

    print(f"\nFinal Reranked Results (Hybrid - Cross-Encoder, Top 5 of {top_k_cross}):")
    for evidence, score in cross_reranked_results[:5]:
        print(f"  - {evidence} ({score:.4f})")

    print("\nGold Evidence:")
    for evidence in evidences:
        print(f"  - {evidence['evidence']}")

