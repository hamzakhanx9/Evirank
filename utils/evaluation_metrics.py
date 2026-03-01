def calculate_precision_at_k(ranked_docs, gold_evidences, k):
    relevant_docs = set(ranked_docs[:k])
    relevant_gold = [doc for doc in gold_evidences if doc in relevant_docs]
    return len(relevant_gold) / k

def calculate_recall_at_k(ranked_docs, gold_evidences, k):
    relevant_docs = set(ranked_docs[:k])
    return len([doc for doc in gold_evidences if doc in relevant_docs]) / len(gold_evidences)

def calculate_ap(ranked_docs, gold_evidences):
    ap_sum = 0
    relevant_count = 0
    for i, doc in enumerate(ranked_docs):
        if doc in gold_evidences:
            relevant_count += 1
            ap_sum += relevant_count / (i + 1)
    
    num_relevant = len(gold_evidences)
    return ap_sum / num_relevant if num_relevant > 0 else 0


