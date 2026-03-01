import os
import time
import numpy as np
from collections import Counter
from utils.data_loader import load_dataset
from utils.bm25_utils import calculate_bm25_scores, get_top_k_unique
from models.cross_encoder_reranker import CrossEncoderReranker
from utils.evaluation_metrics import calculate_precision_at_k, calculate_recall_at_k, calculate_ap

def run_cross_encoder_test(claim_index=-1, k=5, dataset=None, verbose=True):
    if dataset is None:
        dataset = load_dataset('data/climatefever.jsonl')

    if claim_index == -1:
        precision_list = []
        recall_list = []
        ap_list = []
        runtimes = []

        for idx, _ in enumerate(dataset[:250]):
            print(f"Processing Claim {idx + 1}/250")
            start_time = time.time()

            precision, recall, ap_score = run_cross_encoder_test(idx, k=k, dataset=dataset, verbose=False)

            elapsed = time.time() - start_time
            runtimes.append(elapsed)

            precision_list.append(precision)
            recall_list.append(recall)
            ap_list.append(ap_score)

        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_ap = np.mean(ap_list)

        highest_precision = np.max(precision_list)
        lowest_precision = np.min(precision_list)
        highest_recall = np.max(recall_list)
        lowest_recall = np.min(recall_list)
        highest_ap = np.max(ap_list)
        lowest_ap = np.min(ap_list)

        avg_runtime = np.mean(runtimes)
        min_runtime = np.min(runtimes)
        max_runtime = np.max(runtimes)
        std_runtime = np.std(runtimes)

        precision_counts = Counter([round(p, 2) for p in precision_list])
        recall_counts = Counter([round(r, 2) for r in recall_list])
        ap_counts = Counter([round(m, 2) for m in ap_list])

        os.makedirs("results", exist_ok=True)
        with open("results/cross_encoder_avg.txt", "w") as f:
            f.write(f"Number of Claims Processed: 250\n\n")
            f.write(f"Average Precision@{k}: {avg_precision:.4f}\n")
            f.write(f"Average Recall@{k}: {avg_recall:.4f}\n")
            f.write(f"Average AP: {avg_ap:.4f}\n\n")

            f.write(f"Highest Precision@{k}: {highest_precision:.4f}\n")
            f.write(f"Lowest Precision@{k}: {lowest_precision:.4f}\n\n")
            f.write(f"Highest Recall@{k}: {highest_recall:.4f}\n")
            f.write(f"Lowest Recall@{k}: {lowest_recall:.4f}\n\n")
            f.write(f"Highest AP: {highest_ap:.4f}\n")
            f.write(f"Lowest AP: {lowest_ap:.4f}\n\n")

            f.write("--- Runtime Statistics (in seconds) ---\n")
            f.write(f"Average Runtime: {avg_runtime:.4f}s\n")
            f.write(f"Minimum Runtime: {min_runtime:.4f}s\n")
            f.write(f"Maximum Runtime: {max_runtime:.4f}s\n")
            f.write(f"Standard Deviation: {std_runtime:.4f}s\n\n")

            f.write("--- Value Counts ---\n")
            f.write(f"Precision@{k} Distribution:\n")
            for value, count in sorted(precision_counts.items()):
                f.write(f"  {value:.2f}: {count} claims\n")

            f.write(f"\nRecall@{k} Distribution:\n")
            for value, count in sorted(recall_counts.items()):
                f.write(f"  {value:.2f}: {count} claims\n")

            f.write(f"\nAP Distribution:\n")
            for value, count in sorted(ap_counts.items()):
                f.write(f"  {value:.2f}: {count} claims\n")

        print("\nResults saved to results/cross_encoder_avg.txt")
        return

    claim_data = dataset[claim_index]
    claim = claim_data['claim']

    if verbose:
        print(f"\nClaim: {claim}")

    evidences = [e['evidence'] for d in dataset for e in d.get('evidences', [])]
    gold_evidences = [e['evidence'] for e in claim_data.get('evidences', [])]

    bm25_results = calculate_bm25_scores(claim, evidences)
    top_docs = get_top_k_unique(bm25_results)

    cross_encoder = CrossEncoderReranker()
    reranked = cross_encoder.rerank(claim, [doc for doc, _ in top_docs], top_n=k)

    reranked_docs = [doc for doc, _ in reranked]

    if verbose:
        print(f"\nCross-Encoder Reranked Results (Top {k}):")
        for doc, score in reranked:
            print(f"  - {doc} ({score:.4f})")

        print("\nGold Evidence:")
        for gold in gold_evidences:
            print(f"  - {gold}")

    precision_at_k = calculate_precision_at_k(reranked_docs, gold_evidences, k)
    recall_at_k = calculate_recall_at_k(reranked_docs, gold_evidences, k)
    ap_score = calculate_ap(reranked_docs, gold_evidences)

    if verbose:
        print(f"\nEvaluation Metrics (Top {k}):")
        print(f"  - Precision@{k}: {precision_at_k:.4f}")
        print(f"  - Recall@{k}: {recall_at_k:.4f}")
        print(f"  - AP: {ap_score:.4f}")

    return precision_at_k, recall_at_k, ap_score
