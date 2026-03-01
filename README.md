# EviRank

EviRank is a Python-based fact verification pipeline designed to retrieve and rank evidence for claims using a hybrid method combining traditional and deep learning models. It leverages BM25 for initial retrieval, bi-encoders for semantic filtering, and cross-encoders for final reranking.

## Features

- **BM25 Retrieval**: Efficient lexical search using `rank_bm25`.
- **Bi-Encoder Reranking**: Fast semantic filtering using `sentence-transformers`.
- **Cross-Encoder Reranking**: Accurate final evidence ranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Evaluation Tools**: Precision@K, Recall@K, and MAP metrics for performance evaluation.
- **Interactive CLI**: User-friendly terminal menu to run and test the pipeline.

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script via the terminal:
```bash
python main.py
```

You will be presented with an interactive command line interface:

```
Select an option:
1. Run full pipeline
2. BM25 test
3. Bi-Encoder test
4. Cross-Encoder test
5. Hybrid test
```

Enter the desired option and follow the prompts to input claim indices.

### Notes
- Claim data should be in a file called `data/climatefever.jsonl`.
- For most options, you can enter `-1` to evaluate all claims in the dataset.
- Results and evaluation metrics will be saved to the `results/` directory.

## Project Structure

```
├── data/
│   └── climatefever.jsonl
├── models/
│   ├── bi_encoder_reranker.py
│   └── cross_encoder_reranker.py
├── tests/
│   ├── bm25_test.py
│   ├── bi_encoder_test.py
│   ├── cross_encoder_test.py
│   └── hybrid_test.py
├── utils/
│   ├── data_loader.py
│   ├── bm25_utils.py
│   └── evaluation_metrics.py
├── main.py
├── main_pipeline.py
├── requirements.txt
└── README.md
```

## Requirements

See `requirements.txt`.

