from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, claim, documents, top_n=5):
        pairs = [(claim, doc) for doc in documents]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_n]
