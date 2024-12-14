from retriv import SparseRetriever, HybridRetriever, DenseRetriever

class Retriever:
    def __init__(self, retrievel_type: str, retriever_version: str):
        if retrievel_type == "sparse":
            self.retriever = SparseRetriever.load(retriever_version)
        elif retrievel_type == "dense":
            self.retriever = DenseRetriever.load(retriever_version)
        elif retrievel_type == "hybrid":
            self.retriever = HybridRetriever.load(retriever_version)
        else:
            raise ValueError("Invalid retriever type")

    def retrieve(self, query: str, top_k: int = 5):
        return self.retriever.search(query=query, return_docs=True, cutoff=top_k)