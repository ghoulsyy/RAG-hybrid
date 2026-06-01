#对初检结果进行精确重排

from typing import List
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker

import config_data as config

class RerankerService:
    def __init__(self,model_name = "BAAI/bge-reranker-v2-m3"):
        self.model = FlagReranker(model_name,use_fp16=True)

    def rerank(self,query,documents,top_k = None):
        if not documents:
            return []

        top_k = top_k or config.similarity_top_k

        pairs = [[query,doc.page_content] for doc in documents]

        scores = self.model.compute_score(pairs,normalize = True)

        ranked = sorted(zip(documents,scores),key  = lambda x: x[1],reverse = True)

        return [doc for doc,_ in ranked[:top_k]]