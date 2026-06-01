#混合检索服务

from typing import List

from langchain_core.documents import Document

from sparse_retriever import BM25Retriever
from reranker import RerankerService
from vector_stores import VectorStoreService
import config_data as config


class HybridSearchService:
    def __init__(self):
        self.dense_store = VectorStoreService(config.embeddings)
        self.bm25 = BM25Retriever()
        self.reranker = RerankerService()

        #同步ChromaDB中已有文档
        self._sync_bm25_from_chroma()

    def _sync_bm25_from_chroma(self):
        """从 ChromaDB 同步文档到 BM25 索引"""
        try:
            data = self.dense_store.vector_store.get()
            if data and data.get("documents"):
                docs = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(data["documents"], data["metadatas"])
                ]
                self.bm25.build_index(docs)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"BM25 同步失败（可能是知识库为空）: {e}")

    def refresh_bm25_from_chroma(self):
        self._sync_bm25_from_chroma()

    def search(self,query,top_k = None):
        top_k = top_k or config.similarity_top_k
        recall_k = top_k *5

        #稠密检索
        dense_results = self.dense_store.vector_store.similarity_search(
            query,
            k = recall_k,
        )

        #稀疏检索
        sparse_results = self.bm25.search(query,top_k=recall_k)
        sparse_docs = [doc for doc,_ in sparse_results]

        #合并去重
        seen = set()
        merged = []
        for doc in dense_results + sparse_docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        #Rerank精排（如版本不兼容则跳过）
        if len(merged) <= top_k:
            return merged

        try:
            return self.reranker.rerank(query, merged, top_k)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Reranker 调用失败，降级为无需重排序的混合检索结果"
            )
            return merged[:top_k]