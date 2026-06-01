#稀疏检索

import json
import os
import pickle
from typing import List,Tuple

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config_data as config

def tokenize(text):
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip())>1]

class BM25Retriever:
    def __init__(self, persist_path=None):
        if persist_path is None:
            persist_path = os.path.join(config.ROOT_DIR, "data", "bm25_index")
        self.persist_path = persist_path
        self.documents = []
        self.bm25 = None

    def build_index(self,documents):
        if not documents:
            return
        self.documents = documents
        tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save()

    def search(self,query,top_k = None):
        """
        检索并返回[(文档,BM25分数),...]
        只返回分数 >0 的结果
        """
        if self.bm25 is None:
            self._load()
        if self.bm25 is None:
            return []

        top_k = top_k or config.similarity_top_k
        tokenized_query = tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x:x[1],reverse=True)

        return [
            (self.documents[idx],float(score))
            for idx,score in indexed[:top_k]
            if score > 0
        ]

    def _save(self):
        #持久化索引
        os.makedirs(self.persist_path,exist_ok=True)
        with open(os.path.join(self.persist_path,"docs.json"),"w",encoding="utf-8") as f:
            json.dump([{
                "content":d.page_content,
                "metadata":d.metadata,
            } for d in self.documents],f,ensure_ascii=False)
        with open(os.path.join(self.persist_path,"bm25.pkl"),"wb") as f:
            pickle.dump(self.bm25,f)

    def _load(self):
        docs_path = os.path.join(self.persist_path,"docs.json")
        idx_path = os.path.join(self.persist_path,"bm25.pkl")
        if os.path.exists(docs_path) and os.path.exists(idx_path):
            with open(docs_path,"r",encoding="utf-8") as f:
                data = json.load(f)
                self.documents = [
                    Document(page_content=d["content"],metadata=d["metadata"])
                    for d in data
                ]
            with open(idx_path,"rb") as f:
                self.bm25 = pickle.load(f)