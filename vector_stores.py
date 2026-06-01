from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
            return self.vector_store.as_retriever(
                search_kwargs = {"k":config.similarity_top_k}
            )

if __name__ == '__main__':
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-small-zh-v1.5"
    # )
    retriever = VectorStoreService(config.embeddings).get_retriever()

    res = retriever.invoke("体重60kg推荐尺码多少？")
    print(res)


