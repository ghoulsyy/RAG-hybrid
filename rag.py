from operator import itemgetter
import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_deepseek import ChatDeepSeek
import vector_stores
import config_data as config
from file_history_store import get_history

from hybrid_search import HybridSearchService

logger = logging.getLogger(__name__)


def format_document(docs: list[Document]):
    if not docs:
        return "无相关参考资料"

    formatted_str = ""
    for doc in docs:
        formatted_str += f"文档片段：{doc.page_content}\n文档元数据{doc.metadata}\n\n"
    return formatted_str

class RagService(object):

    import logging
    logger = logging.getLogger(__name__)

    def ask(self,question,session_id = "default"):
        #安全调用RAG链，带异常处理和奖及策略
        session_config = {
            "configurable":{"session_id":session_id},
        }
        try:
            result = self.chain.invoke({"input":question},session_config)
            return result
        except Exception as e:
            logger.error(f"RAG 链调用失败[session = {session_id}]：{e}",exc_info=True)
            return "抱歉，系统暂时无法处理您的请求，请稍后重试。"

    def __init__(self):
        self.search_service = HybridSearchService()
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个万能助手，以我提供的资料为主，简洁和专业的回答用户的问题，参考资料{context}"
             "如果参考资料没有相关信息，请如实告知用户，不要编造。"),
            ("system","并且我提供用户的对话历史记录如下："),
            MessagesPlaceholder("history"),
            ("user", "请回答用户提问:{input}"), ]
        )
        self.chat_model = ChatDeepSeek(model=config.chat_model)
        self.chain = self.__get_chain()

    def __get_chain(self):
        #retriever = self.vector_service.get_retriever()

        # itemgetter("input") 从 {"input": "..."} 中提取用户问题字符串
        # 两边各自独立提取，互不嵌套，不再需要 format_for_prompt_template
        chain = (
            {
                "input": itemgetter("input"),
                "context": itemgetter("input") | RunnableLambda(self.search_service.search) | format_document,
                "history": itemgetter("history"),
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        conversasion_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversasion_chain


if __name__ == '__main__':
    session_config = {
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"我的身高180，给我推荐尺码"},session_config)
    print(res)
