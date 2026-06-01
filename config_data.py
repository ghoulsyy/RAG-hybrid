from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

#加载环境变量  .env文件
load_dotenv()

#项目根目录（config_data.py 所在目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

#路径配置
md5_path = os.path.join(ROOT_DIR, "data", "md5.txt")
persist_directory = os.path.join(ROOT_DIR, "data", "chroma_db")

#向量库配置
collection_name = "rag"

#文本切割配置
chunk_size = 500    #太大会稀释语义，引入噪声，太小丢失上下文
chunk_overlap = 45  #保证关键信息不会恰好落在切割边界上
max_split_char_number = 200
separators = ["\n\n", "\n", "。", "？", "！", "，", ".", "?", "!", ","]

#检索配置
similarity_top_k = 4


#模型配置
embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL","BAAI/bge-small-zh-v1.5")
    )
chat_model = os.getenv("CHAT_MODEL","deepseek-v4-flash")    #第二个参数是默认值，是兜底机制
# session_config = {
#         "configurable":{
#             "session_id":"user_001",
#         }
#     }

#logging 配置
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, "app.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_logger(name: str):
    return logging.getLogger(name)