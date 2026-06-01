'''
知识库
'''
import datetime
import hashlib

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
import os


def check_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        with open(config.md5_path, 'w', encoding="utf-8") as f:
            pass
        return False
    else:
        with open(config.md5_path, 'r', encoding="utf-8") as f:
            for line in f:
                if line == md5_str:
                    return True
        return False


def save_md5(md5_str: str):
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding="utf-8"):
    str_bytes = input_str.encode(encoding)

    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()

    return md5_hex

#
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5"
# )
#

class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在就创建文件夹，如果文件夹存在就忽略这条语句
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=config.embeddings,
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self, data: str, filename):
        logger = config.get_logger(__name__)

        if not data or not data.strip():
            logger.warning(f"上传内容为空，文件名称：{filename}")
            return "[失败]上传内容不能为空"

        try:
            md5_hex = get_string_md5(data)
        except Exception as e:
            logger.error(f"计算 MD5 失败：{e}")
            return "[失败]内容处理异常"

        if check_md5(md5_hex):
            logger.info(f"内容已存在，跳过:{filename}")
            return "[跳过]内容已经存在知识库当中"
        try:
            if len(data) > config.max_split_char_number:
                knowledge_chunks: list[str] = self.spliter.split_text(data)
            else:
                knowledge_chunks = [data]

            metadata = {
                "source":filename,
                "create_time":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            self.chroma.add_texts(
                knowledge_chunks,
                metadatas = [metadata for _ in knowledge_chunks],
            )

            save_md5(md5_hex)
            logger.info(f"知识库更新成功：{filename},分块数：{len(knowledge_chunks)}")
            return "[成功]内容已经成功添加至向量库"
        except Exception as e:
            logger.error(f"写入向量库失败[{filename}]:{e}",exc_info=True)
            return f"[失败]写入向量库异常：{str(e)}"
if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("你好，今天天气很好，阳光明媚","testfile")
    print(r)