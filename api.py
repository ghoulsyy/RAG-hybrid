#api服务

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

import config_data as config
from knolege_base import KnowledgeBaseService
from rag import RagService

#数据模型
class ChatRequest(BaseModel):
    question: str = Field(...,description="用户问题",min_length=1,max_length=500)
    session_id: str  =Field(
        default_factory=lambda:str(uuid.uuid4()),
        description = "会话id，不传入则自动生成"
    )

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class UploadResponse(BaseModel):
    success: bool
    filename: str
    message: str

#应用初始化
app = FastAPI(
    title="小灵通",
    description="基于混合检索的，小百科系统",
    version="1.0.0",
)

#懒加载
_rag: Optional[RagService] = None
_kb: Optional[KnowledgeBaseService] = None

def get_rag() -> RagService:
    global _rag
    if _rag is None:
        _rag = RagService()
    return _rag

def get_kb() -> KnowledgeBaseService:
    global _kb
    if _kb is None:
        _kb = KnowledgeBaseService()
    return _kb

#API端点
@app.get("/")
def root():
    return {"service":"小灵通","status":"ok"}

@app.post("/api/chat",response_model=ChatResponse)
def chat(req: ChatRequest):
    rag = get_rag()
    session_config = {"configurable":{"session_id":req.session_id}}

    try:
        result = rag.chain.invoke({"input":req.question},session_config)
        return ChatResponse(answer=result,session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"服务异常：{str(e)}")

@app.post("/api/upload",response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400,detail="仅支持 .txt 文件")

    try:
        content = await file.read()
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400,detail="文件编码需为 UTF-8")

    kb = get_kb()
    result = kb.upload_by_str(text,file.filename)

    return UploadResponse(
        success = "成功" in result,
        filename = file.filename,
        message=result,
    )

@app.get("/api/health")
def health():
    try:
        kb = get_kb()
        count = kb.chroma._collection.count()
        return {"status":"healthy","documents_count":count}
    except Exception as e:
        return {"status":"unhealthy","error":str(e)}
















