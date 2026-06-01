import streamlit as st
from knolege_base import KnowledgeBaseService
st.title("知识库更新服务")

uploader_file = st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # kb
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")
    try:
        text = uploader_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error("文件编码不支持，请上传 UTF-8 编码的 .txt 文件")
        text = None
    if text:
        with st.spinner("正在载入中..."):
            result = st.session_state["service"].upload_by_str(text,file_name)
            st.write(result)