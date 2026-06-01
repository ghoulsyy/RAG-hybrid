import streamlit as st
import config_data as config
from rag import RagService

st.title("小灵通")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"您好，请问需要什么帮助？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list = []
    with st.spinner("思考中......"):
        session_config = {
            "configurable": {
                "session_id": "user_001"  # Streamlit 单用户场景下固定 session 是可以的
            }
        }
        res_stream = st.session_state["rag"].chain.stream({"input":prompt},
                                                          session_config)


        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role":"assistant","content":"".join(ai_res_list)})