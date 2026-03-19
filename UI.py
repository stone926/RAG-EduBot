# UI.py
# 只包含Streamlit界面相关的代码

import streamlit as st
from API import RAGChatAPI

# 设置页面标题和基本布局
st.set_page_config(page_title="Z.ai RAG 智能助教", layout="centered")

# 初始化 API 客户端（放在最前保证全局可用）
if "api" not in st.session_state:
    try:
        st.session_state.api = RAGChatAPI()
        st.session_state.api.check_initialized()
    except Exception as e:
        st.error(f"初始化失败: {e}")
        st.stop()

# 页面路由：判断是否已经选择了课程
if "course_selected" not in st.session_state:
    st.session_state.course_selected = False

# ================= 1. 课程选择界面 =================
if not st.session_state.course_selected:
    st.title("📚 Z.ai RAG 智能助教系统")
    st.write("请先选择您想要咨询的课程：")
    
    course_choice = st.radio(
        "选择课程科目:",
        ("操作系统 (OS)", "计算机组成原理 (CO)")
    )
    
    if st.button("进入助教系统", type="primary"):
        # 根据选择设置课程信息
        if "操作系统" in course_choice:
            st.session_state.course_type = "os"
            st.session_state.course_name = "操作系统"
        else:
            st.session_state.course_type = "co"
            st.session_state.course_name = "计算机组成原理"
            
        try:
            # 动态加载对应课程的知识库
            st.session_state.api.load_course_db(st.session_state.course_type)
            
            # 初始化与课程绑定的对话记录
            st.session_state.conversation = [
                {"role": "system", "content": f"你是一名AI{st.session_state.course_name}课程助教，负责解答学生关与{st.session_state.course_name}的相关问题，请确保回答的准确性"}
            ]
            st.session_state.display_history = [
                {"role": "system", "content": f"欢迎使用智能《{st.session_state.course_name}》课程助教！您可以开始提问了。"}
            ]
            
            # 标记为已选择，并刷新页面进入对话界面
            st.session_state.course_selected = True
            st.rerun()
            
        except Exception as e:
            st.error(f"加载课程知识库失败: {e}\n请确保 vector_db_{st.session_state.course_type}.json 文件存在。")

# ================= 2. 对话聊天界面 =================
else:
    # 支持返回重选课程
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"Z.ai 智能助教 - {st.session_state.course_name}")
    with col2:
        if st.button("切换课程"):
            st.session_state.course_selected = False
            # 清理之前的对话记录，防止串台
            if "conversation" in st.session_state:
                del st.session_state["conversation"]
            if "display_history" in st.session_state:
                del st.session_state["display_history"]
            st.rerun()

    # 渲染历史聊天记录
    for msg in st.session_state.display_history:
        if msg["role"] == "system":
            with st.chat_message("system", avatar="⚙️"):
                st.info(msg["content"])
        elif msg["role"] == "rag":
            with st.chat_message("system", avatar="🔍"):
                st.success(msg["content"])
        elif msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # 底部输入框
    user_input = st.chat_input("请输入您的问题...")

    # 处理用户输入逻辑
    if user_input and "api" in st.session_state:
        # 1. 在界面上显示并保存用户的原始问题
        st.session_state.display_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            # 2. RAG 检索（调用API层）
            similar_text = st.session_state.api.retrieve_similar_text(user_input)
            
            # 3. 准备RAG检索结果（调用API层）
            rag_result = st.session_state.api.prepare_rag_result(similar_text)
            rag_msg = rag_result["display_message"]
            combined_text = rag_result["combined_text"]

            # 4. 在界面显示并保存 RAG 的检索结果
            st.session_state.display_history.append({"role": "rag", "content": rag_msg})
            with st.chat_message("system", avatar="🔍"):
                st.success(rag_msg)

            # 5. 构建包含知识库内容的提示词（调用API层）
            enhanced_prompt = st.session_state.api.build_enhanced_prompt(user_input, combined_text)

            # 6. 添加到对话历史
            st.session_state.conversation.append({"role": "user", "content": enhanced_prompt})

            # 7. 调用API获取流式响应
            with st.chat_message("assistant"):
                response = st.session_state.api.stream_chat(
                    messages=st.session_state.conversation
                )

                # 定义流式生成器
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                # 流式打印并获取完整响应
                full_response = st.write_stream(stream_generator())

            # 8. 保存到历史记录
            st.session_state.conversation.append({"role": "assistant", "content": full_response})
            st.session_state.display_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            # 异常处理
            error_msg = f"发生错误: {e}"
            st.error(error_msg)
            st.session_state.display_history.append({"role": "system", "content": error_msg})