import streamlit as st

def display_chat(user_question, response):
#เก็บคำถามและคำตอบใน display_history
    if "display_history" not in st.session_state:
        st.session_state.display_history = []
        st.session_state.display_history.append({"user": user_question, "tuthink": response["output_text"]})
    else:
        st.session_state.display_history.append({"user": user_question, "tuthink": response["output_text"]})
    
    st.write(st.session_state.display_history)
    
    # feature แสดงประวัติการสนทนา จ้า //start
    for chat in st.session_state.display_history:
        # แสดงคำถามของ user ด้านขวา    
        st.markdown(
            f"""
            <div style="text-align: right; background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <b>User:</b> {chat["user"]}
            </div>
            """,
            unsafe_allow_html=True
        )    
        
        # แสดงคำตอบของ AI ด้านซ้าย    
        st.markdown(
            f"""
            <div style="text-align: left; background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <b>TUTHINK 🤖:</b> {chat["tuthink"]}
            </div>
            """,
            unsafe_allow_html=True)