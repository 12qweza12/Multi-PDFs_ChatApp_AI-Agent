import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()

    if not text.strip():
        raise ValueError("No text extracted from the PDF. Please check the PDF file.")
    #ที่มัน Error เพราะ PDF บางไฟล์อ่านไม่ได้ มันอาจเป็นภาพสแกนหรือมีการเข้ารหัสที่ไม่สามารถอ่านได้
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=800)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks found. Please check the input text.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") #บันทึก vector store ใน folder ชื่อ faiss_index


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "ไม่มีข้อมูลในเอกสารครับ ลองถามคำถามอื่นเพิ่มเติมครับ", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) #เพิ่ม allow_dangerous_deserialization
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    return response["output_text"]  # Return the output text for further processing or display


def main():
    st.set_page_config("TUTHINK-PDF", page_icon=":computer:")
    st.header("TUTHINK - Chatbot 📋🗂️🏥")

    with st.sidebar:
        st.image("img/chatbot.jpg")
        st.write("---")
        
        st.title("About TUTHINK")
        st.markdown("📖 TUTHINK เป็นแอปพลิเคชันที่ช่วยตอบคำถามเกี่ยวกับเอกสาร PDF")

        st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; padding: 15px; text-align: left;">
            © <a href="https://intranet.hospital.tu.ac.th/" target="_blank">Thammasat Hospital University</a>
        </div>
        """,
        unsafe_allow_html=True
    )
        
    # ตรวจสอบว่า vector_store อยู่ใน session_state หรือไม่
    # state คือ ตัวแปรของ streamlit เก็บข้อมูลประมวลผลไว้ในหน่วยความจำ session และไม่ประมวลผลซ้ำเมื่อถามคำถามใหม่
    if "vector_store" not in st.session_state:
    # บังคับให้ user ถามคำถามจาก PDF ที่กำหนดไว้เท่า
        with st.spinner("กำลังเริ่มต้นและประมวลผลเอกสาร PDF ครับ..."):
            predefined_pdf_path = "docs\Rules.pdf"  # Path to the embedded PDF file
            with open(predefined_pdf_path, "rb") as pdf_file:  # rb คือ read binary อ่านข้อมูลจากไฟล์ PDFที่เป็น binary
                raw_text = get_pdf_text([pdf_file])  # Process the predefined PDF
                text_chunks = get_text_chunks(raw_text)  # Get text chunks
                get_vector_store(text_chunks)  # Create vector store
            st.session_state.vector_store = True # บันทึกสถานะเป็น True เมื่อประมวลผลเสร็จแล้ว
            st.toast("ประมวลผล PDF เสร็จเรียบร้อย!", icon="✅")  # ใช้ st.toast แสดงข้อความสำเร็จ
    # else:
    #     st.success("ประมวลผล PDF เสร็จเรียบร้อยแล้วถามคำถามได้เลยครับ!!")

    #1. เริ่มต้น chat_history ถ้าไม่มี
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role" : "assistant", "content" : "สวัสดีครับ! ผมคือ TUTHINK Chatbot ยินดีให้บริการตอบคำถามเกี่ยวกับเอกสาร PDF ของคุณ ถามมาได้เลยครับ!"}]
    
    #2. แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #3. รอรับคำถามจากผู้ใช้
    if prompt := st.chat_input("Ask a Question from PDF ✍️📝"):
        st.session_state.messages.append({"role":"user", "content": prompt})  # บันทึกคำถามของผู้ใช้ใน chat_history
        with st.chat_message("user" , avatar="👤"):
            st.markdown(prompt)

        #ประมวลผลคำถาม
        #with st.chat_message("assistant"):
        with st.spinner("TUTHINK กำลังประมวลผลคำถามของคุณ..."):
            response = user_input(prompt)  # ประมวลผลคำถาม
            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role" : "assistant", "content":response}) # บันทึกคำตอบของ AI ใน chat_history

if __name__ == "__main__":
    main()















    # #เก็บคำถามและคำตอบใน chat_history
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []
    # #st.write(st.session_state.chat_history)
    
    # #st.session_state.chat_history.append({"user": user_question, "tuthink": response["output_text"]})
    
    # # feature conversation history
    # for chat in st.session_state.chat_history:
    #     if isinstance (chat, user_input):
    #         with st.chat_message("user"):
    #             st.markdown(chat.content)      # แสดงคำถามของ user
    #     else:
    #         with st.chat_message("tuthink"):
    #             st.markdown(chat.content)     # แสดงคำตอบของ AI 

        # st.markdown(
        #     f"""
        #     <div style="text-align: right; background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        #         <b>User:</b> {chat["user"]}
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )    
        
        # แสดงคำตอบของ AI ด้านซ้าย    
        # st.markdown(
        #     f"""
        #     <div style="text-align: left; background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        #         <b>TUTHINK 🤖:</b> {chat["tuthink"]}
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
    # feature แสดงประวัติการสนทนา //stop

    # print(response)
    # st.write("Reply: ", response["output_text"])
    
    # st.write("Reply: ", response)
    # แสดงคำถามของ user ด้านขวา
    # st.markdown(
    #     f"""
    #     <div style="text-align: right; background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    #         <b>User:</b> {user_question}
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    # แสดงคำตอบของ AI ด้านซ้าย
    # st.markdown(
    #     f"""
    #     <div style="text-align: left; background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    #         <b>TUTHINK 🤖:</b> {response["output_text"]}
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )








    # ช่องถามคำถามของ user
    #user_question = st.chat_input("Ask a Question from PDF ✍️📝")

    # run function ประมวลผลคำถามของ user
    # if user_question:
    #     user_input(user_question)

    # with st.sidebar:

    #     st.image("img/chatbot.jpg")
    #     st.write("---")
        
    #     st.title("📁 PDF File's Section")
    #     pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
    #     if st.button("Submit & Process"):
    #         with st.spinner("Processing..."): # user friendly message.
    #             raw_text = get_pdf_text(pdf_docs) # get the pdf text
    #             text_chunks = get_text_chunks(raw_text) # get the text chunks
    #             get_vector_store(text_chunks) # create vector store
    #             st.success("Done")
        





