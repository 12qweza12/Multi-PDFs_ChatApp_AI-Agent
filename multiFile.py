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

# --- โค้ดส่วนบนของคุณ (ตั้งแต่ load_dotenv() จนถึง get_conversational_chain()) ---
# --- ให้คงไว้เหมือนเดิมทุกประการ ---
load_dotenv()
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
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer
    Answer in Thai\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) # แนะนำให้ใช้ Model ที่ใหม่กว่า
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- สิ้นสุดโค้ดส่วนบน ---


# [ใหม่] ฟังก์ชันสำหรับดึงคำตอบจาก AI โดยเฉพาะ
def get_ai_response(user_question):
    """
    ฟังก์ชันนี้ทำหน้าที่ค้นหาข้อมูลใน Vector Store และรับคำตอบจากโมเดลภาษา
    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # ตรวจสอบว่าไฟล์ index มีอยู่หรือไม่ ก่อนที่จะโหลด
    if not os.path.exists("faiss_index"):
        return "Vector store index ไม่พบ กรุณาประมวลผล PDF ก่อน"

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) #เพิ่ม allow_dangerous_deserialization
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}, 
        return_only_outputs=True
    )
    
    return response["output_text"]


def main():
    st.set_page_config("TUTHINK-PDF", page_icon=":computer:")
    
    # --- Sidebar ยังคงเดิม ---
    with st.sidebar:
        st.image("img/chatbot.jpg")
        st.write("---")


        st.title("About TUTHINK")
        st.markdown("📖 TUTHINK เป็นแอปพลิเคชันที่ช่วยตอบคำถามเกี่ยวกับเอกสาร PDF")
        st.write("---")
        st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; padding: 15px; text-align: left;">
            © <a href="https://intranet.hospital.tu.ac.th/" target="_blank">Thammasat Hospital University</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("TUTHINK - Chatbot 📋🗂️🏥")

    # --- ตรรกะการประมวลผล PDF ครั้งแรก (ยังคงเดิม) ---
    if "vector_store_created" not in st.session_state:
        with st.spinner("กำลังเริ่มต้นและประมวลผลเอกสาร PDF ครับ..."):
            try:
                predefined_pdf_path = "docs\สาระสำคัญข้อบังคับวินัย 2566.pdf"  # Path to the embedded PDF file
                if os.path.exists(predefined_pdf_path):
                    with open(predefined_pdf_path, "rb") as pdf_file:
                        raw_text = get_pdf_text([pdf_file])
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                    st.session_state.vector_store_created = True # บันทึกสถานะ
                    st.toast("ประมวลผล PDF เสร็จเรียบร้อย!", icon="✅")
                else:
                    st.error(f"ไม่พบไฟล์ PDF ที่: {predefined_pdf_path}")
                    st.stop()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผล PDF: {e}")
                st.stop()

    # --- ส่วน UI ของแชทที่ปรับปรุงใหม่ ---

    # 1. เริ่มต้น session state สำหรับเก็บข้อความถ้ายังไม่มี
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ! ผม TUTHINK-Bot ยินดีให้บริการเกี่ยวกับข้อมูลในเอกสารครับ"}]

    # 2. แสดงข้อความเก่าทั้งหมดในประวัติการแชท
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. ใช้ st.chat_input เพื่อรอรับคำถามจากผู้ใช้
    if prompt := st.chat_input("ถามคำถามเกี่ยวกับเอกสารของคุณ..."):
        # เพิ่มคำถามของผู้ใช้ vào session state และแสดงผล
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # แสดงสถานะ "กำลังคิด..." และรับคำตอบจาก AI
        with st.chat_message("assistant"):
            with st.spinner("TUTHINK กำลังคิด..."):
                response = get_ai_response(prompt)
                st.markdown(response)
        
        # เพิ่มคำตอบของ AI vào session state
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()