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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from side_bar.sidebar import sidebar
# from side_bar.footer import footer

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks found. Please check the input text.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") #บันทึก vector store ใน folder ชื่อ faiss_index
    return vector_store


def get_conversational_chain():

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    prompt = PromptTemplate(
        template = """
            คุณคือผู้ช่วยในการตอบคำถามเกี่ยวกับโรงพยาบาลธรรมศาตร์เฉลิมพระเกียรติ มีข้อกำหนดดังนี้:
            - หากผู้ใช้ทักทาย ให้ทักทายผู้ใช้ด้วยความสุภาพ
            - กรุณาปรับคำตอบให้ดูเป็นธรรมชาติและเหมาะสมสำหรับการสนทนาลงท้ายด้วยคำว่าคะหรือค่ะ
            - ตอบคำถามจากข้อมูลที่มีอยู่คำตอบจากระบบเท่านั้น และไม่สร้างข้อมูลเพิ่มเติม ปรับคำตอบให้ดูเป็นธรรมชาติและเหมาะสมสำหรับการสนทนา แต่ไม่ต้องตัดคำหรือเพิ่มคำลงไป คงคำตอบเดิมไว้ให้ได้มากที่สุด
            - หากคำถามไม่เกี่ยวข้องกับข้อมูลในคำตอบ ให้ตอบกลับว่า "ขออภัยค่ะ ฉันไม่สามรถตอบคำถามนี้ได้ กรุณาติดต่อเจ้าหน้าที่"
            - หากในคำตอบมีหลายประเด็น ให้สรุปใจความสำคัญและตอบให้กระชับ
            - หากในคำตอบมีลิงก์ ให้ใส่ลิงก์นั้นในคำตอบส่งไปให้ผู้ใช้ด้วย

            ข้อมูล:
            {context}

            คำถาม:
            {question}

        """,
        input_variables = ["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) #เพิ่ม allow_dangerous_deserialization
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    # st.write(chain)
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    # st.write(response) #จะออกมาเป็น output_text จั๊ฟ

    #เก็บคำถามและคำตอบใน display_history
    if "display_history" not in st.session_state:
        st.session_state.display_history = []
        st.session_state.display_history.append(
            {"user": user_question, "tuthink": response["output_text"]}
        )
    else:
        st.session_state.display_history.append(
            {"user": user_question, "tuthink": response["output_text"]}
        )
    
    # st.write(st.session_state.display_history)  
    
    # feature แสดงประวัติการสนทนา จ้า //start
    for chat in st.session_state.display_history:
        st.chat_message("user").markdown(chat["user"])  # แสดงคำถามของ user
        st.chat_message("tuthink", avatar='assistant').markdown(chat["tuthink"])

        # # แสดงคำถามของ user ด้านขวา    
        # st.markdown(
        #     f"""
        #     <div style="text-align: right; background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        #         <b>User:</b> {chat["user"]}
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )    
        
        # # แสดงคำตอบของ AI ด้านซ้าย    
        # st.markdown(
        #     f"""
        #     <div style="text-align: left; background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        #         <b>TUTHINK 🤖:</b> {chat["tuthink"]}
        #     </div>
        #     """,
        #     unsafe_allow_html=True)




def main():
    st.set_page_config("TUTHINK-PDF", page_icon=":computer:")
    st.header("TUTHINK - Chatbot 📋🗂️🏥")

    # sidebar() #import sidebar มาจาก side_bar/sidebar.py
    # footer() #import footer มาจาก side_bar/footer.py

    pdf_options = {
        "pBoat": "docs\pBoat\BOT_ans.pdf",
        "pBoat_1": "docs\pBoat\BOT_ans_1.pdf",
        "ระเบียบการแต่งกาย": "docs\ระเบียบการแต่งกาย\เอกสารแนบท้าย2.pdf",
        "สวัสดิการยืดหยุ่น" : "docs\สวัสดิการยืดหยุ่น\ประกาศ มธ.สวัสดิการด้านสุขภาพ พ.ศ.2566.pdf",
        "ข้อบังคับว่าด้วยวินัย" : "docs\ข้อบังคับว่าด้วยวินัย\สาระสำคัญข้อบังคับวินัย 2566.pdf",
        "ทดสอบ" : "docs\ทดสอบ\สิรวิชญ์_จุทอง.pdf",
    }

    quick_questions = [
        "สรุปเอกสารนี้",
        "ตอบคำถามอะไรได้บ้าง",
        "What are the disciplinary rules?",
        "Can you summarize the key points of the uploaded document?"
    ]

    with st.sidebar:
        st.image("img/chatbot.jpg")
        st.write("---")
            
        st.title("About TUTHINK")
        st.markdown("📖 TUTHINK เป็นแอปพลิเคชันที่ช่วยตอบคำถามเกี่ยวกับเอกสาร PDF")

        selected_pdf = st.selectbox("เลือกหมวดหมู่ที่ต้องการถาม", list(pdf_options.keys()))

        st.write("---")
        st.title("Quick Questions")
        for question in quick_questions:
            if st.button(question, key=question, use_container_width=True):
                st.session_state.selected_question = question

    # ตรวจสอบว่า vector_store อยู่ใน session_state หรือไม่
    # state คือ ตัวแปรของ streamlit เก็บข้อมูลประมวลผลไว้ในหน่วยความจำ session และไม่ประมวลผลซ้ำเมื่อถามคำถามใหม่
    #ตอนแรกมีแค่ not in st.session_state ตอนหลังมาเพิ่ม st.session_state.selected_doc ด้วย
    if "vector_store" not in st.session_state or st.session_state.selected_pdf != selected_pdf:

    # บังคับให้ user ถามคำถามจาก PDF ที่กำหนดไว้เท่า
        with st.spinner("กำลังเริ่มต้นและประมวลผลเอกสาร PDF ครับ...", show_time=True):
            # predefined_pdf_path = "docs\ระเบียบการแต่งกาย\เอกสารแนบท้าย2.pdf"  # Path to the embedded PDF file
            # with open(predefined_pdf_path, "rb") as pdf_file:  # rb คือ read binary อ่านข้อมูลจากไฟล์ PDFที่เป็น binary
            with open(pdf_options[selected_pdf], "rb") as pdf_file:
                raw_text = get_pdf_text([pdf_file])  # Process the predefined PDF
                text_chunks = get_text_chunks(raw_text)  # Get text chunks
                get_vector_store(text_chunks)  # Create vector store
            st.session_state.vector_store = True # บันทึกสถานะเป็น True เมื่อประมวลผลเสร็จแล้วเพื่อไม่ให้ประมวลผลซ้ำในรอบถัดไปที่ถามคำถาม
            st.session_state.selected_pdf = selected_pdf # บันทึก PDF ที่เลือกไว้ใน session_state
            st.toast("ประมวลผล PDF เสร็จแล้ว!!", icon="✅")
            
    st.info(f"คุณกำลังถามคำถามจากหมวดหมู่ : {selected_pdf}")
    # else:
    #     st.success("ประมวลผล PDF เสร็จเรียบร้อยแล้วถามคำถามได้เลยครับ!!")
        
    # ช่องถามคำถามของ user
    user_question = st.chat_input(placeholder="Ask a Question from PDF ✍️📝")

    # run function ประมวลผลคำถามของ user
    if user_question or "selected_question" in st.session_state:
        with st.spinner("กำลังประมวลผลคำถามของคุณ...", show_time=True):
            user_input(user_question if user_question else st.session_state.pop("selected_question"))




    

if __name__ == "__main__":
    main()
