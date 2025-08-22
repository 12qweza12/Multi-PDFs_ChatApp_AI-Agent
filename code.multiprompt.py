# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#     if not text.strip():
#         raise ValueError("No text extracted from the PDF. Please check the PDF file.")
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=800)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     if not text_chunks:
#         raise ValueError("No text chunks found. Please check the input text.")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_prompt_by_category(category):
#     if category == "สวัสดิการด้านสุขภาพ📁":
#         return """
#         คุณคือแชทบอทผู้เชี่ยวชาญด้านสวัสดิการสุขภาพพนักงาน ตอบคำถามโดยอ้างอิงจากเนื้อหาในเอกสารหมวด 'สวัสดิการด้านสุขภาพ' เท่านั้น
#         กรุณาตอบอย่างละเอียดและถูกต้อง โดยอ้างอิงหัวข้อย่อยที่เกี่ยวข้อง เช่น:
#         - สิทธิการรักษาพยาบาล
#         - เงื่อนไขการเบิกจ่าย
#         - ประเภทสวัสดิการที่ได้รับ
#         - ขั้นตอนการขอรับสวัสดิการ
#         - ข้อจำกัดและข้อยกเว้น
#         หากไม่มีข้อมูลในเอกสารนี้ ให้ตอบว่า "ไม่มีข้อมูลในเอกสารหมวดสวัสดิการด้านสุขภาพนี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" และห้ามเดาหรือให้ข้อมูลผิด

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer:
#         """
#     elif category == "วินัยลูกจ้าง📁":
#         return """
#         คุณคือแชทบอทผู้เชี่ยวชาญด้านวินัยลูกจ้าง ตอบคำถามโดยอ้างอิงจากเนื้อหาในเอกสารหมวด 'วินัยลูกจ้าง' เท่านั้น
#         กรุณาตอบอย่างละเอียดและถูกต้อง โดยอ้างอิงหัวข้อย่อยที่เกี่ยวข้อง เช่น:
#         - ข้อบังคับเกี่ยวกับวินัย
#         - ขั้นตอนการดำเนินการทางวินัย
#         - การลงโทษและการอุทธรณ์
#         - สิทธิของลูกจ้างในกระบวนการทางวินัย
#         หากไม่มีข้อมูลในเอกสารนี้ ให้ตอบว่า "ไม่มีข้อมูลในเอกสารหมวดวินัยลูกจ้างนี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" และห้ามเดาหรือให้ข้อมูลผิด

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer:
#         """
#     elif category == "รายงานสหกิจ":
#         return """
#         คุณคือแชทบอทผู้เชี่ยวชาญด้านรายงานสหกิจ ตอบคำถามโดยอ้างอิงจากเนื้อหาในเอกสารหมวด 'รายงานสหกิจ' เท่านั้น
#         กรุณาตอบอย่างละเอียดและถูกต้อง โดยอ้างอิงหัวข้อย่อยที่เกี่ยวข้อง เช่น:
#         - สรุปเนื้อหาการฝึกงาน
#         - ผลงานหรือโครงการที่ทำ
#         - ข้อเสนอแนะจากการฝึกงาน
#         หากไม่มีข้อมูลในเอกสารนี้ ให้ตอบว่า "ไม่มีข้อมูลในเอกสารหมวดรายงานสหกิจนี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" และห้ามเดาหรือให้ข้อมูลผิด

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer:
#         """
#     else:
#         # Default prompt
#         return """
#         You are an expert AI chatbot. Please answer the user's question based only on the provided context.
#         If the answer is not found in the context, reply: "ไม่มีข้อมูลในเอกสารนี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" and do not guess or provide incorrect information.

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer:
#         """

# def get_conversational_chain(category):
#     prompt_template = get_prompt_by_category(category)
#     model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question, category):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain(category)
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )
#     return response["output_text"]

# def main():
#     st.set_page_config("TUTHINK-PDF", page_icon=":computer:")
#     st.header("TUTHINK - Chatbot 📋🗂️🏥")

#     pdf_options = {
#         "สวัสดิการด้านสุขภาพ📁": "docs\\Rules.pdf",
#         "วินัยลูกจ้าง📁": "docs\\วินัยลูกจ้าง.pdf",
#         "รายงานสหกิจ": "docs\\65104788_เบญญาภา_ถนอมใจ.pdf"
#     }

#     with st.sidebar:
#         st.image("img/chatbot.jpg")
#         st.write("---")
#         st.title("About TUTHINK")
#         st.markdown("📖 TUTHINK เป็นแอปพลิเคชันที่ช่วยตอบคำถามเกี่ยวกับเอกสาร PDF")
#         st.write("---")
#         select_pdf = st.selectbox("เลือกหมวดหมู่เอกสารที่ต้องการถาม", list(pdf_options.keys()))
#         st.markdown(
#             """
#             <div style="position: fixed; bottom: 0; left: 0; padding: 15px; text-align: left;">
#                 © <a href="https://intranet.hospital.tu.ac.th/" target="_blank">Thammasat Hospital University</a>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

#     if "vector_store" not in st.session_state or st.session_state.select_pdf != select_pdf:
#         with st.spinner("กำลังเริ่มต้นและประมวลผลเอกสาร PDF ครับ..."):
#             with open(pdf_options[select_pdf], "rb") as pdf_file:
#                 raw_text = get_pdf_text([pdf_file])
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#             st.session_state.vector_store = True
#             st.session_state.select_pdf = select_pdf
#             st.toast("ประมวลผล PDF เสร็จเรียบร้อย!", icon="✅")
#     st.info(f"คุณกำลังถามคำถามจากเอกสาร: {select_pdf} 📁")

#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "สวัสดีครับ! ผมคือ TUTHINK Chatbot ยินดีให้บริการตอบคำถามเกี่ยวกับเอกสาร PDF ของคุณ ถามมาได้เลยครับ!"}
#         ]

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("Ask a Question from PDF ✍️📝"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(prompt)
#         with st.spinner("TUTHINK กำลังประมวลผลคำถามของคุณ..."):
#             response = user_input(prompt, select_pdf)
#             with st.chat_message("assistant"):
#                 st.markdown(response)
#                 st.session_state.messages.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()