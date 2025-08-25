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
#from langchain.chains import ConversationalRetrievalChain
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


# def get_conversational_chain():

#     prompt_template = """
#     You are an expert AI chatbot designed to answer user questions based on the selected document category. Always use only the information from the provided context below, which comes from the document category: "{category}".
#     Answer the user's question as accurately and completely as possible. If the answer is not found in the context, reply:
#     "ไม่มีข้อมูลในเอกสารหมวดหมู่นี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" and do not guess or provide incorrect information.
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

def get_prompt_by_category(category):
    if category == "สวัสดิการด้านสุขภาพ📁":
        return """
        คุณคือแชทบอทผู้เชี่ยวชาญด้านสวัสดิการสุขภาพพนักงาน จากเนื้อหาในเอกสาร 'หลักเกณฑ์และวิธีการจัดสวัสดิการด้านสุขภาพสาหรับพนักงานมหาวิทยาลัย พ.ศ. ๒๕๖๖' 
        คุณตอบคำถามโดยอ้างอิงจากเนื้อหาใน category 'สวัสดิการด้านสุขภาพ' เท่านั้น โดยตอบอย่างละเอียดและถูกต้อง โดยอ้างอิงหัวข้อย่อยที่เกี่ยวข้อง เช่น : 
        - เอกสาร ประกาศมหาวิทยาลัยธรรมศาสตร์ เรื่อง หลักเกณฑ์และวิธีการจัดสวัสดิการด้านสุขภาพสาหรับพนักงานมหาวิทยาลัยธรรมศาสตร์ พ.ศ.2566
        แบ่งออกเป็น 6 หมวดใหญ่ และบทเฉพาะกาล
            - หมวกที่ 1 คณะอนุกรรมการสวัสดิการด้านสุขภาพสาหรับพนักงานมหาวิทยาลัย
            - หมวดที่ 2 สวัสดิการด้านสุขภาพ
            - หมวดที่ 3 สวัสดิการด้านสุขภาพสาหรับพนักงานมหาวิทยาลัยซึ่งจ้างด้วยเงินงบประมาณแผ่นดิน
            - หมวดที่ 4 สวัสดิการด้านสุขภาพสาหรับพนักงานมหาวิทยาลัย (ส่วนงาน) พนักงานเงินรายได้ และพนักงานตามภารกิจ ที่สังกัดสานักงาน
            - หมวดที่ 5 สวัสดิการด้านสุขภาพสาหรับพนักงานมหาวิทยาลัย (ส่วนงาน) พนักงานเงินรายได้ และพนักงานตามภารกิจ ที่สังกัดส่วนงาน
            - หมวดที่ 6 หลักเกณฑ์และวิธีการขอรับสวัสดิการด้านสุขภาพแบบยืดหยุ่น
            - บทเฉพาะกาล
        
        - บัญชีแนบท้ายประกาศมหาวิทยาลัยธรรมศาสตร์
            - รายการสวัสดิการด้านสุขภาพแบบยืดหยุ่นที่สามารถเบิกได้ แบ่งเป็น รายการที่สามารถเบิกได้ หลักฐานประกอบการเบิกจ่าย
            - หมายเหตุ หลักฐานประกอบการเบิกจ่าย

            [ข้อมูลเพิ่มเติม รายการสวัสดิการด้านสุขภาพแบบยืดหยุ่นที่สามารถเบิกได้]
            คำอธิบาย : ตารางนี้อยู่ในหัวข้อ บัญชีแนบท้ายประกาศมหาวิทยาลัยธรรมศาสตร์ รายการสวัสดิการด้านสุขภาพแบบยืดหยุ่นที่สามารถเบิกได้
            ข้อมูลตาราง : 
            | รายการที่สามารถเบิกได้ | หลักฐานประกอบการเบิกจ่าย* |
            | :--- | :--- |
            | หมวด ๑ การป้องกันโรค | |
            | ๑.๑ ค่าใช้จ่ายในการตรวจสุขภาพ หรือค่าฉีดวัคซีน ทุกประเภท โดยต้องเป็นการใช้บริการจากสถานพยาบาลของรัฐหรือเอกชน หรือส่วนงานของมหาวิทยาลัย | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- ใบรับรองแพทย์ กรณีเข้ารับการรักษาพยาบาลในคลินิก/โพลีคลินิก |
            | ๑.๒ การประกันสุขภาพ <br>๑.๒.๑ ค่าเบี้ยประกันสุขภาพส่วนที่พนักงานมหาวิทยาลัยจ่ายเพิ่มเติม เพื่อทำประกันสุขภาพกลุ่มกับบริษัทประกันซึ่งมหาวิทยาลัยจัดหา ไม่ว่าเพื่อตนเองหรือบุคคลในครอบครัว<br>๑.๒.๒ ค่าเบี้ยประกันสุขภาพที่พนักงานมหาวิทยาลัยจ่ายเพื่อทำประกันสุขภาพให้แก่ตนเองหรือบุคคลในครอบครัวเพิ่มเติมจากข้อ ๑.๒.๑ | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- สำเนากรมธรรม์ประกันสุขภาพ (เฉพาะกรณี ๑.๒.๒) |
            | หมวด ๒ การรักษาพยาบาล | |
            | ๒.๑ ค่ารักษาพยาบาลหรือค่าบริการทางการแพทย์เฉพาะเพื่อการตรวจและการรักษาโรค | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- ใบรับรองแพทย์ กรณีเข้ารับการรักษาพยาบาลในคลินิก/โพลีคลินิก |
            | ๒.๒ ค่าบริการและค่าใช้จ่ายทางทันตกรรมเฉพาะเพื่อการรักษาพยาบาล | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- ใบรับรองแพทย์ กรณีเข้ารับการรักษาพยาบาลในคลินิก/โพลีคลินิก |
            | ๒.๓ ค่ารักษาพยาบาลหรือค่าบริการแพทย์แผนไทย แพทย์แผนจีน เฉพาะเพื่อการตรวจและการรักษาโรค | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- ใบรับรองแพทย์ กรณีเข้ารับการรักษาพยาบาลในคลินิก/โพลีคลินิก |
            | ๒.๔ ค่าใช้จ่ายเพื่อแก้ไขปัญหาความผิดปกติทางสายตา ดังนี้ ค่าแว่นและเลนส์สายตา ค่าคอนแทคเลนส์สายตาพร้อมอุปกรณ์ที่ต้องใช้ประกอบการใส่คอนแทคเลนส์ การทำเลสิก (LASIK) | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- กรณีค่าแว่นและเลนส์สายตา หรือค่าคอนแทคเลนส์พร้อมอุปกรณ์ ให้ระบุมูลค่าสายตาหรือความผิดปกติทางสายตาลงในใบเสร็จรับเงินด้วย |
            | ๒.๕ ค่าใช้จ่ายเพื่อแก้ไขปัญหาความผิดปกติทางการได้ยินและเครื่องช่วยฟัง | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- ใบสั่งแพทย์ |
            | ๒.๖ ค่าเวชภัณฑ์ทางการแพทย์ตามใบสั่งแพทย์ | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ และเอกสารเพิ่มเติม ดังนี้<br>- ใบสั่งแพทย์ |
            | หมวด ๓ การสร้างเสริมและฟื้นฟูสุขภาพ | |
            | ๓.๑ ค่าใช้จ่ายในการฟื้นฟูสมรรถภาพทางการแพทย์ ด้านกายภาพบำบัด เวชกรรมฟื้นฟู จิตบำบัด นวดรักษา อบหรือประคบเพื่อการรักษา ในสถานพยาบาลของภาครัฐ ภาคเอกชน หรือส่วนงานของมหาวิทยาลัย | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ - ๔ และเอกสารเพิ่มเติม ดังนี้<br>- ใบรับรองแพทย์ หรือใบประกอบวิชาชีพเวชกรรม กรณีเข้ารับการรักษาพยาบาลใน คลินิก/โพลีคลินิก |
            | ๓.๒ ค่ายา ดังนี้<br>๓.๒.๑ ยาสามัญประจำบ้าน<br>๓.๒.๒ ยาบำรุงร่างกาย (ตามรายการยาสามัญประจำบ้านแผนปัจจุบัน) ได้แก่<br>  - ยาวิตามินรวม<br>  - ยาวิตามินบีรวม<br>  - ยาวิตามินซี<br>  - ยาเม็ดบำรุงโลหิต เฟอร์รัส ซัลเฟต<br>  - น้ำมันตับปลาชนิดแคปซูล<br>  - น้ำมันตับปลาชนิดน้ำ<br>๓.๒.๓ ยาอื่นนอกจากยาสามัญประจำบ้าน เพื่อการรักษาโรค ไม่ใช่เพื่อการเสริมความงาม | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ - ๔ |
            | ๓.๓ ค่าเวชภัณฑ์ทางการแพทย์ ดังนี้<br>- เครื่องวัดความดัน<br>- อุปกรณ์วัดไข้<br>- เครื่องตรวจน้ำตาลในเลือดและแผ่นตรวจน้ำตาล<br>- เครื่องตรวจวัดออกซิเจน<br>- เครื่องผลิตออกซิเจน<br>- ชุดตรวจหาเชื้อไวรัสโคโรนา ๒๐๑๙ (โควิด ๑๙)<br>- เครื่องฟอกอากาศ อุปกรณ์บำรุงรักษาเครื่องฟอกอากาศ<br>- หน้ากากอนามัย หน้ากากกันฝุ่นละออง | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ - ๔ |
            | ๓.๔ ค่าสมาชิก ค่าใช้บริการ หรือค่าสมัครเข้าร่วมกิจกรรมที่เกี่ยวกับการออกกำลังกายหรือสร้างเสริมสุขภาพร่างกาย ทั้งภาครัฐ ภาคเอกชน หรือส่วนงานของมหาวิทยาลัย ดังนี้<br>๓.๔.๑ ค่าสมาชิกหรือค่าบริการสนามกีฬาหรือสถานที่ออกกำลังกายทุกประเภท<br>๓.๔.๒ ค่าสมัครเรียนกีฬาหรือฝึกฝนการออกกำลังกายทุกประเภท<br>๓.๔.๓ ค่าสมัครเข้าร่วมกิจกรรมการออกกำลังกายหรือสร้างเสริมสุขภาพร่างกายประเภทเดิน วิ่ง ปั่นจักรยาน ว่ายน้ำและไตรกีฬา ที่มีการจัดขึ้นภายในประเทศ | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ |
            | ๓.๕ อุปกรณ์การกีฬาหรือการออกกำลังกาย ไม่รวมอุปกรณ์เสริมและอุปกรณ์ตกแต่งเพื่อความสวยงาม ดังนี้<br>๓.๕.๑ อุปกรณ์การกีฬาทุกประเภทกีฬา หรืออุปกรณ์เพื่อการออกกำลังกาย โดยต้องเป็นอุปกรณ์ที่ใช้ในการเล่นกีฬาหรือในการออกกำลังกายโดยตรง<br>   ๓.๕.๒ รองเท้ากีฬาหรือรองเท้าที่ต้องใช้เพื่อการออกกำลังกายโดยตรง และรองเท้าเพื่อสุขภาพแบบหุ้มส้นหรือรัดส้น<br>๓.๕.๓ นาฬิกาสำหรับการออกกำลังกาย | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ |
            | ๓.๖ อุปกรณ์ดูแลสุขภาพ ดังนี้<br>๓.๖.๑ เข็มขัด เสื้อพยุงหลังเพื่อสุขภาพ<br>๓.๖.๒ เบาะรองนั่ง เบาะหนุนหลัง คอหรือศีรษะ เพื่อสุขภาพ<br>๓.๖.๓ อุปกรณ์หรือเครื่องนวดร่างกายเพื่อสุขภาพ เช่น คอ บ่า ไหล่ หลัง และขา<br>๓.๖.๔ หมวกนิรภัยหรือหมวกกันน็อก สำหรับผู้ขับขี่หรือคนโดยสารรถจักรยานยนต์ | - หลักฐานประกอบการเบิกจ่ายตามหมายเหตุ ข้อ ๑ – ๔ |

        - แนวทางการปฏิบัติสวัสดิการด้านสุขภาพสำหรับพนักงานมหาวิทยาลัย
        - คุณสมบัติของผู้มีสิทธิได้รับสวัสดิการด้านสุขภาพแบบยืดหยุ่น
        - การขอรับสวัสดิการ
        - รายงานการเบิกจ่ายเงินสวัสดิการด้านสุขภาพแบบยืดหยุ่น

        1. หากไม่มีข้อมูลในเอกสารนี้ ให้ตอบว่า "ไม่พบอมูลในเอกสารหมวดสวัสดิการด้านสุขภาพนี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" และห้ามเดาหรือให้ข้อมูลผิด
        2. หากข้อมูลที่ใช้ตอบมาจากหลายส่วน ให้ระบุทุกแหล่งอ้างอิง
        3. ห้ามให้ความเห็นส่วนตัวหรือตีความนอกเหนือจากที่ระบุในเอกสาร
        4. หากผู้ใช้ถามถึงเรื่องที่ไม่เกี่ยวกับเอกสาร หลักเกณฑ์และวิธีการจัดสวัสดิการด้านสุขภาพสาหรับพนักงานมหาวิทยาลัย พ.ศ. ๒๕๖๖ ฉบับนี้ ให้ตอบว่า "คำถามของคุณไม่อยู่ในขอบเขตของเอกสารนี้"
        5. ตอบเป็นภาษาไทยเสมอ ตอบอย่างเป็นมิตรและสุภาพ



        Context:\n {context}?\n

        Question: \n{question}\n

        Answer:



        """
    
    elif category == "วินัยลูกจ้าง📁":
        return """ 
        คุณคือแชทบอทผู้เชี่ยวชาญด้านวินัยลูกจ้าง จากเนื้อหาในเอกสาร 'ระเบียบกระทรวงการคลังว่าด้วยลูกจ้างประจำ ของส่วนราชการ พ.ศ. ๒๕๓๗'
        ตอบคำถามโดยอ้างอิงจากเนื้อหาใน category 'วินัยลูกจ้าง' เท่านั้น โดยตอบอย่างละเอียดและถูกต้อง โดยอ้างอิงหัวข้อย่อยที่เกี่ยวข้อง เช่น :
        - ระเบียบแบ่งออกเป็น 5 หมวดใหญ่ด้วยกัน โดยเริ่มที่
            หมวดที่ 4 วินัยและการรักษาวินัย
            หมวดที่ 5 การดำเนินการทางวินัย
            หมวดที่ 6 การออกจากราชการ
            หมวดที่ 7 การอุทธรณ์
            หมวดที่ 8 การร้องทุกข์
        
        1. หากไม่มีข้อมูลในเอกสารนี้ ให้ตอบว่า "ไม่พบอมูลในเอกสารหมวด 'วินัยลูกจ้าง' นี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" และห้ามเดาหรือให้ข้อมูลผิด
        2. หากข้อมูลที่ใช้ตอบมาจากหลายส่วน ให้ระบุทุกแหล่งอ้างอิง
        3. ห้ามให้ความเห็นส่วนตัวหรือตีความนอกเหนือจากที่ระบุในเอกสาร
        4. หากผู้ใช้ถามถึงเรื่องที่ไม่เกี่ยวกับเอกสาร ระเบียบกระทรวงการคลังว่าด้วยลูกจ้างประจำ ของส่วนราชการ พ.ศ. ๒๕๓๗ ฉบับนี้ ให้ตอบว่า "คำถามของคุณไม่อยู่ในขอบเขตของเอกสารนี้"
        5. ตอบเป็นภาษาไทยเสมอ ตอบอย่างเป็นมิตรและสุภาพ



        Context:\n {context}?\n

        Question: \n{question}\n

        Answer:
        
        """
    elif category == "รายงานสหกิจ":
        return """ 
        คุณคือแชทบอทผู้เชี่ยวชาญด้านรายงานสหกิจ จากเนื้อหาในเอกสาร 'รายงานการปฏิบัติสหกิจศึกษา ณ โรงพยาบาลธรรมศาสตร์เฉลิมพระเกียรติ'
        ตอบคำถามโดยอ้างอิงจากเนื้อหาใน category 'รายงานสหกิจ' เท่านั้น โดยตอบอย่างละเอียดและถูกต้อง โดยอ้างอิงหัวข้อย่อยที่เกี่ยวข้อง เช่น :
        รายงานฉบับนี้สามารถแบ่งตามหัวข้อต่างๆ จากสารบัญ
        - คำนำ
        - บทคัดย่อ
        - Abstract
        - กิตติกรรมประกาศ
        - สารบัญ
        - บทที่ 1 บทนำ (ชื่อ-สถานที่ตั้งสถานประกอบการ, ข้อมูลทั่วไป, ลักษณะสถานประกอบการ, รูปแบบการจัดการองค์กรและการบริหารงาน, ตำแหน่งและลักษณะงานที่ได้รับมอบหมายให้รับผิดชอบ, ชื่อและตำแหน่งงานของพนักงานที่ปรึกษา, ระยะเวลาที่ปฏิบัติงาน)
        - บทที่ 2 วัตถุประสงค์ ประโยชน์ที่คาดว่าจะได้รับ และแผนการปฏิบัติสหกิจศึกษา
        - บทที่ 3 รายงานการปฏิบัติงานที่ได้รับมอบหมาย (การศึกษาข้อมูล , งานที่ได้รับมอบหมาย, ขั้นตอนในการปฏิบัติงาน, ข้อควรระวังในการปฏิบัติงาน, กิจกรรมที่เข้าร่วม)
        - บทที่ 4 โครงงาน หรืองานพิเศษที่ได้รับมอบหมาย (ที่มาของระบบ, วัตถุประสงค์ของระบบ, ผลที่คาดว่าจะได้รับ, ขอบเขตการศึกษา, รูปแบบการศึกษา, วิธีการศึกษา, ผลการศึกษา, ประโยชน์ที่ได้รับจากการทำโครงงาน, อุปสรรคและข้อจำกัดในการทำงาน)
        - บรรณานุกรม
        - ภาคผนวก
        
        [ข้อมูลเพิ่มเติม ตารางแสดงแผนการปฏิบัติสหกิจศึกษา]
        คำอธิบาย : ตารางนี้อยู่ในบทที่ 2 
        ข้อมูลตาราง : 
        | ลำดับ | หัวข้องาน | ช่วงเวลาดำเนินงาน |
        |---|---|---|
        | 1 | ศึกษาโครงสร้างโรงพยาบาลและรายละเอียดงานสารสนเทศ | สัปดาห์ที่ 4 ของเดือนเมษายน |
        | 2 | ศึกษาการใช้งาน RPA (Robotic Process Automation) | สัปดาห์ที่ 1-2 ของเดือนพฤษภาคม |
        | 3 | ศึกษาวิธีการดึงข้อมูล (Query) จากฐานข้อมูลโรงพยาบาล | สัปดาห์ที่ 3-4 ของเดือนพฤษภาคม |
        | 4 | ศึกษาองค์ความรู้เกี่ยวกับหัวข้อ AI Chatbot Assistant | ตลอดเดือนมิถุนายน (สัปดาห์ที่ 1-4) |
        | 5 | เริ่มดำเนินงานโครงงาน AI ChatBot Assistant | ตลอดเดือนกรกฎาคม (สัปดาห์ที่ 1-4) |
        | 6 | เข้าร่วมประชุมกับผู้เกี่ยวข้องเพื่อขอความร่วมมือเป็นที่ปรึกษาในการดำเนินงาน | สัปดาห์ที่ 3 ของเดือนกรกฎาคม |
        | 7 | รายงานความคืบหน้าแก่งานพี่พนักงานที่ปรึกษา | สัปดาห์ที่ 1 ของเดือนสิงหาคม |
        
        Context:\n {context}?\n

        Question: \n{question}\n

        Answer:
        
        """
    else:
        # Default prompt
        return """
        You are an expert AI chatbot. Please answer the user's question based only on the provided context.
        If the answer is not found in the context, reply: "ไม่มีข้อมูลในเอกสารนี้ครับ ลองถามคำถามอื่นเพิ่มเติมครับ" and do not guess or provide incorrect information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
def get_conversational_chain(category):
    prompt_template = get_prompt_by_category(category)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

        
def user_input(user_question, category):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) #เพิ่ม allow_dangerous_deserialization
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(category)

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    return response["output_text"]  # Return the output text for further processing or display


def main():
    st.set_page_config("TUTHINK-PDF", page_icon=":computer:")
    st.header("TUTHINK - Chatbot 📋🗂️🏥")

    pdf_options = {
        "สวัสดิการด้านสุขภาพ📁": "docs\Rules.pdf",
        "วินัยลูกจ้าง📁": "docs\วินัยลูกจ้าง.pdf",
        "รายงานสหกิจ":"docs\\65104788_เบญญาภา_ถนอมใจ.pdf"
    }

    with st.sidebar:
        st.image("img/chatbot.jpg")
        st.write("---")
        
        st.title("About TUTHINK")
        st.markdown("📖 TUTHINK เป็นแอปพลิเคชันที่ช่วยตอบคำถามเกี่ยวกับเอกสาร PDF")


        st.write("---")
        select_pdf = st.selectbox("เลือกหมวดหมู่เอกสารที่ต้องการถาม",list(pdf_options.keys()))    

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
    if "vector_store" not in st.session_state or st.session_state.select_pdf != select_pdf:
    # บังคับให้ user ถามคำถามจาก PDF ที่กำหนดไว้เท่านั้น
        with st.spinner("กำลังเริ่มต้นและประมวลผลเอกสาร PDF ครับ..."):
            #predefined_pdf_path = "docs\Rules.pdf"  # Path to the embedded PDF file
            #with open(predefined_pdf_path, "rb") as pdf_file:  # rb คือ read binary อ่านข้อมูลจากไฟล์ PDFที่เป็น binary
            with open(pdf_options[select_pdf],"rb") as pdf_file:
                raw_text = get_pdf_text([pdf_file])  # Process the predefined PDF
                text_chunks = get_text_chunks(raw_text)  # Get text chunks
                get_vector_store(text_chunks)  # Create vector store
            st.session_state.vector_store = True # บันทึกสถานะเป็น True เมื่อประมวลผลเสร็จแล้ว
            st.session_state.select_pdf = select_pdf
            st.toast("ประมวลผล PDF เสร็จเรียบร้อย!", icon="✅")  # ใช้ st.toast แสดงข้อความสำเร็จ
    st.info(f"คุณกำลังถามคำถามจากเอกสาร: {select_pdf} 📁")  # แสดงชื่อเอกสารที่เลือก
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
            response = user_input(prompt ,select_pdf)  # ประมวลผลคำถาม
            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.messages.append({"role" : "assistant", "content":response}) # บันทึกคำตอบของ AI ใน chat_history

if __name__ == "__main__":
    main()