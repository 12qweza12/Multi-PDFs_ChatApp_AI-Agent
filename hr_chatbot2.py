import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
import tempfile
import json
from datetime import datetime
from typing import List, Dict, Any
import uuid

# Page configuration
st.set_page_config(
    page_title="HR Assistant ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    height: 70vh;
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    background-color: #fafafa;
}

.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}

.user-bubble {
    background-color: #007bff;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    word-wrap: break-word;
}

.bot-message {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

.bot-bubble {
    background-color: #f1f3f4;
    color: #333;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 70%;
    word-wrap: break-word;
    border: 1px solid #e0e0e0;
}

.category-badge {
    background-color: #28a745;
    color: white;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    margin-bottom: 8px;
    display: inline-block;
}

.stTextInput > div > div > input {
    border-radius: 25px;
    border: 2px solid #e0e0e0;
    padding: 12px 20px;
}

.stButton > button {
    border-radius: 25px;
    border: none;
    background-color: #007bff;
    color: white;
    padding: 12px 24px;
    font-weight: bold;
}

.sidebar-content {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

class HRChatBot:
    def __init__(self):
        self.categories = {
            "Dress Code Regulations": "dress_code",
            "Salary and Promotion Policies": "salary_promotion", 
            "Leave and Absence Rules": "leave_absence"
        }
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_stores = {}
        self.conversation_memory = []
        
    def setup_gemini_api(self, api_key: str):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini API: {str(e)}")
            return False
    
    def process_pdf_documents(self, uploaded_files: List, category: str):
        """Process uploaded PDF documents and create vector embeddings"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                
                # Add metadata
                for page in pages:
                    page.metadata.update({
                        "category": category,
                        "source_file": uploaded_file.name,
                        "processed_date": datetime.now().isoformat()
                    })
                
                documents.extend(pages)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        if documents:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create or update vector store
            if category in self.vector_stores:
                # Add to existing vector store
                self.vector_stores[category].add_documents(chunks)
            else:
                # Create new vector store
                self.vector_stores[category] = FAISS.from_documents(chunks, self.embeddings)
            
            return len(chunks)
        return 0
    
    def retrieve_relevant_context(self, query: str, category: str = None, k: int = 5) -> List[Document]:
        """Retrieve relevant documents based on query"""
        relevant_docs = []
        
        if category and category in self.vector_stores:
            # Search in specific category
            docs = self.vector_stores[category].similarity_search(query, k=k)
            relevant_docs.extend(docs)
        else:
            # Search across all categories
            for cat_name, vector_store in self.vector_stores.items():
                docs = vector_store.similarity_search(query, k=k//len(self.vector_stores) + 1)
                relevant_docs.extend(docs)
        
        return relevant_docs[:k]
    
    def generate_response(self, query: str, category: str = None) -> Dict[str, Any]:
        """Generate response using Gemini API with context"""
        try:
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(query, category)
            
            # Prepare context from retrieved documents
            context_text = ""
            sources = []
            
            for doc in relevant_docs:
                context_text += f"\n--- Document from {doc.metadata.get('category', 'Unknown')} ---\n"
                context_text += doc.page_content + "\n"
                sources.append({
                    "category": doc.metadata.get('category', 'Unknown'),
                    "source_file": doc.metadata.get('source_file', 'Unknown'),
                    "page": doc.metadata.get('page', 'Unknown')
                })
            
            # Prepare conversation history for context
            conversation_context = ""
            if self.conversation_memory:
                conversation_context = "\n--- Previous Conversation Context ---\n"
                for msg in self.conversation_memory[-6:]:  # Last 3 exchanges
                    conversation_context += f"{msg['role']}: {msg['content']}\n"
            
            # Create comprehensive prompt
            prompt = f"""
You are an HR Assistant ChatBot designed to help employees with HR-related questions. 
You have access to organizational documents covering Dress Code Regulations, Salary and Promotion Policies, and Leave and Absence Rules.

CONTEXT FROM RELEVANT DOCUMENTS:
{context_text}

{conversation_context}

CURRENT USER QUESTION: {query}

INSTRUCTIONS:
1. Provide clear, accurate, and helpful answers based on the document context
2. If the information is not available in the documents, clearly state that
3. Offer practical advice and decision support when appropriate
4. Maintain a professional yet friendly tone
5. Reference specific policies or sections when relevant
6. If the query is ambiguous, ask clarifying questions
7. Consider the conversation history to maintain context

Please provide a comprehensive response to the user's question:
"""

            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Store in conversation memory
            self.conversation_memory.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat(),
                "category": category
            })
            
            self.conversation_memory.append({
                "role": "assistant", 
                "content": response.text,
                "timestamp": datetime.now().isoformat(),
                "sources": sources
            })
            
            return {
                "response": response.text,
                "sources": sources,
                "category": category,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": [],
                "category": category,
                "success": False,
                "error": error_msg
            }
    
    def clear_conversation(self):
        """Clear conversation memory"""
        self.conversation_memory = []

def main():
    st.title("🤖 HR Assistant ChatBot")
    st.markdown("*Ask me anything about HR policies - I'm here to help!*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HRChatBot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'gemini_configured' not in st.session_state:
        st.session_state.gemini_configured = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        st.markdown("### Gemini API Setup")
        api_key = st.text_input("Enter your Gemini API Key:", type="password", key="gemini_api_key")
        
        if api_key and not st.session_state.gemini_configured:
            if st.session_state.chatbot.setup_gemini_api(api_key):
                st.session_state.gemini_configured = True
                st.success("✅ Gemini API configured successfully!")
            else:
                st.error("❌ Failed to configure Gemini API")
        
        # Document Upload Section
        st.markdown("### 📄 Document Management")
        selected_category = st.selectbox(
            "Select Category for Document Upload:",
            options=list(st.session_state.chatbot.categories.keys()),
            key="upload_category"
        )
        
        uploaded_files = st.file_uploader(
            f"Upload PDF documents for {selected_category}:",
            type=['pdf'],
            accept_multiple_files=True,
            key=f"upload_{selected_category}"
        )
        
        if uploaded_files and st.button("📤 Process Documents"):
            with st.spinner("Processing documents..."):
                category_key = st.session_state.chatbot.categories[selected_category]
                chunks_created = st.session_state.chatbot.process_pdf_documents(uploaded_files, category_key)
                if chunks_created > 0:
                    st.success(f"✅ Processed {chunks_created} document chunks for {selected_category}")
                else:
                    st.error("❌ Failed to process documents")
        
        # Vector Store Status
        st.markdown("### 📊 Knowledge Base Status")
        for category, key in st.session_state.chatbot.categories.items():
            if key in st.session_state.chatbot.vector_stores:
                doc_count = st.session_state.chatbot.vector_stores[key].index.ntotal
                st.success(f"✅ {category}: {doc_count} documents")
            else:
                st.warning(f"⚠️ {category}: No documents")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chatbot.clear_conversation()
            st.session_state.chat_history = []
            st.success("Conversation cleared!")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Category selection for queries
        query_category = st.selectbox(
            "Select category (optional):",
            options=["All Categories"] + list(st.session_state.chatbot.categories.keys()),
            key="query_category"
        )
        
        # Chat display area
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="user-message">
                            <div class="user-bubble">
                                {message["content"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        category_badge = ""
                        if message.get("category"):
                            category_name = next((k for k, v in st.session_state.chatbot.categories.items() if v == message["category"]), message["category"])
                            category_badge = f'<div class="category-badge">{category_name}</div>'
                        
                        st.markdown(f"""
                        <div class="bot-message">
                            <div class="bot-bubble">
                                {category_badge}
                                {message["content"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; color: #666;">
                    👋 Welcome! Ask me any HR-related question to get started.
                    <br><br>
                    💡 <strong>Examples:</strong>
                    <br>• "What is the dress code policy for remote work?"
                    <br>• "How do I apply for annual leave?"  
                    <br>• "What are the criteria for salary promotion?"
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💡 Quick Actions")
        
        if st.button("📋 Dress Code Info", use_container_width=True):
            st.session_state.quick_query = "What are the main dress code regulations?"
        
        if st.button("💰 Salary Policies", use_container_width=True):
            st.session_state.quick_query = "Tell me about salary and promotion policies"
        
        if st.button("🏖️ Leave Policies", use_container_width=True):
            st.session_state.quick_query = "What are the leave and absence rules?"
        
        if st.button("❓ General Help", use_container_width=True):
            st.session_state.quick_query = "What kind of HR questions can you help me with?"
    
    # Chat input
    st.markdown("---")
    col_input, col_send = st.columns([4, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your HR question here...",
            key="user_input",
            placeholder="Ask me about dress codes, salaries, leave policies, or any HR topic...",
            value=st.session_state.get('quick_query', '')
        )
    
    with col_send:
        send_button = st.button("Send 📤", use_container_width=True)
    
    # Handle quick query
    if 'quick_query' in st.session_state:
        user_input = st.session_state.quick_query
        del st.session_state.quick_query
        send_button = True
    
    # Process user input
    if (send_button or user_input) and user_input.strip():
        if not st.session_state.gemini_configured:
            st.warning("⚠️ Please configure your Gemini API key in the sidebar first!")
            return
        
        if not any(st.session_state.chatbot.vector_stores.values()):
            st.warning("⚠️ Please upload some HR documents first to enable intelligent responses!")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate bot response
        with st.spinner("🤔 Thinking..."):
            category_key = None
            if query_category != "All Categories":
                category_key = st.session_state.chatbot.categories[query_category]
            
            response_data = st.session_state.chatbot.generate_response(user_input, category_key)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_data["response"],
                "category": response_data.get("category"),
                "sources": response_data.get("sources", [])
            })
        
        # Clear input and rerun to show new messages
        st.rerun()

if __name__ == "__main__":
    main()