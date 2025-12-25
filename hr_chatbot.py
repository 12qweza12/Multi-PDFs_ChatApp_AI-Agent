import streamlit as st
import os
from datetime import datetime
import json
from typing import List, Dict, Any
import time

# Mock LangChain and document processing (since we can't install actual packages)
# In a real implementation, you would use:
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import OpenAI

class MockDocument:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.page_content = content
        self.metadata = metadata

class MockVectorStore:
    def __init__(self):
        # Mock HR knowledge base
        self.documents = {
            "dress_code": [
                MockDocument(
                    "Business casual attire is required Monday through Thursday. Casual dress is permitted on Fridays. "
                    "Professional attire includes collared shirts, dress pants, blouses, and closed-toe shoes. "
                    "Prohibited items include shorts, flip-flops, tank tops, and clothing with offensive graphics.",
                    {"category": "dress_code", "section": "general_guidelines"}
                ),
                MockDocument(
                    "Remote work dress code: Business casual from waist up for video calls. "
                    "Full professional attire required for client meetings regardless of location.",
                    {"category": "dress_code", "section": "remote_work"}
                ),
                MockDocument(
                    "Safety requirements: Closed-toe shoes mandatory in manufacturing areas. "
                    "High-visibility vests required in warehouse zones. Hard hats required in construction sites.",
                    {"category": "dress_code", "section": "safety"}
                )
            ],
            "salary_promotion": [
                MockDocument(
                    "Annual performance reviews conducted in Q1. Salary adjustments effective April 1st. "
                    "Merit increases range from 2-8% based on performance ratings. "
                    "Promotion eligibility requires minimum 12 months in current role.",
                    {"category": "salary_promotion", "section": "performance_review"}
                ),
                MockDocument(
                    "Promotion criteria: Consistent high performance, demonstrated leadership, skill development, "
                    "and availability of position. Internal candidates given preference for open positions. "
                    "Promotion applications must include manager recommendation.",
                    {"category": "salary_promotion", "section": "promotion_process"}
                ),
                MockDocument(
                    "Salary bands reviewed annually. Market rate adjustments considered for retention. "
                    "Bonus eligibility based on company performance and individual contribution. "
                    "Stock options available for senior positions after 2 years tenure.",
                    {"category": "salary_promotion", "section": "compensation"}
                )
            ],
            "leave_absence": [
                MockDocument(
                    "Vacation leave: 15 days annually for first 2 years, 20 days for 3-7 years, 25 days for 8+ years. "
                    "Sick leave: 10 days annually, can be carried over up to 40 days maximum. "
                    "Personal leave: 3 days annually for personal matters.",
                    {"category": "leave_absence", "section": "paid_leave"}
                ),
                MockDocument(
                    "Maternity leave: 12 weeks paid leave. Paternity leave: 4 weeks paid leave. "
                    "Adoption leave: 8 weeks paid leave. Extended unpaid leave available up to 6 months "
                    "with medical certification or exceptional circumstances.",
                    {"category": "leave_absence", "section": "family_leave"}
                ),
                MockDocument(
                    "Leave request process: Submit requests 2 weeks in advance via HR portal. "
                    "Emergency leave can be requested same day with manager approval. "
                    "Leave balance visible in employee self-service portal.",
                    {"category": "leave_absence", "section": "process"}
                )
            ]
        }
    
    def similarity_search(self, query: str, category: str = None, k: int = 3) -> List[MockDocument]:
        # Simple keyword matching for mock implementation
        query_lower = query.lower()
        relevant_docs = []
        
        categories_to_search = [category] if category else ["dress_code", "salary_promotion", "leave_absence"]
        
        for cat in categories_to_search:
            if cat in self.documents:
                for doc in self.documents[cat]:
                    # Simple relevance scoring based on keyword matches
                    content_lower = doc.page_content.lower()
                    keywords = query_lower.split()
                    score = sum(1 for keyword in keywords if keyword in content_lower)
                    if score > 0:
                        relevant_docs.append((doc, score))
        
        # Sort by relevance score and return top k
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in relevant_docs[:k]]

class HRChatBot:
    def __init__(self):
        self.vector_store = MockVectorStore()
        self.conversation_history = []
    
    def get_response(self, query: str, category: str = None, chat_history: List = None) -> str:
        # Simulate processing time
        time.sleep(1)
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query, category)
        
        if not relevant_docs:
            return "I couldn't find specific information about your question in our HR documents. Please try rephrasing your question or contact HR directly for assistance."
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Simple response generation based on context
        # In real implementation, this would use an LLM
        response = self.generate_response(query, context, chat_history)
        
        return response
    
    def generate_response(self, query: str, context: str, chat_history: List = None) -> str:
        # Mock response generation - in real implementation, use LLM
        query_lower = query.lower()
        
        # Dress code responses
        if any(word in query_lower for word in ['dress', 'attire', 'clothing', 'wear', 'uniform']):
            if 'friday' in query_lower or 'casual' in query_lower:
                return "According to our dress code policy, casual dress is permitted on Fridays. However, business casual is required Monday through Thursday. For remote work, business casual from waist up is required for video calls, and full professional attire is needed for client meetings."
            elif 'remote' in query_lower or 'work from home' in query_lower:
                return "For remote work, our dress code requires business casual attire from waist up during video calls. When attending client meetings, full professional attire is required regardless of your location."
            elif 'safety' in query_lower or 'warehouse' in query_lower or 'manufacturing' in query_lower:
                return "Safety dress requirements include: closed-toe shoes mandatory in manufacturing areas, high-visibility vests in warehouse zones, and hard hats in construction sites. These are non-negotiable safety requirements."
            else:
                return "Our dress code requires business casual attire Monday through Thursday, with casual dress permitted on Fridays. Professional attire includes collared shirts, dress pants, blouses, and closed-toe shoes. Prohibited items include shorts, flip-flops, tank tops, and clothing with offensive graphics."
        
        # Salary and promotion responses
        elif any(word in query_lower for word in ['salary', 'promotion', 'raise', 'increase', 'bonus', 'pay']):
            if 'promotion' in query_lower:
                return "Promotion eligibility requires a minimum of 12 months in your current role. The criteria include consistent high performance, demonstrated leadership, skill development, and position availability. Internal candidates are given preference, and you'll need a manager recommendation for your application."
            elif 'review' in query_lower or 'performance' in query_lower:
                return "Annual performance reviews are conducted in Q1, with salary adjustments effective April 1st. Merit increases range from 2-8% based on your performance rating. The review evaluates your contributions and determines eligibility for raises and bonuses."
            elif 'bonus' in query_lower:
                return "Bonus eligibility is based on both company performance and your individual contribution. We also offer stock options for senior positions after 2 years of tenure. Salary bands are reviewed annually with market rate adjustments considered for retention."
            else:
                return "Salary adjustments are made annually following performance reviews in Q1, effective April 1st. Merit increases range from 2-8% based on performance. We also consider market rate adjustments for retention and offer bonuses based on company and individual performance."
        
        # Leave and absence responses
        elif any(word in query_lower for word in ['leave', 'vacation', 'sick', 'time off', 'absence', 'holiday']):
            if 'maternity' in query_lower or 'paternity' in query_lower or 'family' in query_lower:
                return "We offer comprehensive family leave: 12 weeks paid maternity leave, 4 weeks paid paternity leave, and 8 weeks paid adoption leave. Extended unpaid leave up to 6 months is available with medical certification or exceptional circumstances."
            elif 'sick' in query_lower:
                return "You receive 10 sick days annually, which can be carried over up to a maximum of 40 days. This leave is for your health needs and can be used when you're unable to work due to illness."
            elif 'vacation' in query_lower or 'personal' in query_lower:
                return "Vacation leave allocation depends on tenure: 15 days annually for first 2 years, 20 days for 3-7 years, and 25 days for 8+ years. You also get 3 personal days annually for personal matters. Submit requests 2 weeks in advance via the HR portal."
            else:
                return "Leave requests should be submitted 2 weeks in advance through the HR portal. Emergency leave can be requested same day with manager approval. You can check your leave balance in the employee self-service portal."
        
        # General response
        else:
            return f"Based on our HR policies, here's what I found relevant to your question:\n\n{context[:500]}{'...' if len(context) > 500 else ''}\n\nWould you like more specific information about any particular aspect?"

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = HRChatBot()

def main():
    st.set_page_config(
        page_title="HR Assistant ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Custom CSS for ChatGPT-like interface
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #f0f2f6;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    
    .chat-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .category-button {
        margin: 0.2rem;
        padding: 0.5rem 1rem;
        border: 1px solid #ddd;
        border-radius: 20px;
        background-color: #f8f9fa;
        color: #333;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .category-button:hover {
        background-color: #667eea;
        color: white;
        text-decoration: none;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="chat-header">
        <h1>🤖 HR Assistant ChatBot</h1>
        <p>Your intelligent assistant for HR policies and procedures</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Quick Categories")
        
        col1, col2 = st.columns(2)
        
        categories = {
            "👔 Dress Code": "dress_code",
            "💰 Salary & Promotion": "salary_promotion", 
            "🏖️ Leave & Absence": "leave_absence"
        }
        
        selected_category = None
        
        if st.button("👔 Dress Code", use_container_width=True):
            selected_category = "dress_code"
        if st.button("💰 Salary & Promotion", use_container_width=True):
            selected_category = "salary_promotion"
        if st.button("🏖️ Leave & Absence", use_container_width=True):
            selected_category = "leave_absence"
        
        st.markdown("---")
        st.markdown("### 💡 Quick Questions")
        
        quick_questions = [
            "What is the dress code policy?",
            "How do I request vacation leave?",
            "When are performance reviews?",
            "What are the promotion requirements?",
            "How much sick leave do I have?",
            "Can I wear casual clothes?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.get_response(
                        question, 
                        selected_category, 
                        st.session_state.messages
                    )
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ Information")
        st.info("This chatbot provides information based on company HR policies. For official matters, please contact HR directly.")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about HR policies..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.get_response(
                        prompt, 
                        selected_category, 
                        st.session_state.messages
                    )
                st.markdown(response)
            
            # Add assistant response to messages
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.markdown("### 📊 Chat Statistics")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("User Questions", len([m for m in st.session_state.messages if m["role"] == "user"]))
        
        st.markdown("### 🕒 Recent Activity")
        if st.session_state.messages:
            last_message_time = datetime.now().strftime("%H:%M")
            st.write(f"Last message: {last_message_time}")
        else:
            st.write("No messages yet")
        
        st.markdown("### 🎯 Tips")
        st.markdown("""
        - Be specific in your questions
        - Use category buttons for quick access
        - Ask follow-up questions for clarity
        - Check the sidebar for common questions
        """)

if __name__ == "__main__":
    main()