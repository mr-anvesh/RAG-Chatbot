import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models_config import AVAILABLE_MODELS
from datetime import datetime, timedelta
from collections import deque
import time

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📚",
    layout="wide"
)

class RateLimiter:
    def __init__(self, daily_limit=20, minute_limit=3, minutes_window=1):
        self.daily_limit = daily_limit
        self.minute_limit = minute_limit
        self.minutes_window = minutes_window
        
        # Initialize session state for rate limiting
        if "daily_requests" not in st.session_state:
            st.session_state.daily_requests = []
        if "minute_requests" not in st.session_state:
            st.session_state.minute_requests = deque(maxlen=minute_limit)
        if "last_reset_date" not in st.session_state:
            st.session_state.last_reset_date = datetime.now().date()
    
    def _reset_if_new_day(self):
        current_date = datetime.now().date()
        if current_date > st.session_state.last_reset_date:
            st.session_state.daily_requests = []
            st.session_state.last_reset_date = current_date
    
    def _clean_old_requests(self):
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=self.minutes_window)
        
        # Remove requests older than the window
        while (st.session_state.minute_requests and 
               st.session_state.minute_requests[0] < minute_ago):
            st.session_state.minute_requests.popleft()
    
    def can_make_request(self):
        self._reset_if_new_day()
        self._clean_old_requests()
        
        # Check daily limit
        if len(st.session_state.daily_requests) >= self.daily_limit:
            remaining_time = datetime.now().date() + timedelta(days=1)
            st.error(f"Daily message limit reached ({self.daily_limit} messages). "
                    f"Please try again tomorrow.")
            return False
        
        # Check per-minute limit
        if len(st.session_state.minute_requests) >= self.minute_limit:
            oldest_request = st.session_state.minute_requests[0]
            wait_time = (oldest_request + timedelta(minutes=self.minutes_window) - 
                        datetime.now()).total_seconds()
            if wait_time > 0:
                st.error(f"Rate limit reached. Please wait {int(wait_time)} seconds before sending another message.")
                return False
        
        return True
    
    def add_request(self):
        current_time = datetime.now()
        st.session_state.daily_requests.append(current_time)
        st.session_state.minute_requests.append(current_time)
    
    def get_remaining_requests(self):
        self._reset_if_new_day()
        self._clean_old_requests()
        
        daily_remaining = self.daily_limit - len(st.session_state.daily_requests)
        minute_remaining = self.minute_limit - len(st.session_state.minute_requests)
        
        return daily_remaining, minute_remaining

class PDFChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.chunks = []
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.rate_limiter = RateLimiter(daily_limit=20, minute_limit=3, minutes_window=1)
        
    def process_pdfs(self, pdf_files):
        """Process multiple PDF files and create document vectors."""
        self.chunks = []
        
        # Extract text from PDFs
        for pdf in pdf_files:
            text = ""
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Split text into chunks (simple paragraph-based splitting)
            paragraphs = text.split('\n\n')
            self.chunks.extend([p.strip() for p in paragraphs if p.strip()])
        
        # Create document vectors
        self.document_vectors = self.vectorizer.fit_transform(self.chunks)
        return len(self.chunks)
    
    def get_relevant_context(self, query, k=3):
        """Get most relevant chunks for the query."""
        if not self.chunks:
            return ""
        
        # Get query vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors)
        
        # Get top k chunks
        top_indices = similarities[0].argsort()[-k:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_indices]
        
        return "\n\n".join(relevant_chunks)
    
    def get_model_response(self, messages, model_id):
        """Get response from the selected model."""
        # Check rate limits
        if not self.rate_limiter.can_make_request():
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost:8501")
        }
        
        data = {
            "model": model_id,
            "messages": messages
        }
        
        try:
            response = requests.post(
                url=self.api_url,
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                # Record the successful request
                self.rate_limiter.add_request()
                
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    if "message" in response_data["choices"][0]:
                        return response_data["choices"][0]["message"]["content"]
                    elif "text" in response_data["choices"][0]:
                        return response_data["choices"][0]["text"]
                elif "response" in response_data:
                    return response_data["response"]
                else:
                    st.error("Unexpected API response format")
                    return "I apologize, but I received an unexpected response format. Please try again."
            else:
                st.error(f"Error from API: {response.text}")
                return "I apologize, but I encountered an error. Please try again."
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

def create_message_container():
    return st.container()

def display_message(container, role, content, message_id=None):
    with container:
        if role == "user":
            st.write(f"🧑 **You:** {content}")
        else:
            st.write(f"🤖 **Assistant:** {content}")
            # Add reply button
            if st.button("↩️ Reply", key=f"reply_{message_id}"):
                st.session_state.replying_to = message_id
                st.session_state.reply_context = content

def main():
    st.title("💬 Chat with Your PDFs")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = False
    if "replying_to" not in st.session_state:
        st.session_state.replying_to = None
    if "reply_context" not in st.session_state:
        st.session_state.reply_context = None
    if "message_containers" not in st.session_state:
        st.session_state.message_containers = {}

    # Sidebar for model selection and file upload
    with st.sidebar:
        st.header("Settings")
        
        # Display rate limits
        daily_remaining, minute_remaining = st.session_state.chatbot.rate_limiter.get_remaining_requests()
        st.write("**Rate Limits:**")
        st.write(f"- Daily messages remaining: {daily_remaining}/20")
        st.write(f"- Messages per minute remaining: {minute_remaining}/3")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose a model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            help="Select the AI model to use for chat"
        )
        
        st.write(f"**Model Description:**  \n{AVAILABLE_MODELS[selected_model]['description']}")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    num_chunks = st.session_state.chatbot.process_pdfs(uploaded_files)
                    st.session_state.processed_files = True
                    st.success(f"Processed {len(uploaded_files)} files into {num_chunks} chunks!")

    # Main chat interface
    if not st.session_state.processed_files:
        st.info("Please upload and process some PDF files to start chatting!")
    else:
        # Show if replying to a message
        if st.session_state.replying_to is not None:
            st.info(f"Replying to: {st.session_state.reply_context[:100]}...")
            if st.button("Cancel Reply"):
                st.session_state.replying_to = None
                st.session_state.reply_context = None
                st.rerun()

        # Chat input
        user_question = st.text_input(
            "Ask a question or reply:" if st.session_state.replying_to else "Ask a question about your PDFs:"
        )
        
        if user_question:
            # Get relevant context
            context = st.session_state.chatbot.get_relevant_context(user_question)
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. "
                              "If you cannot find the answer in the context, say so."
                }
            ]
            
            # Add reply context if replying
            if st.session_state.replying_to is not None:
                messages.append({"role": "assistant", "content": st.session_state.reply_context})
                messages.append({"role": "user", "content": f"Regarding your previous response, {user_question}"})
            else:
                messages.append({
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {user_question}"
                })
            
            # Get model response
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_model_response(
                    messages,
                    AVAILABLE_MODELS[selected_model]["id"]
                )
                # Generate a unique message ID
                message_id = f"msg_{len(st.session_state.chat_history)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.chat_history.append({
                    "id": message_id,
                    "question": user_question,
                    "answer": response,
                    "reply_to": st.session_state.replying_to
                })
                
                # Reset reply state
                st.session_state.replying_to = None
                st.session_state.reply_context = None
                
                # Rerun to update the UI
                st.rerun()

        # Display chat history with reply buttons
        for msg in st.session_state.chat_history:
            # Create a new container for each message pair if it doesn't exist
            if msg["id"] not in st.session_state.message_containers:
                st.session_state.message_containers[msg["id"]] = create_message_container()
            
            container = st.session_state.message_containers[msg["id"]]
            with container:
                # Show reply-to context if this is a reply
                if msg["reply_to"]:
                    st.markdown("*↳ Replying to previous message*")
                
                # Display messages
                display_message(container, "user", msg["question"])
                display_message(container, "assistant", msg["answer"], msg["id"])
                st.write("---")

if __name__ == "__main__":
    main() 
