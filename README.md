# Chat with PDF

A Streamlit application that allows you to chat with your PDF documents using RAG (Retrieval Augmented Generation) and free AI models through OpenRouter and HuggingFace.

## 🌟 Features

- Upload multiple PDF documents
- Extract and process text from PDFs
- Chat interface to ask questions about your documents
- Conversation memory to maintain context
- Uses free AI models (Deepseek through OpenRouter)
- HuggingFace embeddings for semantic search

## 🚀 Demo

Try the app here: https://am-rag-chatbot.streamlit.app/

## 💻 Local Development

1. Clone the repository
```bash
git clone https://github.com/mr-anvesh/RAG-Chatbot.git
cd RAG-Chatbot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenRouter API key
```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_REFERRER=http://localhost:8501
```

4. Run the app
```bash
streamlit run app.py
```

## 🔑 Environment Variables

For deployment on Streamlit Cloud, set the following secrets:

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_REFERRER`: Your Streamlit Cloud app URL

## 📝 License

MIT

## 🤝 Contributing

Pull requests are welcome! 
