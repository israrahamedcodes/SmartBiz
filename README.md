# 🚀 SmartBIZ — AI-Powered Business Intelligence Platform

SmartBIZ is an advanced AI-powered Business Intelligence (BI) platform that transforms raw CSV data into actionable insights, interactive dashboards, and intelligent business recommendations using a Retrieval-Augmented Generation (RAG) pipeline.

It combines **Data Science + Machine Learning + Generative AI** to deliver enterprise-grade analytics with a natural language chatbot interface.

---

## 📊 Demo Preview

![SmartBIZ Banner](https://via.placeholder.com/1200x400/0b1326/ffffff?text=SmartBIZ+AI+Powered+BI+Dashboard)

---

## ✨ Key Features

### 🧹 Automated Data Processing
- Smart data cleaning using MCP pipeline
- Handles missing values, duplicates, and outliers
- Automatic datatype detection and formatting

### 📈 AI-Powered Dashboards
- Auto-generated business visualizations
- Revenue trends, product performance, KPI analysis
- Interactive charts using Chart.js

### 🤖 RAG-Based AI Chatbot
- Ask questions in natural language
- Dataset-grounded responses using vector search
- Example queries:
  - "Top selling products?"
  - "Monthly revenue trend?"

### 🔁 Multi-LLM Support
- Primary: Groq API
- Fallback: OpenRouter API
- Ensures high availability of AI responses

### 🧠 Smart Insights Engine
- Automatically generates business insights
- Detects anomalies and trends
- Provides actionable recommendations

### 📂 Dataset History
- Stores previously uploaded datasets
- Revisit reports and analytics anytime

---

## 🛠️ Tech Stack

### Backend
- Python (Flask)
- Pandas, NumPy
- ChromaDB (Vector Database)
- MongoDB Atlas
- SentenceTransformers (Embeddings)
- Groq & OpenRouter APIs

### Frontend
- HTML5, CSS3
- Vanilla JavaScript (ES6+)
- Chart.js
- Google Material Icons

---

## 🏗️ System Architecture
CSV Upload
↓
Data Cleaning (MCP Pipeline)
↓
Feature Engineering (Pandas)
↓
Embedding Generation (Sentence Transformers)
↓
Vector Storage (ChromaDB)
↓
RAG Retrieval Layer
↓
LLM Processing (Groq / OpenRouter)
↓
Insights + Chatbot + Dashboard


---

## 📦 Installation

### 1. Prerequisites
- Python 3.9+
- MongoDB Atlas (or local MongoDB)
- Groq API Key
- OpenRouter API Key (optional)

---

### 2. Clone Repository
```bash```
git clone <repository-url>
cd SmartBIZ

### 3. Install Dependencies
pip install -r backend/requirements.txt

### 4. Environment Setup

Create a .env file:

GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_secret_key
UPLOAD_FOLDER=../data
CHROMA_DB_PATH=../data/chroma_db

### ▶️ Run the Project
Start Backend Server
python run.py

Open in browser:

http://127.0.0.1:5000
💡 How It Works
- Upload CSV dataset
- System cleans and preprocesses data automatically
- Embeddings are generated for semantic search
- Data stored in ChromaDB vector database
- RAG pipeline connects data with LLM
- Dashboard + chatbot generate insights

### 📊 Use Cases
- Sales analytics dashboards
- Product performance tracking
- Revenue forecasting
- Customer behavior analysis
- Business intelligence reporting

### 🔐 Security Features
- JWT authentication
- Secure API endpoints
- Local vector database storage
- No full dataset exposure to LLM

### 📁 Project Structure
SmartBIZ/
│
├── backend/
│   ├── app.py
│   ├── run.py
│   ├── requirements.txt
│   ├── routes/
│   ├── services/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│
├── data/
│   ├── uploads/
│   ├── chroma_db/
│
└── README.md
