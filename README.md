# 🚀 SmartBIZ — AI-Powered Business Intelligence Platform

SmartBIZ is an advanced AI-powered Business Intelligence (BI) platform that transforms raw CSV data into actionable insights, interactive dashboards, and intelligent business recommendations using a Retrieval-Augmented Generation (RAG) pipeline.

It combines **Data Science + Machine Learning + Generative AI** to deliver enterprise-grade analytics with a natural language chatbot interface.


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

A[CSV Upload] --> B[Data Cleaning<br>(MCP Pipeline)]
B --> C[Feature Engineering<br>(Pandas)]
C --> D[Embedding Generation<br>(Sentence Transformers)]
D --> E[Vector Storage<br>(ChromaDB)]
E --> F[RAG Retrieval Layer]
F --> G[LLM Processing<br>(Groq / OpenRouter)]
G --> H[Insights Engine]
H --> I[Chatbot Interface]
H --> J[Dashboard Visualizations]


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


A[SmartBIZ]

A --> B[backend/]
B --> B1[app.py]
B --> B2[run.py]
B --> B3[requirements.txt]
B --> B4[routes/]
B --> B5[services/]

A --> C[frontend/]
C --> C1[index.html]
C --> C2[style.css]
C --> C3[app.js]

A --> D[data/]
D --> D1[uploads/]
D --> D2[chroma_db/]

A --> E[README.md]
