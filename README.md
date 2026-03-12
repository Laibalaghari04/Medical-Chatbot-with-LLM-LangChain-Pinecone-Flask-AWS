# Medical Chatbot with LLM, LangChain, Pinecone, Flask & AWS

An AI-powered medical chatbot built using Retrieval-Augmented Generation (RAG) with **Conversation Buffer Memory**. It answers medical questions based on the Gale Encyclopedia of Medicine using OpenAI GPT-4 and Pinecone vector search. The chatbot remembers previous messages in the same session for natural follow-up conversations.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| LangChain | LLM orchestration & RAG pipeline |
| OpenAI GPT-4 | Language model |
| Pinecone | Vector database for embeddings |
| HuggingFace | Sentence embeddings (all-MiniLM-L6-v2) |
| Flask | Web framework |
| LangChain Memory | Conversation Buffer Memory for chat history |
| AWS EC2 + ECR | Cloud deployment |
| Docker | Containerization |
| GitHub Actions | CI/CD pipeline |

---

## Features

- **RAG Pipeline** — Retrieves relevant medical context from Pinecone before answering
- **Conversation Buffer Memory** — Remembers full chat history within a session for follow-up questions
- **GPT-4 Powered** — Uses OpenAI GPT-4 for accurate, intelligent responses
- **Medical Knowledge Base** — Based on the Gale Encyclopedia of Medicine
- **Clean Chat UI** — Dark/Light mode with a modern chat interface

---

## How to Run Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/Laibalaghari04/Medical-Chatbot-with-LLM-LangChain-Pinecone-Flask-AWS.git
cd Medical-Chatbot-with-LLM-LangChain-Pinecone-Flask-AWS
```

### Step 2: Create Conda Environment
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### Step 3: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables
Create a `.env` file in the root directory:
```
PINECONE_API_KEY = "your_pinecone_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"
```

### Step 5: Store Embeddings to Pinecone
```bash
python store_index.py
```

### Step 6: Run the App
```bash
python app.py
```

Now open your browser and go to: **http://localhost:5003**

---

## How Conversation Buffer Memory Works

The chatbot uses `RunnableWithMessageHistory` from LangChain to maintain a full conversation history per user session. This means:

- You can ask **follow-up questions** without repeating context
- Each browser session gets its **own chat history**
- The memory resets when you **close or refresh** the browser

**Example:**
```
You:  "What is diabetes?"
Bot:  "Diabetes is a chronic condition..."

You:  "What are its symptoms?"        ← Bot remembers you meant diabetes!
Bot:  "Symptoms of diabetes include..."
```

---

## AWS Deployment with GitHub Actions (CI/CD)

### Step 1: Create IAM User
- Go to AWS Console → IAM → Create User
- Attach these policies:
  - `AmazonEC2ContainerRegistryFullAccess`
  - `AmazonEC2FullAccess`
- Save the **Access Key ID** and **Secret Access Key**

### Step 2: Create ECR Repository
- Go to AWS Console → ECR → Create Repository
- Save the URI, e.g.:
```
315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot
```

### Step 3: Create EC2 Instance
- Launch an **Ubuntu** EC2 instance
- Allow inbound traffic on port **5001**

### Step 4: Install Docker on EC2
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 5: Configure EC2 as Self-Hosted GitHub Runner
- Go to your GitHub repo → **Settings → Actions → Runners**
- Click **New Self-Hosted Runner**
- Choose **Linux** and run the commands one by one on your EC2 instance

### Step 6: Add GitHub Secrets
Go to your GitHub repo → **Settings → Secrets and Variables → Actions** and add:

| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your IAM access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret key |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `ECR_REPO` | Your ECR URI |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `OPENAI_API_KEY` | Your OpenAI API key |

---

## Project Structure

```
Medical-Chatbot/
├── data/                   # Medical PDF data
├── src/
│   ├── helper.py           # PDF loading, chunking, embeddings
│   └── prompt.py           # System prompt for chatbot
├── static/
│   └── style.css           # Frontend styles
├── templates/
│   └── index.html          # Chat UI
├── research/
│   └── trials.ipynb        # Jupyter notebook experiments
├── app.py                  # Flask application with memory
├── store_index.py          # Store embeddings to Pinecone
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── .env                    # API keys (never push this!)
```

---

## Important Notes
- The PDF data file is excluded from git due to size
- Make sure your Pinecone index is created before running the app
- Conversation memory resets on browser refresh (session-based)