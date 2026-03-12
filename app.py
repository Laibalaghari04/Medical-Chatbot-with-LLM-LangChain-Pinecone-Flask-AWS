from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
app.secret_key = "medibot-secret-key"

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="gpt-4o")

# Updated prompt with chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Store chat histories per session
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = os.urandom(16).hex()
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    session_id = session.get("session_id", "default")
    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}},
    )
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)