from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st
import os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, openai_api_key):
    file_content = file.read()

    # 디렉토리 미리 생성
    os.makedirs("./.cache/files", exist_ok=True)
    os.makedirs("./.cache/embeddings", exist_ok=True)

    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})

def load_memory(_):
    return memory_tmp.load_memory_variables({})["history"]

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files and enter your OpenAI API key on the sidebar.
"""
)

with st.sidebar:
    st.markdown("[🔗 Git Repo Link](https://github.com/geunsu-son/fullstack-gpt)")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file and openai_api_key:
    retriever = embed_file(file, openai_api_key)

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        openai_api_key=openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                
                Context: {context}
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        memory_tmp = st.session_state["memory"]
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(load_memory),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
            save_memory(message, response.content)

else:
    st.session_state["messages"] = []