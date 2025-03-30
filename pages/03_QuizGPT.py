import streamlit as st
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever

# Streamlit 설정
st.set_page_config(page_title="QuizGPT", page_icon="❓")
st.title("QuizGPT")

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

# 사이드바 설정
with st.sidebar:
    st.markdown("[🔗 Git Repo Link](https://github.com/geunsu-son/fullstack-gpt)")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    difficulty = st.selectbox("난이도를 선택하세요", ["Eazy", "Difficult"])
    choice = st.selectbox("퀴즈 생성 방식 선택", ["Wikipedia", "파일 업로드"])

    docs = None
    if choice == "파일 업로드":
        file = st.file_uploader("파일을 업로드하세요 (.txt, .pdf, .docx)", type=["pdf", "txt", "docx"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Wikipedia에서 검색할 주제 입력")
        if topic:
            docs = wiki_search(topic)

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar")
    st.stop()

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.

        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
    st.stop()

# 퀴즈 생성 프롬프트
quiz_prompt = PromptTemplate.from_template("Make a {difficulty} quiz based on the following context:\n{context}")

# 함수 정의 및 LLM 바인딩
quiz_function = {
    "name": "create_quiz",
    "description": "Generate quiz with specified difficulty level.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

# LLM 설정
llm = ChatOpenAI(api_key=openai_api_key, temperature=0.1).bind(
    function_call={"name": "create_quiz"}, functions=[quiz_function]
)

quiz_chain = quiz_prompt | llm

cache_dir = './.cache/quiz_files'
os.makedirs(cache_dir, exist_ok=True)

if "response_to_json" not in st.session_state:
    context = "\n".join([doc.page_content for doc in docs])
    response = quiz_chain.invoke({"difficulty": difficulty, "context": context})
    st.session_state.response_to_json = json.loads(response.additional_kwargs["function_call"]["arguments"])

cache_file_path = os.path.join(cache_dir, "latest_quiz.json")
with open(cache_file_path, "w") as cache_file:
    json.dump(st.session_state.response_to_json, cache_file, ensure_ascii=False, indent=4)

with st.form("quiz_form"):
    correct_answers = 0
    for idx, question in enumerate(st.session_state.response_to_json["questions"]):
        user_answer = st.radio(
            f"{idx+1}. {question['question']}",
            [answer["answer"] for answer in question["answers"]],
            index=None,
            key=f"question_{idx}"
        )
        if user_answer is not None:
            if next(ans for ans in question["answers"] if ans["answer"] == user_answer)["correct"]:
                correct_answers += 1

    submitted = st.form_submit_button("제출")

    if submitted:
        if correct_answers == len(st.session_state.response_to_json["questions"]):
            st.balloons()
            st.success("축하합니다! 만점입니다!")
        else:
            st.error(f"{correct_answers}/{len(st.session_state.response_to_json['questions'])} 맞췄습니다. 다시 시도해보세요!")
