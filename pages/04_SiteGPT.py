from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from langchain.document_loaders import WebBaseLoader
import os
from datetime import datetime

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("Cloudflare AI Assistant")
st.markdown("""
이 AI 어시스턴트는 Cloudflare의 AI 제품들에 대한 질문에 답변할 수 있습니다:
- AI Gateway
- Vectorize
- Workers AI

문서 출처: [Cloudflare Developers](https://developers.cloudflare.com/)
""")

with st.sidebar:
    st.markdown("[🔗 Git Repo Link](https://github.com/geunsu-son/fullstack-gpt)")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if st.button("벡터 저장소 새로고침"):
        if os.path.exists("./.cache/vector_store"):
            import shutil
            shutil.rmtree("./.cache/vector_store")
        st.cache_data.clear()
        st.success("벡터 저장소가 삭제되었습니다. 페이지를 새로고침하면 문서를 다시 로드합니다.")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar")
    st.stop()

# Initialize LLM with the API key from sidebar
llm = ChatOpenAI(
    temperature=0.1,
    api_key=openai_api_key,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant specialized in Cloudflare's AI products documentation. Using ONLY the following context, answer the user's question.
    If you can't find the answer in the context, just say "I don't know" - do not make up information.
    
    Always mention which Cloudflare product (AI Gateway, Vectorize, or Workers AI) you're referring to in your answer.
    Include relevant pricing, limits, or technical specifications when available.
    
    Context: {context}
    
    Question: {question}
    
    Score (0-5): Rate how well the context helps answer the question.
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata.get("source", "Unknown"),
                # lastmod가 없을 경우 현재 날짜 사용
                "date": doc.metadata.get("lastmod", datetime.now().strftime("%Y-%m-%d")),
                "product": doc.metadata.get("product", "Unknown")
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Cloudflare AI products documentation expert. Use ONLY the following pre-existing answers to respond to the user's question.
            
            Focus on answers with the highest score (most relevant) and most recent dates.
            Always include the specific product name (AI Gateway, Vectorize, or Workers AI) in your response.
            
            When available, include:
            - Pricing information
            - Usage limits
            - Technical specifications
            - Implementation details
            
            Cite sources using their URLs and include the last modification dates for transparency.
            Format URLs as markdown links: [Product Name](URL)
            
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    
    # Sort answers by score and date
    condensed = "\n\n".join(
        f"Product: {answer.get('product', 'Unknown')}\n{answer['answer']}\nSource: [{answer['source']}]({answer['source']})\nLast modified: {answer['date']}\n"
        for answer in answers
    )
    
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

@st.cache_data(show_spinner="Loading Cloudflare documentation...")
def load_cloudflare_docs():
    try:
        # 저장된 벡터 저장소 확인
        vector_store_dir = "./.cache/vector_store"
        os.makedirs(vector_store_dir, exist_ok=True)
        vector_store_path = os.path.join(vector_store_dir, "cloudflare_docs_store")
        
        if os.path.exists(vector_store_path):
            st.write("저장된 벡터 저장소를 불러오는 중...")
            vector_store = FAISS.load_local(vector_store_path, OpenAIEmbeddings(openai_api_key=openai_api_key))
            return vector_store.as_retriever(search_kwargs={"k": 4})
        
        # 1. sitemap에서 URL 목록 가져오기
        sitemap_url = 'https://developers.cloudflare.com/sitemap-0.xml'
        
        response = requests.get(sitemap_url)
        root = ET.fromstring(response.content)
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # URL과 lastmod 정보 함께 가져오기
        urls_with_dates = []
        for url in root.findall('.//ns:loc', namespaces):
            lastmod = url.find('../ns:lastmod', namespaces)
            lastmod_date = lastmod.text if lastmod is not None else datetime.now().strftime("%Y-%m-%d")
            urls_with_dates.append((url.text, lastmod_date))
        
        # 2. URL 필터링
        filter_patterns = ["ai-gateway", "vectorize", "workers-ai"]
        filtered_urls = [(url, date) for url, date in urls_with_dates 
                        if any(pattern in url for pattern in filter_patterns)]
        
        # 3. 필터링된 URL에서 문서 수집
        all_docs = []
        status_container = st.empty()
        
        for idx, (url, lastmod_date) in enumerate(filtered_urls):
            status_container.text(f"문서 로딩 중... ({idx + 1}/{len(filtered_urls)})")
            try:
                loader = WebBaseLoader(
                    url,
                    default_parser="lxml",
                    bs_kwargs={"parser": "lxml", "features": "lxml"}
                )
                docs = loader.load()
                
                # 메타데이터 추가
                for doc in docs:
                    doc.metadata["lastmod"] = lastmod_date
                    if "ai-gateway" in url:
                        doc.metadata["product"] = "AI Gateway"
                    elif "vectorize" in url:
                        doc.metadata["product"] = "Cloudflare Vectorize"
                    elif "workers-ai" in url:
                        doc.metadata["product"] = "Workers AI"
                all_docs.extend(docs)
            except Exception as e:
                continue
        
        status_container.empty()
        
        if not all_docs:
            st.error("문서를 불러오는데 실패했습니다.")
            return None
        
        # 4. 문서 분할
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_docs = splitter.split_documents(all_docs)
        
        # 5. 벡터 저장소 생성 및 저장
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        vector_store.save_local(vector_store_path)
        
        return vector_store.as_retriever(search_kwargs={"k": 4})
        
    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {str(e)}")
        return None

# Main interface
retriever = load_cloudflare_docs()

if retriever is None:
    st.stop()

# 예시 질문 섹션
st.divider()
st.markdown("### 예시 질문")
example_questions = [
    "llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?",
    "Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?",
    "벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?"
]

# 버튼 컬럼 생성
cols = st.columns(len(example_questions))
clicked_question = None

# 각 예시 질문을 버튼으로 만들기
for idx, question in enumerate(example_questions):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.write(f"* {question}")
    with col2:
        if st.button(f"예시 {idx + 1}", key=f"example_{idx}"):
            clicked_question = question

st.divider()
st.markdown("### Cloudflare AI 제품에 대해 질문해보세요")
# 사용자 입력 필드 (예시 질문이 선택되면 그 질문으로 채워짐)
query = st.text_input(
    "",
    value=clicked_question if clicked_question else "",
    key="user_question"
)

if query:
    with st.spinner("문서를 검색하고 답변을 생성하고 있습니다..."):
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\\$"))
