import streamlit as st
from openai import OpenAI
import time
import json
import os

# 페이지 설정
st.set_page_config(
    page_title="Research Assistant AI",
    page_icon="🔍"
)

# 사이드바 설정
with st.sidebar:
    st.markdown("[🔗 Git Repo Link](https://github.com/geunsu-son/fullstack-gpt)")
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")

# 메인 타이틀
st.title("Research Assistant AI")
st.write("""연구하고 싶은 주제나 키워드를 입력하세요. AI가 관련 정보를 수집하고 분석하여 답변해드립니다. 이 AI 어시스턴트는 다음과 같은 기능을 제공합니다.  
    - 주제에 대한 전문적인 분석 수행
    - 수집한 정보를 종합하여 한국어로 답변
    - 연구 결과를 파일로 저장
    - 저장된 연구 결과를 기반으로 추가 질문 응답
    """)

# OpenAI 클라이언트 초기화 함수 (캐싱)
@st.cache_resource
def get_openai_client(api_key: str):
    return OpenAI(api_key=api_key)

# Assistant 초기화 함수 (캐싱)
@st.cache_resource
def get_or_create_assistant(_client):
    # 기존 Assistant 목록 확인
    assistants = _client.beta.assistants.list(
        order="desc",
        limit=1
    )
    
    # 기존 Research Assistant가 있는지 확인
    for assistant in assistants.data:
        if assistant.name == "Research Assistant":
            return assistant
    
    # 없으면 새로 생성
    return _client.beta.assistants.create(
        name="Research Assistant",
        instructions="""당신은 리서치를 수행하는 한국어 AI 어시스턴트입니다.
        사용자의 질문에 대해 다음과 같이 답변해주세요:

        새로운 연구 주제인 경우:
        1. 주어진 주제에 대해 전문적인 분석 수행
        2. 수집한 정보를 종합하여 한국어로 답변
        3. 모든 조사 내용을 txt 파일로 저장
        
        이전 연구 내용에 대한 추가 질문인 경우:
        1. 저장된 연구 내용을 참고하여 답변
        2. 필요한 경우 추가 분석 수행
        3. 새로운 통찰이나 관점 제시
        
        답변은 항상 한국어로 해주시고, 전문적이고 객관적인 톤을 유지해주세요.
        새로운 연구 결과를 저장할 때는 파일 이름과 저장경로를 출력해주세요.""",
        model="gpt-4-turbo-preview",
        tools=[{
            "type": "function",
            "function": {
                "name": "save_research_to_text",
                "description": "연구 결과를 텍스트 파일로 저장합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "저장할 연구 내용"
                        },
                        "filename": {
                            "type": "string",
                            "description": "저장할 파일 이름 (확장자 포함)"
                        }
                    },
                    "required": ["content", "filename"]
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "get_research_content",
                "description": "저장된 연구 내용을 불러옵니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "불러올 파일 이름 (확장자 포함)"
                        }
                    },
                    "required": ["filename"]
                }
            }
        }]
    )

# 파일 저장 함수
def save_research_to_text(content: str, filename: str):
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    # 연구 결과 저장 디렉토리 생성
    os.makedirs('research_results', exist_ok=True)
    filepath = os.path.join('research_results', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 세션 상태에 현재 연구 파일 저장
    st.session_state.current_research_file = filename
    return f"연구 결과가 {filepath} 파일에 저장되었습니다."

# 파일 읽기 함수
def get_research_content(filename: str):
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    filepath = os.path.join('research_results', filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"파일을 찾을 수 없습니다: {filepath}"

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_research_file" not in st.session_state:
    st.session_state.current_research_file = None

# OpenAI API 키 확인
if not openai_api_key:
    st.warning("OpenAI API 키를 입력해주세요.")
    st.stop()

# OpenAI 클라이언트와 Assistant 초기화
openai_client = get_openai_client(openai_api_key)
assistant = get_or_create_assistant(openai_client)

# 현재 연구 파일 표시
if st.session_state.current_research_file:
    st.sidebar.write(f"현재 연구 파일: {st.session_state.current_research_file}")

# 채팅 인터페이스
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("연구하고 싶은 주제나 질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant 응답 처리
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # 스레드 생성
            thread = openai_client.beta.threads.create()
            
            # 현재 연구 파일이 있다면 컨텍스트로 추가
            if st.session_state.current_research_file:
                research_content = get_research_content(st.session_state.current_research_file)
                context_message = f"""이전 연구 내용:
                {research_content}
                
                사용자 질문:
                {prompt}"""
                openai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=context_message
                )
            else:
                openai_client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=prompt
                )
            
            # 실행 시작
            run = openai_client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # 실행 완료 대기
            while run.status in ["queued", "in_progress"]:
                status_placeholder.text(f"상태: {run.status}")
                run = openai_client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                time.sleep(1)
            
            # 도구 호출 처리
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    if tool_call.function.name == "save_research_to_text":
                        output = save_research_to_text(args["content"], args["filename"])
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": output
                        })
                    elif tool_call.function.name == "get_research_content":
                        output = get_research_content(args["filename"])
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": output
                        })
                
                # 도구 출력 제출
                run = openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                
                # 실행 완료 대기
                while run.status in ["queued", "in_progress"]:
                    status_placeholder.text(f"상태: {run.status}")
                    run = openai_client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                    time.sleep(1)
            
            # 응답 메시지 가져오기
            messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
            
            # 최신 assistant 메시지 표시
            for message in messages.data:
                if message.role == "assistant":
                    full_response = message.content[0].text.value
                    break
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
        finally:
            status_placeholder.empty()

# 대화 기록 초기화 버튼
if st.sidebar.button("대화 기록 초기화"):
    st.session_state.messages = []
    st.session_state.current_research_file = None
    st.experimental_rerun() 