import streamlit as st
import os
import uuid
import tempfile
import shutil
from typing import List, TypedDict, Annotated
from PIL import Image
import pytesseract

# =========================
# 1. APP CONFIG
# =========================
st.set_page_config(
    page_title="OmniAgent Ultra",
    page_icon="🧠",
    layout="wide"
)

# Tesseract (Streamlit Cloud Linux)
if os.name == "posix":
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# =========================
# 2. IMPORTS
# =========================
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Safe DDG import
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    DDG_AVAILABLE = True
except Exception:
    DDG_AVAILABLE = False

# =========================
# 3. FILE PROCESSOR
# =========================
class FileProcessor:
    @staticmethod
    def extract_text(uploaded_file):
        ext = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
            f.write(uploaded_file.getbuffer())
            path = f.name

        try:
            if ext == "pdf":
                docs = PyPDFLoader(path).load()
                return "\n".join(d.page_content for d in docs)

            if ext in ["png", "jpg", "jpeg"]:
                text = pytesseract.image_to_string(Image.open(path))
                return f"[IMAGE OCR]: {text}"

            if ext in ["txt", "md", "csv"]:
                return open(path, encoding="utf-8").read()
        finally:
            os.remove(path)

        return None

# =========================
# 4. RAG ENGINE
# =========================
class RAGEngine:
    def __init__(self, keys):
        self.keys = keys
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def get_llm(self, choice):
        if "Llama" in choice:
            return ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=self.keys["GROQ_API_KEY"],
                temperature=0
            )
        if "GPT" in choice:
            return ChatOpenAI(
                model="gpt-4o",
                api_key=self.keys["OPENAI_API_KEY"],
                temperature=0
            )
        if "Gemini" in choice:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=self.keys["GOOGLE_API_KEY"],
                temperature=0
            )

    def ingest(self, files):
        splitter = RecursiveCharacterTextSplitter(1000, 200)
        docs = []

        for f in files:
            text = FileProcessor.extract_text(f)
            if text:
                docs.extend(
                    splitter.split_documents(
                        [Document(page_content=text, metadata={"source": f.name})]
                    )
                )

        if not docs:
            return None

        store = FAISS.from_documents(docs, self.embeddings)
        retriever = store.as_retriever(k=4)

        @tool
        def knowledge_base(query: str):
            """Search uploaded documents"""
            res = retriever.invoke(query)
            if not res:
                return "NO_DATA_FOUND"
            return "\n\n".join(
                f"[Source: {d.metadata['source']}] {d.page_content}"
                for d in res
            )

        return knowledge_base

    def create_agent(self, model_choice, kb_tool=None):
        # ---- ROUTER TOOL ----
        @tool
        def route_query(query: str) -> str:
            """Route query: DOCUMENT | WEB | MATH"""
            q = query.lower()
            if any(x in q for x in ["calculate", "+", "-", "*", "/", "%"]):
                return "MATH"
            if any(x in q for x in ["who", "when", "latest", "news", "today"]):
                return "WEB"
            return "DOCUMENT"

        # ---- CALCULATOR ----
        @tool
        def calculator(expr: str):
            """Evaluate math expressions"""
            try:
                return str(eval(expr, {"__builtins__": {}}))
            except:
                return "Invalid expression"

        tools = [route_query, calculator]

        if kb_tool:
            tools.append(kb_tool)

        tools.append(
            WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1200)
            )
        )
        tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper()))

        if DDG_AVAILABLE:
            tools.append(DuckDuckGoSearchRun())

        llm = self.get_llm(model_choice)
        llm = llm.bind_tools(tools)

        class State(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]

        def agent(state: State):
            return {"messages": [llm.invoke(state["messages"])]}

        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode(tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        return graph.compile(checkpointer=MemorySaver())

# =========================
# 5. STREAMLIT UI
# =========================
def main():
    keys = {
        "GROQ_API_KEY": st.secrets["GROQ_API_KEY"],
        "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
        "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"]
    }

    engine = RAGEngine(keys)

    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
        sid = str(uuid.uuid4())
        st.session_state.sessions[sid] = []
        st.session_state.active = sid

    with st.sidebar:
        st.title("🧠 OmniAgent Ultra")

        model = st.selectbox(
            "Choose Brain",
            ["Llama 3.1 8B (Groq)", "GPT-4o (OpenAI)", "Gemini 1.5 Pro (Google)"]
        )

        uploaded = st.file_uploader(
            "Upload Knowledge Files",
            accept_multiple_files=True
        )

        kb_tool = engine.ingest(uploaded) if uploaded else None

        if st.button("➕ New Chat"):
            sid = str(uuid.uuid4())
            st.session_state.sessions[sid] = []
            st.session_state.active = sid
            st.rerun()

    messages = st.session_state.sessions[st.session_state.active]

    for m in messages:
        st.chat_message(m["role"]).write(m["content"])

    if user := st.chat_input("Ask anything…"):
        messages.append({"role": "user", "content": user})
        st.chat_message("user").write(user)

        agent = engine.create_agent(model, kb_tool)
        config = {"configurable": {"thread_id": st.session_state.active}}

        system = SystemMessage(content="""
You are a strict research agent.

Rules:
1. Always call route_query first
2. DOCUMENT → use knowledge_base
3. If NO_DATA_FOUND → ask permission for web
4. WEB → use DuckDuckGo or Wikipedia
5. MATH → use calculator only
6. Never hallucinate
""")

        with st.chat_message("assistant"):
            placeholder = st.empty()
            out = ""

            for event in agent.stream(
                {"messages": [system] + [HumanMessage(content=user)]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event:
                    out = event["messages"][-1].content
                    placeholder.markdown(out)

        messages.append({"role": "assistant", "content": out})
        st.session_state.sessions[st.session_state.active] = messages

if __name__ == "__main__":
    main()
