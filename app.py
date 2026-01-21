import streamlit as st
import os, uuid, tempfile, shutil
from typing import List, TypedDict, Annotated
from PIL import Image
import pytesseract

# =======================
# PAGE CONFIG
# =======================
st.set_page_config("OmniAgent Ultra", "🧠", layout="wide")

if os.name == "posix":
    pytesseract.pytesseract_cmd = shutil.which("tesseract")

# =======================
# IMPORTS
# =======================
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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    DDG_AVAILABLE = True
except:
    DDG_AVAILABLE = False

# =======================
# FILE PROCESSOR
# =======================
class FileProcessor:
    @staticmethod
    def extract_text(file):
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
            f.write(file.getbuffer())
            path = f.name

        try:
            if ext == "pdf":
                docs = PyPDFLoader(path).load()
                return "\n".join(d.page_content for d in docs)
            if ext in ["png", "jpg", "jpeg"]:
                return pytesseract.image_to_string(Image.open(path))
            if ext in ["txt", "md"]:
                return open(path, encoding="utf-8").read()
        finally:
            os.remove(path)
        return None

# =======================
# RAG ENGINE
# =======================
class RAGEngine:
    def __init__(self, keys):
        self.keys = keys
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
        if not files:
            return None

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
            results = retriever.invoke(query)
            if not results:
                return {"answer": "NO_DATA_FOUND", "source": "Documents"}

            text = "\n\n".join(r.page_content for r in results)
            sources = list({r.metadata["source"] for r in results})
            return {"answer": text, "source": f"Documents: {', '.join(sources)}"}

        return knowledge_base

    def create_agent(self, model_choice, kb_tool):
        references = []

        @tool
        def route_query(query: str):
            if any(x in query.lower() for x in ["calculate", "+", "-", "*", "/", "%"]):
                return "MATH"
            if any(x in query.lower() for x in ["who", "latest", "news", "when"]):
                return "WEB"
            return "DOCUMENT"

        @tool
        def calculator(expr: str):
            try:
                references.append("Calculator")
                return str(eval(expr, {"__builtins__": {}}))
            except:
                return "Invalid expression"

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3)
        )
        arxiv = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper()
        )

        def wrap_tool(tool_obj, name):
            @tool
            def wrapped(query: str):
                references.append(name)
                return tool_obj.run(query)
            return wrapped

        tools = [
            route_query,
            calculator,
            wrap_tool(wiki, "Wikipedia"),
            wrap_tool(arxiv, "Arxiv")
        ]

        if DDG_AVAILABLE:
            tools.append(wrap_tool(DuckDuckGoSearchRun(), "DuckDuckGo"))

        if kb_tool:
            def kb_wrapper(query: str):
                result = kb_tool.run(query)
                if isinstance(result, dict):
                    references.append(result["source"])
                    return result["answer"]
                return result
            tools.append(tool(kb_wrapper))

        llm = self.get_llm(model_choice).bind_tools(tools)

        class State(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]

        def agent(state: State):
            response = llm.invoke(state["messages"])
            if references:
                response.content += "\n\n---\n**References:**\n" + "\n".join(
                    f"- {r}" for r in dict.fromkeys(references)
                )
            return {"messages": [response]}

        graph = StateGraph(State)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        return graph.compile(checkpointer=MemorySaver())

# =======================
# STREAMLIT UI (CHATGPT STYLE)
# =======================
def main():
    keys = {
        "GROQ_API_KEY": st.secrets["GROQ_API_KEY"],
        "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
        "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"]
    }

    engine = RAGEngine(keys)

    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.active_chat = str(uuid.uuid4())
        st.session_state.chats[st.session_state.active_chat] = []

    with st.sidebar:
        st.title("🧠 OmniAgent Ultra")

        model = st.selectbox(
            "Choose Brain",
            ["Llama 3.1 8B (Groq)", "GPT-4o (OpenAI)", "Gemini 1.5 Pro (Google)"]
        )

        uploaded = st.file_uploader(
            "Knowledge Base",
            accept_multiple_files=True
        )

        kb_tool = engine.ingest(uploaded)

        st.markdown("### 💬 Chats")
        for cid in st.session_state.chats:
            if st.button(cid[:8], key=cid):
                st.session_state.active_chat = cid

        if st.button("➕ New Chat"):
            cid = str(uuid.uuid4())
            st.session_state.chats[cid] = []
            st.session_state.active_chat = cid
            st.rerun()

    messages = st.session_state.chats[st.session_state.active_chat]

    for m in messages:
        st.chat_message(m["role"]).markdown(m["content"])

    if prompt := st.chat_input("Ask anything…"):
        messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        agent = engine.create_agent(model, kb_tool)
        config = {"configurable": {"thread_id": st.session_state.active_chat}}

        with st.chat_message("assistant"):
            result = agent.invoke(
                {"messages": [SystemMessage(content="You are a factual agent."), HumanMessage(content=prompt)]},
                config=config
            )
            reply = result["messages"][-1].content
            st.markdown(reply)

        messages.append({"role": "assistant", "content": reply})
        st.session_state.chats[st.session_state.active_chat] = messages

if __name__ == "__main__":
    main()
