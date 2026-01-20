import streamlit as st
import os
import uuid
import tempfile
from PIL import Image
import pytesseract
from typing import List, TypedDict, Annotated

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="OmniAgent Ultra", page_icon="🧠", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .stChatInput { position: fixed; bottom: 3rem; }
    .status-box {
        padding: 10px; border-radius: 5px; margin-bottom: 10px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CORE IMPORTS ---
try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
    from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
    from langchain_core.documents import Document
    from langgraph.graph.message import add_messages
except ImportError as e:
    st.error(f"🚨 Libraries missing! Error: {e}")
    st.stop()

# --- 3. UNIVERSAL FILE PROCESSOR (OCR & PDF) ---
class FileProcessor:
    """Handles PDFs, Text files, and Images (OCR)"""
    
    @staticmethod
    def extract_text_from_file(uploaded_file):
        file_ext = uploaded_file.name.split(".")[-1].lower()
        text_content = ""
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        try:
            # STRATEGY 1: PDF
            if file_ext == "pdf":
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                text_content = "\n".join([d.page_content for d in docs])
            
            # STRATEGY 2: IMAGES (OCR)
            elif file_ext in ["png", "jpg", "jpeg", "bmp"]:
                image = Image.open(temp_path)
                text_content = pytesseract.image_to_string(image)
                text_content = f"[IMAGE CONTENT FROM {uploaded_file.name}]:\n{text_content}"
            
            # STRATEGY 3: TEXT/CODE
            elif file_ext in ["txt", "md", "py", "csv"]:
                with open(temp_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            
            else:
                return None  # Unsupported file
                
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        return text_content

# --- 4. RAG ENGINE (BRAIN) ---
class RAGEngine:
    def __init__(self, api_keys):
        self.groq_key = api_keys.get("GROQ_API_KEY")
        self.openai_key = api_keys.get("OPENAI_API_KEY")
        self.google_key = api_keys.get("GOOGLE_API_KEY")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def get_llm(self, model_choice):
        """Factory for Model Selection"""
        if "Llama" in model_choice:
            return ChatGroq(model="llama-3.1-8b-instant", api_key=self.groq_key, temperature=0)
        elif "GPT" in model_choice:
            return ChatOpenAI(model="gpt-4o", api_key=self.openai_key, temperature=0)
        elif "Gemini" in model_choice:
            return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=self.google_key, temperature=0)
        return ChatGroq(model="llama-3.1-70b-versatile", api_key=self.groq_key)

    def ingest_files(self, files_list):
        """Processes MULTIPLE files (PDFs/Images) into ONE vector store"""
        all_text_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        total_files = 0
        
        for file in files_list:
            raw_text = FileProcessor.extract_text_from_file(file)
            if raw_text and len(raw_text) > 10: # Ignore empty/unreadable files
                doc = Document(page_content=raw_text, metadata={"source": file.name})
                chunks = splitter.split_documents([doc])
                all_text_chunks.extend(chunks)
                total_files += 1
        
        if not all_text_chunks:
            return None, 0

        # Create Vector Store
        vectorstore = FAISS.from_documents(all_text_chunks, self.embeddings)
        retriever = vectorstore.as_retriever()
        
        # Define the Retrieval Tool
        @tool
        def knowledge_base(query: str):
            """Search the uploaded files (PDFs, Images, Text) for answers."""
            results = retriever.invoke(query)
            if not results:
                return "NO_DATA_FOUND"
            # Return context with Source Filename attached
            return "\n\n".join([f"[Source: {d.metadata['source']}] {d.page_content}" for d in results])
            
        return knowledge_base, total_files

    def create_agent(self, model_name, file_tool=None):
        # 1. Define Web Tools
        web_tools = [
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
            ArxivQueryRun(api_wrapper=ArxivAPIWrapper()),
            DuckDuckGoSearchRun()
        ]
        
        # 2. Combine Tools (File Tool Priority)
        all_tools = web_tools
        if file_tool:
            all_tools = [file_tool] + web_tools

        llm = self.get_llm(model_name)
        llm_with_tools = llm.bind_tools(all_tools)
        
        # 3. LangGraph Setup
        class State(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            
        def reasoner(state: State):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
            
        graph = StateGraph(State)
        graph.add_node("agent", reasoner)
        graph.add_node("tools", ToolNode(all_tools))
        
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        
        return graph.compile(checkpointer=MemorySaver())

# --- 5. MAIN APP UI WITH MULTI-CHAT ---
def main():
    # Load Secrets
    try:
        api_keys = {
            "GROQ_API_KEY": st.secrets["GROQ_API_KEY"],
            "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
            "GOOGLE_API_KEY": st.secrets["GOOGLE_API_KEY"]
        }
        engine = RAGEngine(api_keys)
    except Exception:
        st.error("🚨 Secrets not found! If you are on Streamlit Cloud, go to 'Settings' > 'Secrets' and paste your keys.")
        st.stop()

    # --- INITIALIZE SESSION STATE FOR MULTI-CHAT ---
    if "chat_sessions" not in st.session_state:
        # Structure: { "thread_id": [messages_list], ... }
        st.session_state.chat_sessions = {}
    
    if "active_thread_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.active_thread_id = new_id
        st.session_state.chat_sessions[new_id] = []

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🧠 OmniAgent Ultra")
        st.caption("Multi-Modal • Anti-Hallucination")
        
        model_choice = st.selectbox("Choose Brain", 
            ["Llama 3.1 8B (Groq)", "GPT-4o (OpenAI)", "Gemini 1.5 Pro (Google)"])
        
        # --- SESSION MANAGEMENT ---
        st.divider()
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("**💬 Chat History**")
        with col2:
            # Button to create NEW chat
            if st.button("➕", help="New Chat"):
                new_id = str(uuid.uuid4())
                st.session_state.active_thread_id = new_id
                st.session_state.chat_sessions[new_id] = []
                st.rerun()

        # Display list of past chats
        session_ids = list(st.session_state.chat_sessions.keys())
        # Reverse to show newest at top
        for s_id in reversed(session_ids):
            # Label the chat based on the first user message (or "New Chat")
            msgs = st.session_state.chat_sessions[s_id]
            first_msg = next((m["content"] for m in msgs if m["role"] == "user"), "New Chat")
            label = (first_msg[:25] + '...') if len(first_msg) > 25 else first_msg
            
            # Highlight current chat
            if s_id == st.session_state.active_thread_id:
                st.markdown(f"**👉 {label}**")
            else:
                if st.button(label, key=s_id):
                    st.session_state.active_thread_id = s_id
                    st.rerun()

        st.divider()
        st.markdown("**📂 Knowledge Base**")
        uploaded_files = st.file_uploader("Upload Docs (PDF, PNG, TXT)", 
                                        type=["pdf", "txt", "png", "jpg", "md"], 
                                        accept_multiple_files=True)
        
        knowledge_tool = None
        if uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                knowledge_tool, count = engine.ingest_files(uploaded_files)
                if knowledge_tool:
                    st.success(f"✅ Ingested {count} documents!")
                else:
                    st.warning("⚠️ No readable text found.")

    # --- CHAT LOGIC ---
    active_id = st.session_state.active_thread_id
    current_messages = st.session_state.chat_sessions[active_id]

    # Display history for the ACTIVE thread only
    for msg in current_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Ask about your documents..."):
        # 1. Append User Message to State
        current_messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # 2. Strict Protocol
        system_instruction = """You are a strict Data Analyst Agent.
        PROTOCOL:
        1. IF FILES ARE UPLOADED: You MUST use the 'knowledge_base' tool first.
        2. IF ANSWER FOUND IN FILES: Reply with the answer and append "**Source:** [Filename]".
        3. IF ANSWER NOT FOUND IN FILES: 
           - DO NOT use the Web immediately.
           - Reply EXACTLY: "I could not find this information in the uploaded documents. Would you like me to search the web?"
        4. IF NO FILES UPLOADED OR PERMISSION GIVEN:
           - Use DuckDuckGo, Wikipedia, or Arxiv.
        """
        
        agent = engine.create_agent(model_choice, knowledge_tool)
        # Use the ACTIVE thread_id for LangGraph memory
        config = {"configurable": {"thread_id": active_id}}
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            inputs = [SystemMessage(content=system_instruction)] + current_messages
            
            events = agent.stream({"messages": inputs}, config=config, stream_mode="values")
            
            for event in events:
                if "messages" in event:
                    msg = event["messages"][-1]
                    if msg.type == "ai":
                        full_response = msg.content
                        message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # 3. Append AI Response to State
            current_messages.append({"role": "assistant", "content": full_response})
            # Force update the session state dictionary
            st.session_state.chat_sessions[active_id] = current_messages
            st.rerun() # Rerun to update sidebar labels immediately

if __name__ == "__main__":
    main()
