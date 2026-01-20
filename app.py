# ... (Keep all imports and classes like FileProcessor and RAGEngine exactly the same) ...

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
        st.error("🚨 Secrets not found! Please check your .streamlit/secrets.toml")
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
        st.title("🧠 NAVAM-LLM")
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
