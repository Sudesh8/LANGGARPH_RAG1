import streamlit as st
import uuid  # unique id
import datetime
from rag1 import chatbot, ask_agentic_ai
from langchain_core.messages import HumanMessage, AIMessage


# creating ids like token
def generate_thread_id():
    return str(uuid.uuid4())


# initialize Chats if missing
if "chats" not in st.session_state:
    st.session_state["chats"] = {}

# initialize Counter if missing
if "session_counter" not in st.session_state:
    st.session_state["session_counter"] = 1

# initialize Current Thread if missing
if "current_thread_id" not in st.session_state:
    new_id = generate_thread_id()
    st.session_state["chats"][new_id] = {"title": "Chat Session 1", "history": []}
    st.session_state["current_thread_id"] = new_id


# helper to switch chats
def switch_chat(thread_id):
    st.session_state["current_thread_id"] = thread_id


# helper to add new chat
def create_new_chat():
    new_id = generate_thread_id()

    # increase the counter for next name
    st.session_state["session_counter"] += 1
    count = st.session_state["session_counter"]

    st.session_state["chats"][new_id] = {
        "title": f"Chat Session {count}",
        "history": [],
    }
    st.session_state["current_thread_id"] = new_id


# actual disply things
st.sidebar.title("Agentic 🇦🇮 Chatbot")

# New Chat
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("History")

# list all threads in reverse order newest first
thread_ids = list(st.session_state["chats"].keys())[::-1]

for tid in thread_ids:
    chat_data = st.session_state["chats"][tid]
    # highlight the active button
    is_active = tid == st.session_state["current_thread_id"]

    if st.sidebar.button(
        chat_data["title"], key=tid, type="primary" if is_active else "secondary"
    ):
        switch_chat(tid)
        st.rerun()

st.sidebar.markdown(f"**Session ID:** `{st.session_state['current_thread_id']}`")


# Acual UI
st.title("TheCophil Agentic AI RAG System")
st.caption("Strictly answers based on the provided PDF.")

# current history for display
current_id = st.session_state["current_thread_id"]

# if current_id somehow isnot in chats reset
if current_id not in st.session_state["chats"]:
    st.session_state["chats"][current_id] = {"title": "New Chat", "history": []}

current_history = st.session_state["chats"][current_id]["history"]

# display Chat History if available
for msg in current_history:
    role = msg["role"]
    content = msg["content"]

    with st.chat_message(role):
        st.write(content)
        # if there context stored in this message history item show it
        if "context" in msg and msg["context"]:
            with st.expander("▶️ View Retrieved Context"):
                for i, ctx in enumerate(msg["context"]):
                    st.markdown(f"**Chunk {i+1}:** {ctx[:300]}...")
                st.caption(f"Confidence Score: {msg.get('confidence', 0)}")

# User querry innput
user_input = st.chat_input("Ask about Agentic AI...")

if user_input:
    # Add user message to display
    st.session_state["chats"][current_id]["history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # spinner thinking
    with st.spinner("Analyzing PDF..."):
        # my bcakeng code triggers
        answer, context, confidence, tool_used = ask_agentic_ai(user_input, current_id)

    # add assistant message to display so we can ssee all chat strat to end
    st.session_state["chats"][current_id]["history"].append(
        {
            "role": "assistant",
            "content": answer,
            "context": context if tool_used else None,
            "confidence": confidence,
        }
    )

    # for new mesage
    st.rerun()
