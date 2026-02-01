import os
import sqlite3
import json
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# predifined path
PDF_PATH = "Ebook-Agentic-AI.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"

print("*************** starting the process ************")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if os.path.exists(DB_FAISS_PATH):
    print("-****** loading existing vectorstore databse ******---")
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print("--- Vectorstore Loaded Successfully ---")
else:
    print("---### creating vectorstore ---###")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")

    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print("#########----pdf loaded from pypdf ------#####")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    print("#########----Vectorstore created and saved ------#####")

# --- 2. LLM Setup ---
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# as here we using tool calling so add tool designer
@tool
def agentic_ai_rag_tool(query: str) -> str:
    """
    Use this tool to retrieve information from the Agentic AI ebook.
    You MUST use this tool for any question regarding Agentic AI.
    """
    results = vectorstore.similarity_search_with_score(query, k=5)

    contexts = []
    similarities = []

    for doc, distance in results:
        # store all adat so we can cross chcek by page no nall
        page_num = doc.metadata.get("page", "Unknown")

        # If it's a number, add 1 to make it human-readable (Page 0 -> Page 1)
        if isinstance(page_num, int):
            page_num += 1

        # we need to add page num in context to varify
        formatted_text = f"**[Page {page_num}]**\n{doc.page_content}"
        contexts.append(formatted_text)
        # max sim =0.0
        # 1/(1+0)=1 == 100% correct

        score = float(1 / (1 + distance))
        similarities.append(score)

    avg_score = sum(similarities) / len(similarities) if similarities else 0.0
    confidence = float(round(avg_score, 2))

    return json.dumps(
        {
            "context": contexts,
            "confidence": confidence,
            "tool_called": True,
        }
    )


tools = [agentic_ai_rag_tool]
llm_with_tools = llm.bind_tools(tools)


# chat state to store our all values
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# our first node
def chat_node(state: ChatState):
    system_message = SystemMessage(
        content=(
            """
            You are an expert AI assistant strictly based on the 'Agentic AI' ebook.
            
            RULES:
            1. you have NO internal knowledge. You MUST use 'agentic_ai_rag_tool' to answer.
            2. if the user asks a question, call the tool.
            3. answer ONLY using the retrieved context. 
            4. if the context is empty or irrelevant, say "I cannot find this information in the ebook."
            """
        )
    )
    input_messages = [
        msg for msg in state["messages"] if not isinstance(msg, SystemMessage)
    ]
    messages = [system_message, *input_messages]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Graph creation
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

# short trem memory
checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)


# final output function where we store all the values
def ask_agentic_ai(question: str, thread_id: str):
    """
    Invokes the graph and extracts context from the ToolMessage history.
    """
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    ai_msg = result["messages"][-1]
    final_answer = ai_msg.content

    context = []
    confidence = 0.0
    tool_used = False

    # Look backwards for the ToolMessage
    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                context = data.get("context", [])
                confidence = data.get("confidence", 0.0)
                tool_used = True
                break
            except json.JSONDecodeError:
                continue

    return final_answer, context, confidence, tool_used


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
