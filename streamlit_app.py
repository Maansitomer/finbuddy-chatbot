import asyncio
from dotenv import load_dotenv
import os

# Load environment variables from config/config.env
load_dotenv("config/config.env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ GOOGLE_API_KEY not found in config/config.env!")

# Fix "no running event loop" issue for gRPC
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.vector_store import load_vector_store
from ddgs import DDGS


# ---- CONFIG ----
STORE_PATH = "C:/Users/Admin/Downloads/NeoStats AI Engineer Internship Use Case/AI_UseCase/vectorstore/faiss_store"

st.set_page_config(page_title="FinBuddy", page_icon="💰", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# ---- STYLING ----
st.markdown("""
    <style>
    html, body {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }

    h1, h2, h3, h4 {
        color: #004d40;
    }

    .block-container {
        padding-top: 1rem;
    }

    .chat-container {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        margin-top: 20px;
        max-height: 70vh;
        overflow-y: auto;
    }

    .chat-bubble {
        padding: 14px 20px;
        border-radius: 20px;
        margin-bottom: 10px;
        animation: fadeIn 0.5s ease-in-out;
        line-height: 1.5;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    .user {
        background: linear-gradient(to right, #00bcd4, #009688);
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 5px;
        margin-left: auto;
        text-align: right;
    }

    .bot {
        background: #f1f8e9;
        color: #333;
        align-self: flex-start;
        border-bottom-left-radius: 5px;
        margin-right: auto;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .stButton>button {
        background: linear-gradient(135deg, #00bfa5, #00796b);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 15px;
        box-shadow: 0 4px 12px rgba(0, 121, 107, 0.3);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #004d40, #00695c);
        box-shadow: 0 6px 20px rgba(0, 77, 64, 0.5);
        transform: scale(1.03);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(to right, #4caf50, #2e7d32);
        color: white;
        font-weight: 700;
        padding: 10px 25px;
        font-size: 16px;
        border-radius: 30px;
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(to right, #66bb6a, #388e3c);
        box-shadow: 0 8px 25px rgba(56, 142, 60, 0.6);
        transform: scale(1.05);
    }

    .right-column {
        background: #ffffff;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }

    th, td {
        border: 1px solid #ddd;
        padding: 10px 12px;
    }

    th {
        background-color: #e0f2f1;
        color: #00695c;
        text-align: left;
    }

    td {
        background-color: #fafafa;
    }

    .stNumberInput input {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
    }

    .stAlert > div {
        border-radius: 15px;
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }

    .stChatInput input {
        padding: 12px 20px;
        border-radius: 25px;
        border: 1px solid #ccc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---- LOAD VECTOR STORE ----
try:
    store = load_vector_store(STORE_PATH)
    retriever = store.as_retriever() if store else None
except Exception as e:
    st.warning(f"⚠️ Could not load vector store: {e}")
    store = None
    retriever = None

# ---- GEMINI CHAT MODEL ----
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)
qa_chain = None
if retriever:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ---- LIVE WEB SEARCH ----
def live_web_search(query, max_results=3):
    """Fetch live search results from DuckDuckGo."""
    with DDGS() as ddg:
        results = list(ddg.text(query, max_results=max_results))
        if results:
            combined = "\n".join([f"{r['body']} (Source: {r['href']})" for r in results])
            return combined
        return "No live data found."

def rewrite_with_llm(query, raw_info):
    """Rewrite search results into a chatbot-friendly answer using Gemini."""
    prompt = f"""
    You are FinBuddy, a friendly BFSI loan assistant.
    The user asked: "{query}"
    I found the following web search results:
    {raw_info}

    Please provide a clear, concise, and helpful answer to the user, 
    using a natural conversational tone. 
    Include the source link if relevant.
    """
    response = llm.invoke(prompt)
    return response.content.strip()

# ---------------- UI ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("💬 FinBuddy – Your BFSI Loan Assistant")
    st.write("💡 Ask about loans, interest rates, EMIs, and more.")

    for chat in st.session_state.history:
        st.markdown(f"""
            <div class='chat-bubble {'user' if chat['role'] == 'user' else 'bot'}'>
                {chat['content']}
            </div>
        """, unsafe_allow_html=True)

    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})

        with st.spinner("💭 Thinking..."):
            answer = ""
            if qa_chain:
                result = qa_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_memory})
                answer = result.get("answer", "").strip()
                sources = result.get("source_documents", [])
            else:
                sources = []

            no_info_phrases = [
                "i don't know", "cannot answer", "doesn't specify", "not specified", "no information",
                "no details provided", "i'm sorry, but i don't have", "sorry, but i don't have",
                "sorry, i don't have", "no data available", "cannot find"
            ]

            if (not sources or not answer or any(p in answer.lower() for p in no_info_phrases)):
                st.warning("No relevant info found in internal DB. Searching live web...")
                raw_search_result = live_web_search(user_input)
                answer = rewrite_with_llm(user_input, raw_search_result)

            st.session_state.chat_memory.append((user_input, answer))

        st.session_state.history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.history.clear()
        st.session_state.chat_memory.clear()
        st.rerun()

with col2:
    st.markdown('<div class="right-column">', unsafe_allow_html=True)

    st.subheader("📈 Interest Rate Trends")
    st.table(pd.DataFrame({
        "Loan Type": ["Home Loan", "Personal Loan", "Car Loan", "Education Loan"],
        "Interest Rate (%)": [8.5, 12.5, 9.0, 10.5]
    }))

    st.subheader("🧲 EMI Calculator")
    loan_amt = st.number_input("Loan Amount (₹)", value=500000, step=10000)
    rate = st.number_input("Annual Interest Rate (%)", value=8.5, step=0.1)
    tenure = st.number_input("Tenure (Years)", value=5, step=1)

    if st.button("Calculate EMI", type="primary"):
        monthly_rate = rate / 100 / 12
        emi = loan_amt * monthly_rate * (1 + monthly_rate) ** (tenure * 12) / ((1 + monthly_rate) ** (tenure * 12) - 1)
        st.success(f"Your EMI: ₹{emi:,.2f} per month")

    st.subheader("🏦 Loan Highlights")
    st.markdown("""
    - **Home Loan:** Low-interest rates starting from 8.5%  
    - **Personal Loan:** Quick disbursal within 24 hours  
    - **Car Loan:** Flexible tenure up to 7 years  
    - **Education Loan:** Special student discounts  
    """)

    st.markdown('</div>', unsafe_allow_html=True)