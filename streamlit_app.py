import streamlit as st
import os
import re
import feedparser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Setup ---
st.set_page_config(page_title="Jan Sahayak AI", page_icon="🏛", layout="wide")

# --- Function to Extract Scheme Names ---
def get_all_scheme_names(file_path):
    if not os.path.exists(file_path):
        return ["PM Kisan", "Ayushman Bharat"]
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Scans for "Scheme: Name" or "Yojana: Name"
    found_names = re.findall(r"(?:Scheme|Yojana|Name):\s*(.*)", content)
    if not found_names:
        return ["PM Kisan", "Ayushman Bharat", "PM Vishwakarma", "Sukanya Samriddhi"]
    return list(dict.fromkeys(found_names))[:12]

# --- RAG Logic ---
@st.cache_resource
def initialize_rag():
    with open("schemes.txt", "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = PromptTemplate(
        template="""You are Jan Sahayak AI, a helpful assistant for Indian Gov Schemes.
        Match the user's language (Hindi/English). Use context first, then general knowledge.
        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )

# --- Sidebar: News & Disclaimer ---
with st.sidebar:
    st.title("🏛 Info Panel")
    st.markdown("""
    ### 🌐 Bilingual Support
    Ask in **English** or **हिंदी**.
    """)
    st.divider()
    
    st.markdown("### 🔔 Live Gov Updates (PIB)")
    try:
        # Fetching latest releases from PIB English Feed
        feed = feedparser.parse("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1")
        for entry in feed.entries[:5]:
            st.markdown(f"**• [{entry.title}]({entry.link})**")
            st.caption(f"📅 {entry.published[:16]}")
    except:
        st.write("Live feed temporarily unavailable.")
    
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Main App Logic ---
st.title("🏛 Jan Sahayak AI")
st.caption("🚀 Supports English & Hindi | 24/7 Government Scheme Guide")

# API Key Check
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Missing API Key in Streamlit Secrets!")
    st.stop()

# --- Suggestions Section ---
st.write("### Quick Access:")
all_schemes = get_all_scheme_names("schemes.txt")
selected_pill = st.pills("Click to learn more:", all_schemes, selection_mode="single", label_visibility="collapsed")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Combined Input Logic ---
user_query = None
if selected_pill:
    user_query = f"Tell me about {selected_pill}"

if chat_input := st.chat_input("Ask a question..."):
    user_query = chat_input

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                chain = initialize_rag()
                response = chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
