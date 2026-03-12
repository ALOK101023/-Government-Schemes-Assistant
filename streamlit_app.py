import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Setup ---
st.set_page_config(page_title="Jan Sahayak AI", page_icon="🏛")

# --- Sidebar: Scheme Selection ---
with st.sidebar:
    st.title("📂 Quick Selection")
    st.write("Pick a scheme to get instant details:")
    
    # List of schemes from your dataset
    schemes_list = [
        "Select a scheme...",
        "PM Kisan Samman Nidhi",
        "Ayushman Bharat (PM-JAY)",
        "PM Awas Yojana (PMAY)",
        "PM Ujjwala Yojana",
        "Sukanya Samriddhi Yojana",
        "Atal Pension Yojana",
        "PM MUDRA Yojana",
        "PM Vishwakarma",
        "PM YASASVI Scholarship",
        "Ladli Yojana (Delhi)",
        "Free Bus for Women (Delhi)"
    ]
    
    selected_scheme = st.selectbox("Current Schemes:", schemes_list)
    
    st.divider()
    st.info("Tip: You can also type questions in Hindi in the chat below.")

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
        template="Answer based on context. Language must match user query.\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )

# --- App Logic ---
st.title("🏛 Government Schemes Assistant")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Missing API Key in Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Sidebar Selection
# If the user picks something other than the placeholder, treat it as a message
if selected_scheme != "Select a scheme...":
    user_query = f"Tell me about {selected_scheme}"
    # Check if this was the last message to prevent infinite loops
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            chain = initialize_rag()
            response = chain.invoke(user_query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Handle Manual Chat Input
if prompt := st.chat_input("Or type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chain = initialize_rag()
        response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
