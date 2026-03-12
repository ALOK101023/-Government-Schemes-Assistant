import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Setup ---
st.set_page_config(page_title="Gov Schemes AI", page_icon="🏛")
st.title("🏛 Government Schemes Assistant")
st.info("I can help you with details on PM Kisan, Ayushman Bharat, and more in English or Hindi.")

# --- RAG Logic (Cached for Performance) ---
@st.cache_resource
def load_rag_system():
    # Load your schemes.txt file
    with open("schemes.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Split and Embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # LLM & Prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        template="Answer ONLY from context. If unknown, say so. \nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )

    # Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )

# --- App Execution ---
# Check for API Key in Streamlit Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add your OPENAI_API_KEY to the Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me anything about government schemes"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chain = load_rag_system()
        response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
