import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Jan Sahayak AI", page_icon="🏛")

# --- Initialize RAG (Same as before) ---
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
        template="Answer based on context. Answer in user's language.\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    return (
        RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )

st.title("🏛 Government Schemes Assistant")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key in Secrets!")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- NEW: Suggestions Feature ---
st.markdown("### Quick Suggestions")
suggestions = [
    "PM Kisan details", 
    "Ayushman Bharat eligibility", 
    "PM Vishwakarma benefits", 
    "Sukanya Samriddhi Yojana"
]

# Use st.pills to show options (available in latest Streamlit)
selected_suggestion = st.pills("Select an option or type below:", suggestions, selection_mode="single", label_visibility="collapsed")

# Logic to process input (either from pills or from chat_input)
user_query = None

if selected_suggestion:
    user_query = selected_suggestion
    # Clear selection after use (optional, helps avoid repeating)
    # st.rerun() if you want to clear the pill immediately

if prompt := st.chat_input("Type your question here..."):
    user_query = prompt

# --- Handle the Response ---
if user_query:
    # Check if we should ignore if it's a repeat of the last pill click
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            chain = initialize_rag()
            response = chain.invoke(user_query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
