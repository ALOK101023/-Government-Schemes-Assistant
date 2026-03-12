import streamlit as st
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Setup ---
st.set_page_config(page_title="Jan Sahayak AI", page_icon="🏛")

# --- Function to Automatically Extract Scheme Names ---
def get_all_scheme_names(file_path):
    if not os.path.exists(file_path):
        return ["PM Kisan", "Ayushman Bharat"] # Fallback if file is missing
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # This regex looks for lines like "1. PM Kisan" or "Scheme: PM Kisan"
    # Adjust this regex if your file uses a different pattern (e.g., "Name: ...")
    found_names = re.findall(r"(?:Scheme|Yojana|Name):\s*(.*)", content)
    
    # If no specific "Scheme:" tag exists, we'll just use these defaults 
    # OR you can manually list your top 20 here:
    if not found_names:
        return [
            "PM Kisan", "Ayushman Bharat", "PM Awas Yojana", 
            "PM Ujjwala", "Sukanya Samriddhi", "Atal Pension",
            "PM MUDRA", "PM Vishwakarma", "PM YASASVI"
        ]
    return list(dict.fromkeys(found_names))[:15] # Return top 15 unique names

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
        template="Answer based on context. Support English/Hindi.\nContext: {context}\nQuestion: {question}\nAnswer:",
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

# --- Display Suggestions at Prompt ---
st.write("### Choose a scheme to learn more:")
all_schemes = get_all_scheme_names("schemes.txt")

# Using st.pills for the clickable options
selected_pill = st.pills(
    "Available Schemes:", 
    all_schemes, 
    selection_mode="single", 
    label_visibility="collapsed"
)

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Combine Input: Either from Pill or from Text Box
user_query = None
if selected_pill:
    user_query = f"Tell me about {selected_pill}"

if chat_input := st.chat_input("Or type your question here..."):
    user_query = chat_input

# --- Generate Response ---
if user_query:
    # Avoid repeating if the same pill is clicked twice
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            chain = initialize_rag()
            response = chain.invoke(user_query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
