import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Jan Sahayak AI", page_icon="🏛", layout="centered")

st.title("🏛 Government Schemes Assistant")
st.markdown("Ask me anything about Indian government schemes in **English** or **Hindi**.")

# --- RAG System Setup ---
@st.cache_resource
def initialize_rag():
    file_path = "schemes.txt"
    
    if not os.path.exists(file_path):
        st.error(f"Error: {file_path} not found. Please upload it to your GitHub repository.")
        return None

    # Load and Split Data
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    # Create Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Retriever (increased 'k' to find more relevant context)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # LLM Setup
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Improved Prompt: Bilingual support and fallback to general knowledge
    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions about Indian government schemes.
        
        Rules:
        1. Use the provided context to answer the question.
        2. If the context doesn't contain the specific detail, use your general knowledge to provide a helpful and accurate answer.
        3. Answer in the SAME LANGUAGE the user uses (Hindi or English).
        4. If you don't know the answer at all, refer them to https://myscheme.gov.in.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The Chain
    rag_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return rag_chain

# --- App Execution ---
# 1. Access the API Key from Streamlit Secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.warning("⚠️ Please add your OPENAI_API_KEY to Streamlit Secrets.")
    st.stop()

# 2. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if user_query := st.chat_input("Ask about PM Kisan, Ayushman Bharat, etc..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching schemes..."):
            try:
                chain = initialize_rag()
                response = chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
