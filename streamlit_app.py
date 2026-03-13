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

# --- Function to Extract Scheme Names from your file ---
def get_all_scheme_names(file_path):
    if not os.path.exists(file_path):
        return ["PM Kisan", "Ayushman Bharat"]
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # This looks for lines like "Scheme: PM Kisan" or "Name: PM Kisan"
    # If your file just lists names, we can adjust this regex.
    found_names = re.findall(r"(?:Scheme|Name|Yojana):\s*(.*)", content)
    
    if not found_names:
        # Fallback list if the scanner doesn't find the "Scheme:" tag
        return ["PM Kisan", "Ayushman Bharat", "PM Vishwakarma", "Sukanya Samriddhi"]
    
    return list(dict.fromkeys(found_names))[:15] # Return unique names

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
        template="""You are Jan Sahayak AI. Answer using the context. 
        Match the user's language (Hindi or English).
        If not in context, use your knowledge to help.
        
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

# --- Sidebar: News & Language Support ---
with st.sidebar:
    st.title("🏛 Info Hub")
    st.markdown("### 🌐 Language / भाषा\nSupports **English** & **हिंदी**")
    st.divider()
    
    st.markdown("### 🔔 Latest Gov News (PIB)")
    try:
        # Live RSS feed from Press Information Bureau
        feed = feedparser.parse("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1")
        for entry in feed.entries[:5]:
            st.markdown(f"**• [{entry.title}]({entry.link})**")
            st.caption(f"📅 {entry.published[:16]}")
    except:
        st.write("Live news feed is refreshing...")
    
    st.divider()
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main App ---
st.title("🏛 Jan Sahayak AI")
st.caption("Real-time Government Schemes Guide | अंग्रेजी और हिंदी सहायता")

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Please add OPENAI_API_KEY to Streamlit Secrets!")
    st.stop()

# --- Dynamic Pills (Options from File) ---
st.write("### Quick Select:")
all_schemes = get_all_scheme_names("schemes.txt")
selected_pill = st.pills("Choose a scheme:", all_schemes, selection_mode="single", label_visibility="collapsed")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Combined Input Handling ---
user_query = None
if selected_pill:
    user_query = f"Provide full details of {selected_pill}"

if prompt := st.chat_input("Ask a question in English or Hindi..."):
    user_query = prompt

if user_query:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                chain = initialize_rag()
                response = chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
