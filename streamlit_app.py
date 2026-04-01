import streamlit as st
import os
import re
import json
import tempfile
import feedparser
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Jan Sahayak AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --navy: #1B3A6B;
    --orange: #E05C1A;
    --green: #1A7A4A;
    --light: #F4F7FB;
    --white: #FFFFFF;
    --gray: #6B7280;
}

.stApp { background: #F4F7FB; font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

/* App Header */
.app-header {
    background: linear-gradient(135deg, #1B3A6B 0%, #0f2347 100%);
    padding: 20px 28px; border-radius: 16px;
    margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
}
.app-header-title { color: white; font-size: 26px; font-weight: 800; margin: 0; }
.app-header-sub { color: rgba(255,255,255,0.6); font-size: 13px; margin: 4px 0 0; }

/* Feature Cards */
.feature-card {
    background: white; border-radius: 14px;
    padding: 20px; border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.feature-title {
    font-size: 16px; font-weight: 700;
    color: #1B3A6B; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}

/* Status Badges */
.badge-green { background: #D1FAE5; color: #065F46; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-red { background: #FEE2E2; color: #991B1B; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-blue { background: #DBEAFE; color: #1E40AF; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }

/* Bookmarks */
.bookmark-card {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-radius: 12px; padding: 14px 16px;
    margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;
}

/* Document checklist */
.doc-item {
    background: white; border: 1px solid #E5E7EB;
    border-radius: 10px; padding: 10px 14px;
    margin-bottom: 8px; font-size: 14px;
    display: flex; align-items: center; gap: 10px;
}

/* Comparison table override */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white; border-radius: 12px;
    padding: 4px; gap: 4px;
    border: 1px solid #E5E7EB;
    overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background: #1B3A6B !important;
    color: white !important;
}

/* News item */
.news-item {
    background: white; border-left: 3px solid #1B3A6B;
    padding: 10px 12px; border-radius: 0 8px 8px 0;
    margin-bottom: 8px; font-size: 13px;
}

/* Eligibility result */
.eligible-yes {
    background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
    border: 1px solid #34D399; border-radius: 14px;
    padding: 20px; text-align: center; font-size: 18px;
    font-weight: 700; color: #065F46;
}
.eligible-no {
    background: linear-gradient(135deg, #FEE2E2, #FECACA);
    border: 1px solid #F87171; border-radius: 14px;
    padding: 20px; text-align: center; font-size: 18px;
    font-weight: 700; color: #991B1B;
}
.eligible-maybe {
    background: linear-gradient(135deg, #FEF3C7, #FDE68A);
    border: 1px solid #FBBF24; border-radius: 14px;
    padding: 20px; text-align: center; font-size: 18px;
    font-weight: 700; color: #92400E;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif os.environ.get("OPENAI_API_KEY"):
    pass
else:
    st.error("🔑 Please add OPENAI_API_KEY to Streamlit Secrets!")
    st.stop()


# ══════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════
def get_all_scheme_names(file_path="schemes.txt"):
    fallback = [
        "PM Kisan", "Ayushman Bharat", "PM Awas Yojana",
        "PM Ujjwala Yojana", "Sukanya Samriddhi", "PM Mudra Yojana",
        "Atal Pension Yojana", "PM Vishwakarma", "Ladli Behna",
        "Free Ration (PMGKAY)"
    ]
    if not os.path.exists(file_path):
        return fallback
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    found = re.findall(r"(?:Scheme|Name|Yojana):\s*(.*)", content)
    return list(dict.fromkeys(found))[:15] if found else fallback


# ══════════════════════════════════════════
# RAG SYSTEM — multiple chains
# ══════════════════════════════════════════
@st.cache_resource
def initialize_rag():
    if not os.path.exists("schemes.txt"):
        st.error("schemes.txt not found! Please add your data file.")
        st.stop()

    with open("schemes.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def make_chain(template):
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        return (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            | prompt | llm | StrOutputParser()
        )

    # 1. General chat chain
    chat_chain = make_chain("""You are Jan Sahayak AI, a helpful guide for Indian government schemes.
Answer clearly and helpfully. Match the user's language (Hindi or English).
If not in context, use your general knowledge to help.

Context: {context}
Question: {question}
Answer:""")

    # 2. Eligibility checker chain
    eligibility_chain = make_chain("""You are an eligibility checker for Indian government schemes.
Based on the user's profile and the scheme information, determine if they are eligible.

Start your response with one of these exact words on the first line:
ELIGIBLE, NOT ELIGIBLE, or PARTIALLY ELIGIBLE

Then explain in 3-5 bullet points why or why not.
Match the user's language (Hindi or English).

Context: {context}
User Profile and Scheme: {question}
Assessment:""")

    # 3. Document checklist chain
    docs_chain = make_chain("""You are a document specialist for Indian government schemes.
List ALL documents required to apply for this scheme.
Format as a numbered list. Be complete and specific.
Match the user's language (Hindi or English).

Context: {context}
Scheme: {question}
Required Documents:""")

    # 4. Comparison chain
    comparison_chain = make_chain("""You are comparing Indian government schemes.
Compare the given schemes across these points: Benefits, Eligibility, How to Apply, Documents Needed, Official Website.
Format as a clear comparison. Match the user's language.

Context: {context}
Schemes to compare: {question}
Comparison:""")

    # 5. Complaint letter chain
    complaint_chain = make_chain("""You are a government complaint letter writer.
Write a formal complaint letter in Hindi and English based on the details provided.
Include: proper salutation, clear description of issue, specific demand, and signature block.

Context: {context}
Complaint details: {question}
Formal Complaint Letter:""")

    # 6. Nearby office chain
    office_chain = make_chain("""You are a government office locator for India.
Based on the scheme and state/city provided, tell the user:
1. Which government office/department handles this scheme
2. What is the official website
3. What is the helpline number
4. What is the process to find the nearest office
Match the user's language.

Context: {context}
Scheme and Location: {question}
Office Information:""")

    return {
        "chat": chat_chain,
        "eligibility": eligibility_chain,
        "docs": docs_chain,
        "comparison": comparison_chain,
        "complaint": complaint_chain,
        "office": office_chain,
    }


# ══════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════
defaults = {
    "messages": [],
    "bookmarks": [],
    "eligibility_result": None,
    "docs_result": None,
    "comparison_result": None,
    "complaint_result": None,
    "office_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════
# LOAD CHAINS
# ══════════════════════════════════════════
chains = initialize_rag()
all_schemes = get_all_scheme_names()


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏛️ Jan Sahayak AI")
    st.markdown("**भाषा / Language:** Hindi & English")
    st.divider()

    # Live news
    st.markdown("### 📰 Latest Gov News")
    try:
        feed = feedparser.parse("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1")
        for entry in feed.entries[:5]:
            st.markdown(f"""
            <div class="news-item">
                <a href="{entry.link}" target="_blank" style="color:#1B3A6B; font-weight:600; text-decoration:none;">
                    {entry.title[:70]}...
                </a>
                <div style="color:#6B7280; font-size:11px; margin-top:4px;">
                    📅 {entry.get('published', '')[:16]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.info("News feed loading...")

    st.divider()

    # Bookmarks in sidebar
    st.markdown("### 🔖 My Bookmarks")
    if st.session_state.bookmarks:
        for bm in st.session_state.bookmarks:
            st.markdown(f"""
            <div class="bookmark-card">
                <span style="font-size:13px; font-weight:600; color:#92400E;">⭐ {bm['scheme']}</span>
                <span style="font-size:11px; color:#6B7280;">{bm['date']}</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑️ Clear Bookmarks", use_container_width=True):
            st.session_state.bookmarks = []
            st.rerun()
    else:
        st.caption("No bookmarks yet. Save schemes from the Chat tab!")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div style="font-size:42px;">🏛️</div>
    <div>
        <p class="app-header-title">Jan Sahayak AI</p>
        <p class="app-header-sub">Your Complete Guide to Indian Government Schemes | सरकारी योजनाओं का पूरा मार्गदर्शक</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════
tab_chat, tab_eligibility, tab_docs, tab_compare, tab_complaint, tab_office, tab_bookmarks = st.tabs([
    "💬 Chat",
    "✅ Eligibility Check",
    "📋 Document Checklist",
    "⚖️ Compare Schemes",
    "📝 Complaint Letter",
    "📍 Find Office",
    "🔖 Bookmarks"
])


# ════════════════════════════════
# TAB 1: CHAT
# ════════════════════════════════
with tab_chat:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 💬 Ask Anything About Government Schemes")

        # Quick pills
        selected_pill = st.pills(
            "Quick Select:",
            all_schemes,
            selection_mode="single",
            label_visibility="collapsed"
        )

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle input
        user_query = None
        if selected_pill:
            user_query = f"Provide full details of {selected_pill} scheme including benefits, eligibility, and how to apply."
        if prompt := st.chat_input("Ask in English or Hindi... / हिंदी या अंग्रेज़ी में पूछें..."):
            user_query = prompt

        if user_query:
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.chat_message("assistant"):
                    with st.spinner("Searching schemes..."):
                        response = chains["chat"].invoke(user_query)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        # Voice Input
        st.markdown("### 🎤 Voice Input")
        st.caption("Speak your question in Hindi or English")
        audio = st.audio_input("Tap to speak", key="voice_chat")
        if audio:
            with st.spinner("Transcribing..."):
                try:
                    from openai import OpenAI
                    client = OpenAI()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio.getvalue())
                        tmp_path = tmp.name
                    with open(tmp_path, "rb") as af:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", file=af
                        )
                    os.unlink(tmp_path)
                    voice_text = transcript.text
                    st.success(f"🗣️ You said: **{voice_text}**")
                    # Add to chat
                    st.session_state.messages.append({"role": "user", "content": voice_text})
                    with st.spinner("Getting answer..."):
                        response = chains["chat"].invoke(voice_text)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Voice error: {str(e)}")

        st.divider()

        # Bookmark current scheme
        st.markdown("### 🔖 Bookmark a Scheme")
        scheme_to_save = st.selectbox("Select scheme:", all_schemes, key="bm_select")
        if st.button("⭐ Save Bookmark", use_container_width=True):
            already = any(b["scheme"] == scheme_to_save for b in st.session_state.bookmarks)
            if not already:
                st.session_state.bookmarks.append({
                    "scheme": scheme_to_save,
                    "date": datetime.now().strftime("%d %b %Y")
                })
                st.success(f"Saved: {scheme_to_save}!")
            else:
                st.info("Already bookmarked!")


# ════════════════════════════════
# TAB 2: ELIGIBILITY CHECKER
# ════════════════════════════════
with tab_eligibility:
    st.markdown("### ✅ Check Your Eligibility")
    st.caption("Fill your profile below and we will check if you qualify for a scheme.")

    col1, col2 = st.columns(2)
    with col1:
        e_scheme = st.selectbox("Select Scheme", all_schemes, key="e_scheme")
        e_age = st.number_input("Your Age", min_value=1, max_value=100, value=30)
        e_income = st.selectbox("Annual Family Income", [
            "Below Rs. 1 Lakh", "Rs. 1-2 Lakh", "Rs. 2-5 Lakh",
            "Rs. 5-10 Lakh", "Above Rs. 10 Lakh"
        ])
        e_state = st.selectbox("State", [
            "Delhi", "Uttar Pradesh", "Bihar", "Madhya Pradesh",
            "Maharashtra", "Rajasthan", "Gujarat", "West Bengal",
            "Karnataka", "Tamil Nadu", "Other"
        ])
    with col2:
        e_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        e_category = st.selectbox("Category", ["General", "OBC", "SC", "ST", "EWS"])
        e_occupation = st.selectbox("Occupation", [
            "Farmer", "Daily Wage Worker", "Small Business Owner",
            "Government Employee", "Private Employee", "Unemployed",
            "Student", "Homemaker", "Other"
        ])
        e_land = st.selectbox("Land Ownership", [
            "No land", "Less than 2 acres", "2-5 acres", "More than 5 acres"
        ])

    if st.button("🔍 Check My Eligibility", use_container_width=True, type="primary"):
        profile = f"""
        Scheme: {e_scheme}
        Age: {e_age} years
        Gender: {e_gender}
        Category: {e_category}
        Annual Income: {e_income}
        State: {e_state}
        Occupation: {e_occupation}
        Land: {e_land}
        """
        with st.spinner("Checking eligibility..."):
            result = chains["eligibility"].invoke(profile)
            st.session_state.eligibility_result = result

    if st.session_state.eligibility_result:
        result = st.session_state.eligibility_result
        first_line = result.strip().split("\n")[0].upper()

        if "NOT ELIGIBLE" in first_line:
            st.markdown('<div class="eligible-no">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)
        elif "PARTIALLY" in first_line:
            st.markdown('<div class="eligible-maybe">⚠️ PARTIALLY ELIGIBLE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="eligible-yes">✅ YOU ARE ELIGIBLE!</div>', unsafe_allow_html=True)

        st.markdown("---")
        rest = "\n".join(result.strip().split("\n")[1:])
        st.markdown(rest)

        # Offer to generate documents
        if st.button("📋 Get Required Documents for this Scheme"):
            with st.spinner("Fetching document list..."):
                doc_result = chains["docs"].invoke(e_scheme)
                st.session_state.docs_result = doc_result
            st.success("Check the Document Checklist tab!")


# ════════════════════════════════
# TAB 3: DOCUMENT CHECKLIST
# ════════════════════════════════
with tab_docs:
    st.markdown("### 📋 Document Checklist Generator")
    st.caption("Get the exact list of documents you need to apply for any scheme.")

    d_scheme = st.selectbox("Select Scheme", all_schemes, key="d_scheme")
    lang_pref = st.radio("Response Language", ["English", "Hindi"], horizontal=True)

    if st.button("📄 Generate Document Checklist", use_container_width=True, type="primary"):
        query = f"{d_scheme} - Please respond in {lang_pref}"
        with st.spinner("Generating checklist..."):
            result = chains["docs"].invoke(query)
            st.session_state.docs_result = result

    if st.session_state.docs_result:
        st.markdown("---")
        st.markdown(f"#### 📋 Documents Required for: **{d_scheme}**")

        # Parse and display nicely
        lines = st.session_state.docs_result.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                clean = re.sub(r"^[\d\.\-\•\*]\s*", "", line).strip()
                if clean:
                    st.markdown(f"""
                    <div class="doc-item">
                        <span style="font-size:18px;">📄</span>
                        <span style="font-size:14px; color:#1A1A1A;">{clean}</span>
                    </div>
                    """, unsafe_allow_html=True)
            elif line and not line[0].isdigit():
                st.markdown(f"**{line}**")

        # Download button
        st.download_button(
            "⬇️ Download Checklist",
            data=st.session_state.docs_result,
            file_name=f"{d_scheme}_documents.txt",
            mime="text/plain"
        )


# ════════════════════════════════
# TAB 4: COMPARE SCHEMES
# ════════════════════════════════
with tab_compare:
    st.markdown("### ⚖️ Compare Government Schemes")
    st.caption("Select 2 or 3 schemes to compare side by side.")

    c_schemes = st.multiselect(
        "Select schemes to compare (2-3 recommended):",
        all_schemes,
        default=all_schemes[:2],
        max_selections=3
    )

    if st.button("⚖️ Compare Now", use_container_width=True, type="primary"):
        if len(c_schemes) < 2:
            st.warning("Please select at least 2 schemes to compare.")
        else:
            query = f"Compare these schemes in detail: {', '.join(c_schemes)}"
            with st.spinner("Comparing schemes..."):
                result = chains["comparison"].invoke(query)
                st.session_state.comparison_result = result

    if st.session_state.comparison_result:
        st.markdown("---")
        st.markdown(st.session_state.comparison_result)

        st.download_button(
            "⬇️ Download Comparison",
            data=st.session_state.comparison_result,
            file_name="scheme_comparison.txt",
            mime="text/plain"
        )


# ════════════════════════════════
# TAB 5: COMPLAINT LETTER
# ════════════════════════════════
with tab_complaint:
    st.markdown("### 📝 Complaint Letter Generator")
    st.caption("Generate a formal complaint letter for scheme-related issues.")

    col1, col2 = st.columns(2)
    with col1:
        cp_name = st.text_input("Your Full Name", placeholder="e.g. Ramesh Kumar")
        cp_scheme = st.selectbox("Scheme Related To", all_schemes, key="cp_scheme")
        cp_state = st.text_input("Your State/District", placeholder="e.g. Lucknow, UP")
    with col2:
        cp_issue = st.selectbox("Type of Issue", [
            "Benefit not received",
            "Application rejected without reason",
            "Corruption / Bribery demanded",
            "Wrong information given by official",
            "Technical issue on portal",
            "Discrimination / Denial of service",
            "Other"
        ])
        cp_date = st.date_input("Date of Incident")
        cp_lang = st.radio("Letter Language", ["Hindi", "English", "Both"], horizontal=True)

    cp_desc = st.text_area(
        "Describe your problem in detail",
        placeholder="What happened? When? Who was involved? What do you want as resolution?",
        height=120
    )

    if st.button("📝 Generate Complaint Letter", use_container_width=True, type="primary"):
        if cp_name and cp_desc:
            details = f"""
            Name: {cp_name}
            Scheme: {cp_scheme}
            Issue: {cp_issue}
            State/District: {cp_state}
            Date of Incident: {cp_date}
            Language: {cp_lang}
            Description: {cp_desc}
            """
            with st.spinner("Writing your complaint letter..."):
                result = chains["complaint"].invoke(details)
                st.session_state.complaint_result = result
        else:
            st.warning("Please enter your name and describe the problem.")

    if st.session_state.complaint_result:
        st.markdown("---")
        st.markdown("#### ✉️ Your Complaint Letter:")
        st.text_area("Letter (copy from here):", st.session_state.complaint_result, height=400)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Letter",
                data=st.session_state.complaint_result,
                file_name="complaint_letter.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            import urllib.parse
            wa_text = urllib.parse.quote(st.session_state.complaint_result[:500] + "...")
            st.link_button(
                "💚 Share on WhatsApp",
                f"https://wa.me/?text={wa_text}",
                use_container_width=True
            )


# ════════════════════════════════
# TAB 6: FIND OFFICE
# ════════════════════════════════
with tab_office:
    st.markdown("### 📍 Find Nearest Government Office")
    st.caption("Know which office to visit for your scheme and how to reach them.")

    col1, col2 = st.columns(2)
    with col1:
        o_scheme = st.selectbox("Select Scheme", all_schemes, key="o_scheme")
        o_state = st.selectbox("Your State", [
            "Delhi", "Uttar Pradesh", "Bihar", "Madhya Pradesh",
            "Maharashtra", "Rajasthan", "Gujarat", "West Bengal",
            "Karnataka", "Tamil Nadu", "Punjab", "Haryana",
            "Andhra Pradesh", "Telangana", "Other"
        ])
    with col2:
        o_city = st.text_input("Your City / District", placeholder="e.g. Varanasi")
        o_lang = st.radio("Response Language", ["English", "Hindi"], horizontal=True, key="o_lang")

    if st.button("📍 Find Office & Contact Details", use_container_width=True, type="primary"):
        query = f"Scheme: {o_scheme}, State: {o_state}, City: {o_city}. Respond in {o_lang}."
        with st.spinner("Finding office details..."):
            result = chains["office"].invoke(query)
            st.session_state.office_result = result

    if st.session_state.office_result:
        st.markdown("---")
        st.markdown(st.session_state.office_result)

        # Quick helpline numbers
        st.markdown("---")
        st.markdown("#### 📞 Important Helpline Numbers")
        helplines = [
            ("PM Kisan Helpline", "155261 / 011-23381092"),
            ("Ayushman Bharat", "14555"),
            ("PM Awas Yojana", "1800-11-6446"),
            ("Ujjwala Yojana", "1906"),
            ("PMEGP (MUDRA)", "1800-11-0001"),
            ("General Gov Helpline", "1800-11-7788"),
        ]
        cols = st.columns(3)
        for i, (name, number) in enumerate(helplines):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:white; border:1px solid #E5E7EB; border-radius:10px;
                    padding:12px; margin-bottom:10px; text-align:center;">
                    <div style="font-size:12px; color:#6B7280; margin-bottom:4px;">{name}</div>
                    <div style="font-size:16px; font-weight:700; color:#1B3A6B;">📞 {number}</div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════
# TAB 7: BOOKMARKS
# ════════════════════════════════
with tab_bookmarks:
    st.markdown("### 🔖 My Saved Schemes")

    if not st.session_state.bookmarks:
        st.info("💡 You haven't bookmarked any schemes yet. Go to the Chat tab and save schemes you're interested in!")
    else:
        st.caption(f"You have {len(st.session_state.bookmarks)} bookmarked scheme(s).")

        for i, bm in enumerate(st.session_state.bookmarks):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"### ⭐ {bm['scheme']}")
            with col2:
                st.caption(f"Saved on: {bm['date']}")
            with col3:
                if st.button("🗑️", key=f"del_bm_{i}"):
                    st.session_state.bookmarks.pop(i)
                    st.rerun()

            # Quick action buttons for each bookmark
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(f"💬 Chat about it", key=f"chat_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Tell me everything about {bm['scheme']}"
                    })
                    st.switch_page = "Chat"
                    st.info("Go to Chat tab to see the response!")
            with c2:
                if st.button(f"✅ Check Eligibility", key=f"elig_{i}", use_container_width=True):
                    st.info(f"Go to Eligibility tab and select '{bm['scheme']}'")
            with c3:
                if st.button(f"📋 Get Documents", key=f"docc_{i}", use_container_width=True):
                    with st.spinner("Fetching documents..."):
                        result = chains["docs"].invoke(bm["scheme"])
                        st.session_state.docs_result = result
                    st.success("Document list ready! Check Document Checklist tab.")

            st.divider()

        # Export bookmarks
        bookmark_text = "\n".join([f"- {b['scheme']} (saved {b['date']})" for b in st.session_state.bookmarks])
        st.download_button(
            "⬇️ Export My Bookmarks",
            data=bookmark_text,
            file_name="my_bookmarks.txt",
            mime="text/plain"
        )
