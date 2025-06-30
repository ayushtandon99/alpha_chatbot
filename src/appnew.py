import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# --- Streamlit page config ---
st.set_page_config(
    page_title="AI Policy Assistant Multilingual",
    page_icon="🤖",
    layout="centered"
)

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "user_language" not in st.session_state:
    st.session_state.user_language = "English"

# --- Utility functions ---
def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def load_llm():
    try:
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            do_sample=True,
            temperature=0.3
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

def init_qa_chain():
    llm = load_llm()
    if not llm:
        return None
    vector_store = load_faiss_index()
    if not vector_store:
        st.error("FAISS index not found! Please process documents first.")
        return None
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        return_source_documents=True
    )

def get_translation_pipeline():
    try:
        pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
        return pipe
    except Exception as e:
        st.error(f"Error loading translation pipeline: {e}")
        return None

def translate_text(pipe, text, target_language):
    prompt = f"Translate to {target_language}: {text}"
    try:
        result = pipe(prompt)
        return result[0]['generated_text'].strip()
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# --- Main UI ---
st.title("📄 AI Policy Assistant Multilingual")
st.caption("Ask questions about your pre-processed policy documents in your preferred language")

# --- Language selection ---
st.session_state.user_language = st.selectbox(
    "Choose your preferred language:",
    ["English", "Spanish", "French", "German"],
    index=0
)

# --- Load QA chain if not already loaded ---
if st.session_state.qa_chain is None:
    st.session_state.qa_chain = init_qa_chain()

if st.session_state.qa_chain:
    st.divider()
    st.subheader("💬 Chat with your documents")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.caption(f"- {source}")

    # Chat input
    if prompt := st.chat_input(f"Ask about your documents in {st.session_state.user_language}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                try:
                    # Translation pipeline
                    translator = get_translation_pipeline()
                    # Translate user input to English
                    translated_question = translate_text(
                        translator, 
                        prompt, 
                        "English"
                    ) if st.session_state.user_language != "English" else prompt

                    # Get answer from documents
                    result = st.session_state.qa_chain.invoke({"question": translated_question})
                    answer = result.get('answer', 'Sorry, I could not find an answer.')

                    # Translate answer to user's language
                    translated_answer = (
                        translate_text(translator, answer, st.session_state.user_language)
                        if st.session_state.user_language != "English" else answer
                    )

                    # Stream the response
                    for chunk in translated_answer.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    # Show sources
                    sources = set()
                    if result.get('source_documents'):
                        for doc in result['source_documents']:
                            sources.add(doc.metadata.get('source', 'Unknown'))
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.caption(f"- {source}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": list(sources)
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.error("No FAISS index found. Please ensure your documents are processed and the index exists.")

# --- Sidebar ---
with st.sidebar:
    st.header("Document Repository")
    docs_dir = "../docs"
    if os.path.exists(docs_dir) and os.path.isdir(docs_dir):
        st.success(f"Document repository: {os.path.abspath(docs_dir)}")
        doc_files = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
        st.write(f"Contains {len(doc_files)} file(s)")
    else:
        st.warning("Document repository not found")
    st.divider()
    st.header("Chat Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.success("Chat history cleared!")
        st.rerun()
    st.divider()
    st.header("About")
    st.markdown("""
    This multilingual assistant uses:
    - **Embeddings**: all-MiniLM-L6-v2
    - **LLM & Translation**: google/flan-t5-base
    - **Vector Store**: FAISS
    """)

# --- Create temp directory if needed ---
os.makedirs("temp", exist_ok=True)
