import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_llm():
    """Load HuggingFace LLM pipeline"""
    print("Loading LLM: google/flan-t5-base...")
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
        print(f"Error loading LLM: {e}")
        return None

def load_faiss_index():
    """Load FAISS index with matching embeddings"""
    print("Loading FAISS index...")
    try:
        # Use the same embedding model as ingest.py
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if not os.path.exists("faiss_index"):
            print("FAISS index not found! Run ingest.py first.")
            return None
            
        db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully")
        return db
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def init_memory():
    """Initialize conversation memory"""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

def load_chat_history(memory, history_file="chat_history.pkl"):
    """Load previous chat history"""
    if os.path.exists(history_file):
        try:
            with open(history_file, "rb") as f:
                history = pickle.load(f)
                for entry in history:
                    memory.save_context(
                        {"question": entry["question"]}, 
                        {"answer": entry["answer"]}
                    )
                print(f"Chat history loaded ({len(history)} entries)")
        except Exception as e:
            print(f"Could not load chat history: {e}")

def save_chat_history(memory, history_file="chat_history.pkl"):
    """Save chat history"""
    try:
        history = []
        for message in memory.chat_memory.messages:
            if hasattr(message, 'content'):
                if len(history) % 2 == 0:
                    history.append({"question": message.content, "answer": ""})
                else:
                    history[-1]["answer"] = message.content
        
        with open(history_file, "wb") as f:
            pickle.dump(history, f)
        print("Chat history saved")
    except Exception as e:
        print(f"Could not save chat history: {e}")

def get_translation_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

def translate_text(pipe, text, target_language):
    prompt = f"Translate to {target_language}: {text}"
    try:
        result = pipe(prompt)
        return result[0]['generated_text'].strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Fallback to original

def start_chat():
    print("Initializing AI Document Assistant...")

    # Load components
    llm = load_llm()
    if not llm:
        return

    db = load_faiss_index()
    if not db:
        return

    memory = init_memory()
    load_chat_history(memory)

    # Ask user for preferred language
    print("\nAvailable languages: English, Spanish, French, German")
    user_language = input("Choose your preferred language: ").strip()
    if not user_language:
        user_language = "English"

    print(f"Selected Language: {user_language}")
    translator = get_translation_pipeline()

    # Setup QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    print("\nHi, Welcome to your AI Policy Assistant!")
    print("Ask questions about your documents or type 'exit' to quit\n")

    while True:
        try:
            user_input = input(f"You ({user_language}): ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                save_chat_history(memory)
                print("Goodbye! Chat history saved.")
                break

            if not user_input:
                continue

            # Translate user question to English
            translated_question = translate_text(translator, user_input, "English")

            print("Searching documents...")
            result = qa_chain.invoke({"question": translated_question})

            answer = result.get("answer", "Sorry, I could not find an answer.")

            # Translate assistant answer back to user's language
            translated_answer = translate_text(translator, answer, user_language)
            print(f"\nAssistant ({user_language}): {translated_answer}")

            # Show sources
            if result.get("source_documents"):
                print("\nSources:")
                sources = {doc.metadata.get("source", "Unknown") for doc in result["source_documents"][:2]}
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            print()

        except KeyboardInterrupt:
            print("\nInterrupted. Saving chat history...")
            save_chat_history(memory)
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    start_chat()
