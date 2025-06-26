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

def start_chat():
    """Main chat function"""
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
    
    # Setup QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )

    print("\n AI Document Assistant ready!")
    print("Ask questions about your documents or type 'exit' to quit\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ["exit", "quit", "bye"]:
                save_chat_history(memory)
                print("Goodbye! Chat history saved.")
                break
            
            if not question:
                continue
                
            print("Searching documents...")
            result = qa_chain.invoke({"question": question})
            
            answer = result.get('answer', 'Sorry, I could not find an answer.')
            print(f"\n Assistant: {answer}")
            
            # Show sources
            if result.get('source_documents'):
                print("\n Sources:")
                sources = set()
                for doc in result['source_documents'][:2]:
                    source = doc.metadata.get('source', 'Unknown')
                    sources.add(source)
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n Interrupted. Saving chat history...")
            save_chat_history(memory)
            break
        except Exception as e:
            print(f"\n Error: {str(e)}\n")

if __name__ == "__main__":
    start_chat()
