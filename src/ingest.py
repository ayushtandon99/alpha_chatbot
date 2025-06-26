import os
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import PyPDF2
import docx2txt

def load_documents(directory):
    """Load documents from the specified directory"""
    docs = []
    print(f"Loading documents from: {directory}")
    
    if not os.path.exists(directory):
        print(f"Directory {directory} not found!")
        return docs
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        text = ""
        
        try:
            if filename.endswith(".txt"):
                text = extract_text_from_txt(file_path)
            elif filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif filename.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            else:
                continue
                
            if text.strip():
                docs.append(Document(
                    page_content=text, 
                    metadata={"source": filename, "type": filename.split('.')[-1]}
                ))
                print(f"Loaded: {filename}")
            else:
                print(f"Empty content: {filename}")
                
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    print(f"Total documents loaded: {len(docs)}")
    return docs

def extract_text_from_txt(txt_path):
    """Extract text from TXT file"""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        # Using docx2txt for better text extraction
        text = docx2txt.process(docx_path)
        return text
    except ImportError:
        # Fallback to python-docx
        doc = docx2txt.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])

def embed_and_store(docs):
    """Create embeddings and store in FAISS index"""
    if not docs:
        print("No documents to process!")
        return
    
    print("Initializing embedding model...")
    model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separator="\n"
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating FAISS index...")
    db = FAISS.from_documents(chunks, embedding=model)
    
    print("Saving FAISS index...")
    db.save_local("faiss_index")
    print("Embedding complete. Vector index saved to 'faiss_index/'")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "../docs")
    documents = load_documents(docs_dir)
    if documents:
        embed_and_store(documents)
    else:
        print("No documents found to process!")
