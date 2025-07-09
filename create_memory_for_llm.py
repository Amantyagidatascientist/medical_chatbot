import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


DATA_PATH = Path("data")
EMBEDDING_MODEL_PATH = Path("local_embedding_model")
VECTORSTORE_PATH = Path("vectorstore/db_faiss")

def setup_environment():
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    DATA_PATH.mkdir(exist_ok=True)
    VECTORSTORE_PATH.parent.mkdir(exist_ok=True)

def load_pdf_files(data_path):
    if not any(data_path.glob("*.pdf")):
        raise FileNotFoundError(f"No PDF files found in {data_path}")
    
    loader = DirectoryLoader(
        str(data_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    return loader.load()

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(extracted_data)

def setup_embedding_model():
    if not EMBEDDING_MODEL_PATH.exists():
        print("Downloading embedding model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model.save(str(EMBEDDING_MODEL_PATH))
    
    return HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_PATH),
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "show_progress_bar": True
        }
    )

def create_vector_store(text_chunks, embedding_model):
    db = FAISS.from_documents(text_chunks, embedding_model)
    
    if db.index.ntotal == 0:
        raise ValueError("Empty vector store created - check input documents")
    
    return db

def main():
    try:
        setup_environment()
        
        print("\n[1/4] Loading PDF documents...")
        documents = load_pdf_files(DATA_PATH)
        
        print("[2/4] Splitting text into chunks...")
        chunks = create_chunks(documents)
        print(f"Created {len(chunks)} text chunks")
        
        print("[3/4] Loading embedding model...")
        embeddings = setup_embedding_model()
        
        print("[4/4] Creating vector store...")
        vector_db = create_vector_store(chunks, embeddings)
        vector_db.save_local(str(VECTORSTORE_PATH))
        
        print(f"\nSuccessfully created vector store at: {VECTORSTORE_PATH}")
        print(f"Index contains {vector_db.index.ntotal} vectors")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        # Cleanup failed attempts
        if 'vector_db' in locals():
            try:
                import shutil
                shutil.rmtree(VECTORSTORE_PATH)
            except:
                pass

if __name__ == "__main__":
    main()