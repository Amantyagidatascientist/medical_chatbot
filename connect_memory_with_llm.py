from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Constants
MODEL_DIR = Path("./model_storage")
TEXT_MODEL_PATH = MODEL_DIR / "text_generator"
EMBEDDING_MODEL_PATH = MODEL_DIR / "embedding"
VECTORSTORE_PATH = Path("vectorstore/db_faiss")

# Configuration
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def initialize_directories():
    """Ensure all required directories exist"""
    MODEL_DIR.mkdir(exist_ok=True)
    VECTORSTORE_PATH.parent.mkdir(exist_ok=True)

def download_model():
    """Download and save the language model"""
    print("Downloading flan-t5-base...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        
        TEXT_MODEL_PATH.mkdir(exist_ok=True)
        model.save_pretrained(TEXT_MODEL_PATH)
        tokenizer.save_pretrained(TEXT_MODEL_PATH)
        print(f"✅ Model saved to {TEXT_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}")
        raise

def load_llm():
    """Load the language model pipeline"""
    try:
        if not TEXT_MODEL_PATH.exists():
            download_model()
            
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_PATH)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            device=0 if torch.cuda.is_available() else -1
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"❌ Failed to load LLM: {str(e)}")
        raise

def load_embeddings():
    """Load the embeddings model"""
    try:
        if not EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError("Embedding model directory not found")
            
        return HuggingFaceEmbeddings(
            model_name=str(EMBEDDING_MODEL_PATH),
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "show_progress_bar": True
            }
        )
    except Exception as e:
        print(f"❌ Failed to load embeddings: {str(e)}")
        raise

def load_vectorstore(embeddings):
    """Load the FAISS vector store with validation"""
    try:
        required_files = ["index.faiss", "index.pkl"]
        for file in required_files:
            if not (VECTORSTORE_PATH / file).exists():
                raise FileNotFoundError(f"Missing vectorstore file: {file}")
                
        return FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=False
        )
    except Exception as e:
        print(f"❌ Failed to load vector store: {str(e)}")
        raise

def get_prompt_templates():
    """Return configured prompt templates"""
    question_template = """
    You are a board-certified anatomist with expertise from Gray's Anatomy.
    Answer only using the context provided below. If unsure, say "I don't know."

    <context>
    {context}
    </context>

    Question: {question}
    Answer:"""
    
    combine_template = """
    Synthesize answers from multiple sources as a medical expert:
    
    <summaries>
    {summaries}
    </summaries>
    
    Final Answer:"""
    
    return (
        PromptTemplate(template=question_template, input_variables=["context", "question"]),
        PromptTemplate(template=combine_template, input_variables=["summaries"])
    )

def create_qa_chain(llm, embeddings):
    """Create and configure the QA system"""
    try:
        db = load_vectorstore(embeddings)
        question_prompt, combine_prompt = get_prompt_templates()
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=db.as_retriever(
                search_kwargs={
                    'k': 5,
                    'search_type': 'mmr',
                    'score_threshold': 0.7
                }
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "question_prompt": question_prompt,
                "combine_prompt": combine_prompt
            }
        )
    except Exception as e:
        print(f"❌ Failed to create QA chain: {str(e)}")
        raise

def main():
    """Run the medical chatbot interface"""
    try:
        initialize_directories()
        llm = load_llm()
        embeddings = load_embeddings()
        qa_chain = create_qa_chain(llm, embeddings)
        
        print("\n🩺 Medical Anatomy Chatbot (Type 'quit' to exit)")
        while True:
            try:
                query = input("\nQuestion: ")
                if query.lower() in ['quit', 'exit']:
                    break
                
                result = qa_chain.invoke({'query': query})
                print(f"\nAnswer: {result['result']}")
                
                if result['source_documents']:
                    print("\nSources:")
                    for i, doc in enumerate(result['source_documents'], 1):
                        print(f"{i}. {doc.page_content[:150]}...")
                        
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {str(e)}")
                
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")

if __name__ == "__main__":
    main()