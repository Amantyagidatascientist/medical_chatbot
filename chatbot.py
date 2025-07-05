from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class MedicalChatbot:
    def __init__(self):
        # Configuration
        self.BASE_DIR = Path(__file__).parent
        self.MODEL_DIR = self.BASE_DIR / "model_storage"
        self.TEXT_MODEL_PATH = self.MODEL_DIR / "text_generator"
        self.EMBEDDING_MODEL_PATH = self.MODEL_DIR / "embedding"
        self.VECTORSTORE_PATH = self.BASE_DIR / "vectorstore/db_faiss"
        
        # Environment settings
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # Initialize components
        self._setup_directories()
        self._initialize_components()

    def _setup_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.VECTORSTORE_PATH.parent, exist_ok=True)

    def _initialize_components(self):
        """Initialize all chatbot components"""
        if not self.TEXT_MODEL_PATH.exists():
            self._download_model()
            
        try:
            self.llm = self._load_llm()
            self.embeddings = self._load_embeddings()
            self.db = self._load_vectorstore()
            self.qa_chain = self._create_qa_chain()
        except Exception as e:
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def _download_model(self):
        """Download and save the language model"""
        print("Downloading flan-t5-base...")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                resume_download=True
            )
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            
            # Save model
            self.TEXT_MODEL_PATH.mkdir(exist_ok=True)
            model.save_pretrained(self.TEXT_MODEL_PATH)
            tokenizer.save_pretrained(self.TEXT_MODEL_PATH)
            
            print(f"✅ Model saved to {self.TEXT_MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"❌ Model download failed: {str(e)}")

    def _load_llm(self):
        """Load the language model pipeline"""
        try:
            # Verify model files exist
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            for file in required_files:
                if not (self.TEXT_MODEL_PATH / file).exists():
                    raise FileNotFoundError(f"Missing model file: {file}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.TEXT_MODEL_PATH)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.TEXT_MODEL_PATH)
            
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
            raise RuntimeError(f"❌ LLM loading failed: {str(e)}")

    def _load_embeddings(self):
        """Load the embeddings model"""
        try:
            if not self.EMBEDDING_MODEL_PATH.exists():
                raise FileNotFoundError("Embedding model not found")
                
            return HuggingFaceEmbeddings(
                model_name=str(self.EMBEDDING_MODEL_PATH),
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "show_progress_bar": True
                }
            )
        except Exception as e:
            raise RuntimeError(f"❌ Embeddings loading failed: {str(e)}")

    def _load_vectorstore(self):
        """Load the FAISS vector store with validation"""
        try:
            required_files = ["index.faiss", "index.pkl"]
            for file in required_files:
                if not (self.VECTORSTORE_PATH / file).exists():
                    raise FileNotFoundError(f"Missing vectorstore file: {file}")
                    
            return FAISS.load_local(
                str(self.VECTORSTORE_PATH),
                self.embeddings,
                allow_dangerous_deserialization=False
            )
        except Exception as e:
            raise RuntimeError(f"❌ Vectorstore loading failed: {str(e)}")

    def _get_prompt_templates(self):
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

    def _create_qa_chain(self):
        """Create and configure the QA system"""
        try:
            question_prompt, combine_prompt = self._get_prompt_templates()
            
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="map_reduce",
                retriever=self.db.as_retriever(
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
            raise RuntimeError(f"❌ QA chain creation failed: {str(e)}")

    def query(self, question: str):
        """Handle user queries"""
        if not question or not isinstance(question, str):
            return {"error": "Invalid question - must be non-empty string"}
            
        try:
            response = self.qa_chain.invoke({'query': question})
            return {
                "answer": response["result"],
                "sources": [doc.page_content[:200] + "..." 
                           for doc in response["source_documents"]]
            }
        except Exception as e:
            return {"error": f"Query processing failed: {str(e)}"}

    def run_chat_interface(self):
        """Run interactive chat interface"""
        print("\n🩺 Medical Anatomy Chatbot (Type 'quit' to exit)")
        while True:
            try:
                user_input = input("\nQuestion: ")
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                result = self.query(user_input)
                if "error" in result:
                    print(f"\n⚠️ Error: {result['error']}")
                else:
                    print(f"\nAnswer: {result['answer']}")
                    if result["sources"]:
                        print("\nSources:")
                        for i, source in enumerate(result["sources"], 1):
                            print(f"{i}. {source}")
                            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\n⚠️ Unexpected error: {str(e)}")

if __name__ == "__main__":
    try:
        chatbot = MedicalChatbot()
        chatbot.run_chat_interface()
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")