from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import torch
import os

class MedicalChatbot:
    # Set as class variables to ensure they're set before any HF operations
    HF_HUB_DOWNLOAD_TIMEOUT = "600"
    HF_ENDPOINT = "https://hf-mirror.com"
    
    # Prompt templates as class variables
    QUESTION_PROMPT_TEMPLATE = """
    You are a board-certified anatomist with expertise from Gray's Anatomy.
    Answer only using the context provided below. If the answer is not in the context, say "I don't know."

    <context>
    {context}
    </context>

    Question: {question}
    Answer:
    """

    COMBINE_PROMPT_TEMPLATE = """
    You are a board-certified anatomist with expertise from Gray's Anatomy.
    Use the following summarized answers from multiple sources to construct a final response. 
    If the answer is not in the summaries, say "I don't know."

    <summaries>
    {summaries}
    </summaries>

    Question: {question}
    Answer:
    """

    def __init__(self):
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = self.HF_HUB_DOWNLOAD_TIMEOUT
        os.environ["HF_ENDPOINT"] = self.HF_ENDPOINT
        
        if not os.path.exists("./local_text_generator_model"):
            self._save_model_locally()
            
        try:
            self.llm = self._load_llm()
            self.embeddings = self._load_embeddings()
            self.db = self._load_vectorstore(self.embeddings)
            self.qa_chain = self._create_qa_chain()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MedicalChatbot: {str(e)}")

    def _save_model_locally(self):
        """Save the model locally if it doesn't exist"""
        print("Saving flan-t5-small locally...")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            os.makedirs("./local_text_generator_model", exist_ok=True)
            model.save_pretrained("./local_text_generator_model")
            tokenizer.save_pretrained("./local_text_generator_model")
            print("flan-t5-small saved at ./local_text_generator_model")
        except Exception as e:
            raise RuntimeError(f"Failed to save model locally: {str(e)}")

    def _load_llm(self):
        """Load the language model"""
        try:
            model_path = "./local_text_generator_model"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")
                
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                device=0 if torch.cuda.is_available() else -1
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            raise RuntimeError(f"Failed to load local LLM: {str(e)}")

    def _load_embeddings(self):
        """Load embeddings model"""
        try:
            if not os.path.exists("./local_embedding_model"):
                raise FileNotFoundError("Embedding model directory not found")
                
            return HuggingFaceEmbeddings(
                model_name="./local_embedding_model",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": False}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings: {str(e)}")

    def _load_vectorstore(self, embeddings):
        """Load the FAISS vector store"""
        try:
            if not os.path.exists("vectorstore/db_faiss"):
                raise FileNotFoundError("Vectorstore directory not found")
                
            return FAISS.load_local(
                "vectorstore/db_faiss",
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load vector store: {str(e)}")

    def _get_question_prompt(self):
        """Get the question prompt template"""
        return PromptTemplate(
            template=self.QUESTION_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def _get_combine_prompt(self):
        return PromptTemplate(
            template=self.COMBINE_PROMPT_TEMPLATE,
            input_variables=["summaries", "question"]
        )
    
    def _create_qa_chain(self):
        try:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="map_reduce",
                retriever=self.db.as_retriever(
                    search_kwargs={'k': 5, 'search_type': 'mmr'}
                ),
                return_source_documents=True,
                chain_type_kwargs={
                    "question_prompt": self._get_question_prompt(),
                    "combine_prompt": self._get_combine_prompt()
                }
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create QA chain: {str(e)}")

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
            return {"error": f"Failed to process query: {str(e)}"}