from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import torch
from dotenv import load_dotenv
import os


load_dotenv()
hf_token = os.getenv("HF_TOKEN")  


model_name="./local_embedding_model"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


if not os.path.exists("./local_text_generator_model"):
    print("Saving flan-t5-small locally...")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    model.save_pretrained("./local_text_generator_model")
    tokenizer.save_pretrained("./local_text_generator_model")
    print("✅ flan-t5-small saved at ./local_text_generator_model")


def load_llm():
    try:
        # Local model configuration
        model_path = "./local_text_generator_model"  # Correct model for seq2seq
        
        # Load model and tokenizer with the correct class
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        
        # Create local pipeline
        pipe = pipeline(
            "text2text-generation",  # Changed from text-generation
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    except Exception as e:
        print(f"Failed to load local LLM: {str(e)}")

# For question prompt (used per chunk)
QUESTION_PROMPT_TEMPLATE = """
You are a board-certified anatomist with expertise from Gray’s Anatomy.
Answer only using the context provided below. If the answer is not in the context, say "I don't know."

<context>
{context}
</context>

Question: {question}
Answer:
"""

# For combine prompt (used to merge chunk answers)
COMBINE_PROMPT_TEMPLATE = """
You are a board-certified anatomist with expertise from Gray’s Anatomy.
Use the following summarized answers from multiple sources to construct a final response. 
If the answer is not in the summaries, say "I don't know."

<summaries>
{summaries}
</summaries>

Question: {question}
Answer:
"""


def get_combine_prompt():
    return PromptTemplate(
        template=COMBINE_PROMPT_TEMPLATE,
        input_variables=["summaries", "question"]
    )


def get_question_prompt():
    return PromptTemplate(
        template=QUESTION_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )




def load_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="./local_embedding_model",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
    except Exception as e:
        print(f"Failed to load embeddings: {str(e)}")
        exit(1)

def load_vectorstore(embeddings):
    try:
        return FAISS.load_local(
            "vectorstore/db_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Failed to load vector store: {str(e)}")

def main():
    # Initialize components with error handling
    llm = load_llm()
    embeddings = load_embeddings()
    db = load_vectorstore(embeddings)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=db.as_retriever(search_kwargs={'k': 5, 'search_type': 'mmr'}),
        return_source_documents=True,
        chain_type_kwargs={
        "question_prompt": get_question_prompt(),
        "combine_prompt": get_combine_prompt()
    } 
    )
    
    # Process queries
    while True:
        try:
            user_query = input("\nWrite Query Here (or 'quit' to exit): ")
            if user_query.lower() == 'quit':
                break
                
            response = qa_chain.invoke({'query': user_query})
            print("\nRESULT:", response["result"])
            print("\nSOURCE DOCUMENTS:")
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"{i}. {doc.page_content[:200]}...")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}")

if __name__ == "__main__":
    main()