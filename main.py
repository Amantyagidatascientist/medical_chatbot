from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import MedicalChatbot
import uvicorn
import time


app = FastAPI(title="Medical Chatbot API")

chatbot = MedicalChatbot()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
async def read_root():
    return {"message": "Medical Chatbot API - Visit /docs for Swagger UI"}

@app.post("/query",response_model=QueryResponse)
async def process_query(query:QueryRequest):
    try:
        result=chatbot.query(query.question)
        if "error" in result:
            raise HTTPException(status_code=500,detail=result["error"])
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    

@app.get("/health")
async def health_check():
    return {"status":"healthy"}

if __name__=="__main__":
        
    time.sleep(5)
    uvicorn.run(app,host="0.0.0.0",port=8001)
    
    