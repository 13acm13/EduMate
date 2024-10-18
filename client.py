from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import Textbook_Rag
from agent import run_graph  

app = FastAPI()

origins = ["http://localhost:8000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str

#Endpoint for Rag Tool
@app.post("/rag")
async def agent_endpoint(query: Query):
    try:
        response = Textbook_Rag(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Endpoint for Agent
@app.post("/agent")
async def rag_endpoint(query: Query):
    try:
        response = run_graph(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)