from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "API lista para Checkpoint 2"}