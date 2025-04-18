
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se permiten todos los origenes
    allow_methods=["*"],  # se permiten todos los metodos
    allow_headers=["*"],  # se permiten todos los headers
)


@app.get("/")
async def root():
    return {"message": "Hola mundo"}
