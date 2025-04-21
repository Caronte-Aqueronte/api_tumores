
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from neuronal_network.controllers.neuronal_network_controller import router as neuronal_controller

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se permiten todos los origenes
    allow_methods=["*"],  # se permiten todos los metodos
    allow_headers=["*"],  # se permiten todos los headers
)

# incluimos las rutas de la red neuronal
app.include_router(neuronal_controller)


@app.get("/")
async def root():
    return {"message": "Hola mundo"}
