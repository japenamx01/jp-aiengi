from fastapi import FastAPI
from app.routers import estimations

app = FastAPI(
    title="CAG Software Estimator API",
    description="API para estimación de proyectos de software utilizando arquitectura Cache Augmented Generation (CAG)",
    version="0.1.0"
)

app.include_router(estimations.router)

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "architecture": "CAG"}

