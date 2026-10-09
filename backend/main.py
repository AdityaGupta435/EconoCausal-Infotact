from fastapi import FastAPI

app = FastAPI(
    title="EconoCausal API",
    description="Causal inference for dynamic pricing",
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "project": "EconoCausal",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}