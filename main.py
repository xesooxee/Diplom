import os

from dotenv import load_dotenv

load_dotenv()

from app import app  # noqa: E402 — load_dotenv must run first

if __name__ == "__main__":
    import uvicorn

    reload = os.getenv("ENV", "production") == "development"
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=reload)
