import uvicorn

if __name__ == "__main__":
    # Start uvicorn server pointing to the app package main.py module
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
