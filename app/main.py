from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.chatbot import Chatbot

app = FastAPI()

# Serve static files (like HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

dir_path = Path(__file__).parent

# Initialize the chatbot
resource_path = dir_path.parent.joinpath("word_files")
chatbot = Chatbot(resource_path)


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return f.read()


@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    answer, relevant_files = chatbot.get_response(question)

    return {"answer": answer, "relevant_files": relevant_files}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
