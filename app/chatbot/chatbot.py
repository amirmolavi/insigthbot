__all__ = ["Chatbot"]
import os

import openai
from docx import Document

system_prompt = "You are an assistant that analyzes the content of some word files \
These are general knowledge and onboarding documents. Your job is answer the question from users. If you don't know the answer, just say so."


class Chatbot:
    def __init__(self, word_files_path):
        self.word_files_path = word_files_path
        self.knowledge_base = self.load_word_files()

    def load_word_files(self):
        text = ""
        for filename in os.listdir(self.word_files_path):
            if filename.endswith(".docx"):
                doc = Document(os.path.join(self.word_files_path, filename))

                # Read paragraphs
                for para in doc.paragraphs:
                    text += para.text + "\n"

                # Read tables
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            text += (
                                cell.text + "\t"
                            )  # Use tab to separate cell contents
                        text += "\n"  # New line after each row
        # print("text", text)
        return text

    def get_response(self, question):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{self.knowledge_base}\n\nQuestion: {question}",
                },
            ],
        )
        return response["choices"][0]["message"]["content"]
