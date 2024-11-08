__all__ = ["Chatbot"]
import os

import openai
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

system_prompt = "You are an assistant that analyzes the content of some word files \
These are general knowledge and onboarding documents. Your job is answer the question from users. If you don't know the answer, just say so."


class Chatbot:
    def __init__(self, word_files_path):
        self.word_files_path = word_files_path
        self.knowledge_base = []  # Use a list to store learned information
        self.load_word_files()

    def load_word_files(self):
        for filename in os.listdir(self.word_files_path):
            if filename.endswith(".docx"):
                doc = Document(os.path.join(self.word_files_path, filename))
                self.process_document(doc)

    def process_document(self, doc):
        # Process paragraphs
        for para in doc.paragraphs:
            self.learn(para.text)

        # Process tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    self.learn(cell.text)

    def learn(self, text):
        # Store the text in the knowledge base
        if text:
            self.knowledge_base.append(text)

    def get_response(self, question):
        # Select relevant entries based on the question
        relevant_entries = self.select_relevant_entries(question)

        # Combine the relevant entries into a context string
        context = "\n".join(relevant_entries)

        # Call the OpenAI API with the context and the user's question
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that answers questions based on the provided knowledge."},
                    {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
                ]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def select_relevant_entries(self, question):
        # Use TF-IDF to find the most relevant entries
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(self.knowledge_base + [question])
        cosine_similarities = (vectors * vectors[-1].T).toarray()[:-1, -1]

        # Get the indices of the top relevant entries
        relevant_indices = np.argsort(cosine_similarities)[-5:]  # Adjust the number of entries as needed
        return [self.knowledge_base[i] for i in relevant_indices]
