__all__ = ["Chatbot"]
from dataclasses import dataclass
import os

import openai
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ordered_set import OrderedSet

system_prompt = "You are an assistant that analyzes the content of word files containing how-to guides. \
These are general knowledge and onboarding documents. Your job is to answer questions from users and give them \
itemized instructions whenever applicable. If you don't know the answer, just say so."

class Chatbot:
    def __init__(self, word_files_path):
        self.word_files_path = word_files_path
        self.knowledge_base = []  # Use a list to store learned information (content and filename)
        self.load_word_files()

    def load_word_files(self):
        for filename in os.listdir(self.word_files_path):
            if filename.endswith(".docx"):
                doc = Document(os.path.join(self.word_files_path, filename))
                self.process_document(doc, filename)

    def process_document(self, doc, filename):
        # Process paragraphs
        for para in doc.paragraphs:
            self.learn(para.text, filename)

        # Process tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    self.learn(cell.text, filename)

    def learn(self, text, filename):
        # Store the text and filename in the knowledge base
        if text:
            self.knowledge_base.append((text, filename))

    def get_response(self, question):
        # Select relevant entries based on the question
        relevant_entries, relevant_filenames = self.select_relevant_entries(question)

        # If no relevant entries are found but relevant filenames are present
        if not relevant_entries and relevant_filenames:
            return ("I couldn't find a specific answer, but the following files may contain relevant information:", 
                    relevant_filenames)

        # If no relevant entries or filenames are found
        if not relevant_entries:
            return "I'm here to help! Please ask a specific question.", []

        # Combine the relevant entries into a context string
        context = "\n".join(relevant_entries)

        # Call the OpenAI API with the context and the user's question
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
                ]
            )
            return response["choices"][0]["message"]["content"], relevant_filenames
        except Exception as e:
            return f"An error occurred: {str(e)}", []

    def select_relevant_entries(self, question):
        # Use TF-IDF to find the most relevant entries
        vectorizer = TfidfVectorizer()
        texts = [entry[0] for entry in self.knowledge_base]
        vectors = vectorizer.fit_transform(texts + [question])
        cosine_similarities = (vectors * vectors[-1].T).toarray()[:-1, -1]

        # Get indices of entries with cosine similarity above a certain threshold
        threshold = 0.2  # Set an appropriate threshold for relevance
        relevant_indices = [i for i, score in enumerate(cosine_similarities) if score > threshold]

        # If no entries are above the threshold, return empty lists
        if not relevant_indices:
            return [], []

        # Get top relevant entries by cosine similarity
        top_indices = np.argsort(cosine_similarities)[-20:]  # Adjust the number as needed
        relevant_entries = [self.knowledge_base[i][0] for i in top_indices]
        relevant_filenames = list(OrderedSet([self.knowledge_base[i][1] for i in top_indices]))

        return relevant_entries, relevant_filenames
