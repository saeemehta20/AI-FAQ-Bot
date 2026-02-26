import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from utils import create_vector_store, search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🏢 Company FAQ Bot (RAG Powered)")

faq_list = []

with open("data/faqs.txt", "r", encoding="utf-8") as file:
    for line in file:
        question, answer = line.strip().split("|")
        faq_list.append(question + " " + answer)

index, embeddings = create_vector_store(faq_list)

user_question = st.text_input("Ask your question")

if user_question:
    context = search(user_question, index, faq_list)

    prompt = f"""
You are an AI assistant for XYZ Technologies.
Answer only based on the context below.

Context:
{context}

Question:
{user_question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.success(response.choices[0].message.content)