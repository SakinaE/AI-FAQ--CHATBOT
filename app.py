import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Header
st.markdown(
    """
    <h1 style='text-align:center;'>
        🤖 AI FAQ Chatbot
    </h1>
    <p style='text-align:center;'>
        Ask your questions and get instant answers
    </p>
    """,
    unsafe_allow_html=True
)

# Load FAQ data
faq = pd.read_csv("faq.csv")

questions = faq["Question"]

# Convert text to vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# User Input
user_question = st.text_input(
    "💬 Ask a Question"
)

# Send Button
if st.button("Send 🚀"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:

        user_vector = vectorizer.transform(
            [user_question]
        )

        similarity = cosine_similarity(
            user_vector,
            question_vectors
        )

        best_match = similarity.argmax()

        confidence = similarity.max()

        # User Message
        st.chat_message("user").write(
            user_question
        )

        # Smart Response
        if confidence < 0.30:

            st.chat_message("assistant").write(
                "Sorry, I couldn't find a relevant answer."
            )

        else:

            answer = faq.iloc[
                best_match
            ]["Answer"]

            st.chat_message("assistant").write(
                answer
            )

            st.info(
                f"Confidence Score: {confidence * 100:.2f}%"
            )

# Footer
st.markdown("---")

st.caption(
    "Built using Python, Streamlit, TF-IDF and Cosine Similarity"
)
