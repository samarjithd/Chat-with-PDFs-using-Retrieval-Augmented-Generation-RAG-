# # import streamlit as st

# # st.title("Fill in the Blanks Page")
# # st.write("You have been redirected successfully.")

# # if st.button("Go Back"):
# #     st.switch_page("app")  # No '.py'




# import streamlit as st
# from langchain.schema import HumanMessage
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from io import BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# import hashlib

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Initialize session state
# if "questions_json" not in st.session_state:
#     st.session_state.questions_json = []

# if "user_answers" not in st.session_state:
#     st.session_state.user_answers = {}

# # Function to extract text from PDFs
# def get_pdf_text(pdf_docs):
#     """Extract text from uploaded PDF files."""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to split text into chunks
# def get_text_chunks(text):
#     """Split text into manageable chunks."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
#     return text_splitter.split_text(text)

# # Function to generate Fill-in-the-Blanks questions
# def generate_fill_in_blank_questions(text_chunks, num_questions):
#     """Generate fill-in-the-blank questions."""
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)

#     prompt_template = f"""
#     Based on the following text, generate exactly  
#     {num_questions} unique different fill-in-the-blank
#     questions.
#     Each question must follow this format no other
#     format should be used:
    
#     question: <sentence with a blank>
#     answer: <correct word>
    
#     Strictly give in the above mentioned format
#     otherwise json parsing might be incorrect.
#     Example:
#     question: "The capital of France is _____."
#     answer: "Paris"
    
#     Context:
#     {{context}}

#     Questions:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#     context = " ".join(text_chunks[:3])  # Limit context size
#     formatted_prompt = prompt.format(context=context)

#     input_message = HumanMessage(content=formatted_prompt)
#     response = model.invoke([input_message])
#     print("gemini")
#     print(response.content)
#     return response.content if hasattr(response, "content") else response

# # Function to parse generated questions into JSON
# # def load_fill_in_blank_questions(questions_text):
# #     """Convert questions text into JSON format."""
# #     questions_json = []
# #     question_blocks = questions_text.strip().split("\n\n")

# #     for block in question_blocks:
# #         question_text = None
# #         correct_answer = None

# #         lines = block.strip().split("\n")
# #         for line in lines:
# #             if line.startswith("question:"):
# #                 question_text = line.replace("question:", "").strip()
# #             elif line.startswith("answer:"):
# #                 correct_answer = line.replace("answer:", "").strip()

# #             if question_text and correct_answer:
# #                  questions_json.append({
# #                 "type": "fill",
# #                 "question": question_text,
# #                 "answer": correct_answer
# #             })
# #             else:
# #               st.warning(f"Skipping invalid question block:\n{block}")
   
# #     return questions_json

# def load_fill_in_blank_questions(questions_text):
#     """Convert questions text into JSON format."""
#     questions_json = []
#     question_blocks = questions_text.strip().split("\n\n")

#     for block in question_blocks:
#         question_text = None
#         correct_answer = None

#         lines = block.strip().split("\n")
#         for line in lines:
#             if line.startswith("question:"):
#                 question_text = line.replace("question:", "").strip()
#             elif line.startswith("answer:"):
#                 correct_answer = line.replace("answer:", "").strip()

#             # Ensure both question and answer are set before appending
#             if question_text and correct_answer:
#                 questions_json.append({
#                     "type": "fill",
#                     "question": question_text,
#                     "answer": correct_answer
#                 })
#                 # Reset values to prevent duplicate entries in the same block
#                 question_text = None
#                 correct_answer = None

#     return questions_json


# # Function to generate downloadable PDF
# def generate_pdf(questions_json):
#     """Generate a PDF containing the fill-in-the-blank quiz and answer key."""
#     buffer = BytesIO()
#     pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
#     width, height = letter
#     y_position = height - 50  # Start below the top margin

#     # Write Quiz Questions
#     pdf_canvas.setFont("Helvetica-Bold", 14)
#     pdf_canvas.drawString(50, y_position, "Fill-in-the-Blank Quiz")
#     pdf_canvas.setFont("Helvetica", 10)
#     y_position -= 30

#     for i, question in enumerate(questions_json):
#         if y_position < 50:  # Add a new page if space is insufficient
#             pdf_canvas.showPage()
#             y_position = height - 50

#         pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['question']}")
#         y_position -= 30

#     # Add Answer Key
#     pdf_canvas.showPage()
#     y_position = height - 50
#     pdf_canvas.setFont("Helvetica-Bold", 14)
#     pdf_canvas.drawString(50, y_position, "Answer Key")
#     pdf_canvas.setFont("Helvetica", 10)
#     y_position -= 30

#     for i, question in enumerate(questions_json):
#         pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['answer']}")
#         y_position -= 20

#     pdf_canvas.save()
#     buffer.seek(0)
#     return buffer

# # Function to conduct the fill-in-the-blank quiz
# def conduct_fill_in_blank_quiz():
#     """Display the quiz and collect user answers."""
#     user_answers = {}
#     for i, q in enumerate(st.session_state.questions_json):
         
#              st.subheader(f"Question {i+1}")
#              st.write(q["question"])
#              user_answers[i] = st.text_input("Your answer:", key=f"q{i}")

#     if st.button("Submit Quiz"):
#         st.session_state.user_answers = user_answers
#         st.session_state.quiz_submitted = True  # Mark quiz as submitted
#         # st.experimental_rerun()
#         st.rerun()

# # Function to calculate quiz score
# def calculate_fill_in_blank_score():
#     """Calculate the quiz score and display correct answers."""
#     if not st.session_state.user_answers:
#         st.warning("No answers submitted yet.")
#         return

#     score = 0
#     total = len(st.session_state.questions_json)

#     for i, q in enumerate(st.session_state.questions_json):
#         correct_answer = q["answer"].strip().lower()
#         user_answer = st.session_state.user_answers.get(i, "").strip().lower()

#         if user_answer == correct_answer:
#             score += 1

#     st.success(f"You scored {score}/{total}!")

#     st.write("### Correct Answers:")
#     for i, q in enumerate(st.session_state.questions_json):
#         st.write(f"- **{q['question']}**: {q['answer']}")

# # Main Streamlit Application
# def main():
#     st.title("Fill-in-the-Blank Quiz Generator")
#     st.sidebar.title("Menu")
#     option = st.sidebar.radio("Choose an option", ["Generate Questions", "Take Quiz"])

#     # Generate Questions
#     if option == "Generate Questions":
#         pdf_docs = st.sidebar.file_uploader(
#             "Upload PDF Files", accept_multiple_files=True
#         )
#         num_questions = st.sidebar.number_input("Number of questions:", min_value=1, max_value=50, value=5, step=1)

#         if st.sidebar.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     questions_text = generate_fill_in_blank_questions(text_chunks, num_questions)

#                     if questions_text:
#                         st.session_state.questions_json = load_fill_in_blank_questions(questions_text)
#                         st.success("Questions Generated Successfully!")
                       
#                         # Generate and offer PDF for download
#                         pdf_buffer = generate_pdf(st.session_state.questions_json)
#                         st.download_button(
#                             label="Download Quiz as PDF",
#                             data=pdf_buffer,
#                             file_name="fill_in_the_blank_quiz.pdf",
#                             mime="application/pdf"
#                         )
#                     else:
#                         st.error("No valid questions generated. Please try again.")
#             else:
#                 st.error("Please upload at least one PDF file.")

#     # Take Quiz
#     if option == "Take Quiz":
#         if st.session_state.questions_json:
#             st.write("### Quiz")
#             conduct_fill_in_blank_quiz()
#             if "quiz_submitted" in st.session_state and st.session_state.quiz_submitted:
#                 calculate_fill_in_blank_score()
#         else:
#             st.warning("No questions available. Please generate them first.")

# if __name__ == "__main__":
#     main()
   
import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import requests  # 👈 NEW: for Groq API calls

# ----------------- ENV -----------------
load_dotenv()  # loads GROQ_API_KEY, etc.


# ----------------- SESSION STATE -----------------
if "questions_json" not in st.session_state:
    st.session_state.questions_json = []

if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}


# ----------------- GROQ HELPER -----------------
def call_groq(prompt: str) -> str:
    """
    Call Groq's chat completion API with a simple user prompt.
    Uses LLaMA-3.1-8B-Instant (fast + decent quality).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that generates fill-in-the-blank questions in a strict format.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    res = requests.post(url, json=payload, headers=headers)
    res.raise_for_status()
    data = res.json()
    return data["choices"][0]["message"]["content"]


# ----------------- PDF HELPERS -----------------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=500
    )
    return text_splitter.split_text(text)


# ----------------- QUESTION GENERATION -----------------
def generate_fill_in_blank_questions(text_chunks, num_questions):
    """Generate fill-in-the-blank questions using Groq."""
    # Use first few chunks as context to avoid crazy long prompts
    context = " ".join(text_chunks[:3])

    prompt = f"""
Based on the following text, generate EXACTLY {num_questions} unique fill-in-the-blank questions.

STRICT FORMAT (no extra text, no numbering, no bullets, no JSON):

question: <sentence with a blank using _____>
answer: <single correct word or short phrase>

Repeat this block {num_questions} times, separated by a blank line.

Example:
question: The capital of France is _____.
answer: Paris

Now use this context:

{context}

Questions:
"""

    response_text = call_groq(prompt)
    print("GROQ RESPONSE:\n", response_text)
    return response_text


def load_fill_in_blank_questions(questions_text):
    """Convert questions text into JSON format."""
    questions_json = []
    question_blocks = questions_text.strip().split("\n\n")

    for block in question_blocks:
        question_text = None
        correct_answer = None

        lines = block.strip().split("\n")
        for line in lines:
            if line.startswith("question:"):
                question_text = line.replace("question:", "").strip().strip('"')
            elif line.startswith("answer:"):
                correct_answer = line.replace("answer:", "").strip().strip('"')

            # Ensure both question and answer are set before appending
            if question_text and correct_answer:
                questions_json.append({
                    "type": "fill",
                    "question": question_text,
                    "answer": correct_answer
                })
                # Reset to avoid duplicate appends
                question_text = None
                correct_answer = None

    return questions_json


# ----------------- PDF GENERATION -----------------
def generate_pdf(questions_json):
    """Generate a PDF containing the fill-in-the-blank quiz and answer key."""
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y_position = height - 50  # Start below the top margin

    # Write Quiz Questions
    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(50, y_position, "Fill-in-the-Blank Quiz")
    pdf_canvas.setFont("Helvetica", 10)
    y_position -= 30

    for i, question in enumerate(questions_json):
        if y_position < 50:  # Add a new page if space is insufficient
            pdf_canvas.showPage()
            y_position = height - 50

        pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['question']}")
        y_position -= 30

    # Add Answer Key
    pdf_canvas.showPage()
    y_position = height - 50
    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(50, y_position, "Answer Key")
    pdf_canvas.setFont("Helvetica", 10)
    y_position -= 30

    for i, question in enumerate(questions_json):
        pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['answer']}")
        y_position -= 20

    pdf_canvas.save()
    buffer.seek(0)
    return buffer


# ----------------- QUIZ FLOW -----------------
def conduct_fill_in_blank_quiz():
    """Display the quiz and collect user answers."""
    user_answers = {}
    for i, q in enumerate(st.session_state.questions_json):
        st.subheader(f"Question {i + 1}")
        st.write(q["question"])
        user_answers[i] = st.text_input("Your answer:", key=f"q{i}")

    if st.button("Submit Quiz"):
        st.session_state.user_answers = user_answers
        st.session_state.quiz_submitted = True
        st.rerun()


def calculate_fill_in_blank_score():
    """Calculate the quiz score and display correct answers."""
    if not st.session_state.user_answers:
        st.warning("No answers submitted yet.")
        return

    score = 0
    total = len(st.session_state.questions_json)

    for i, q in enumerate(st.session_state.questions_json):
        correct_answer = q["answer"].strip().lower()
        user_answer = st.session_state.user_answers.get(i, "").strip().lower()

        if user_answer == correct_answer:
            score += 1

    st.success(f"You scored {score}/{total}!")

    st.write("### Correct Answers:")
    for i, q in enumerate(st.session_state.questions_json):
        st.write(f"- **{q['question']}**: {q['answer']}")


# ----------------- MAIN APP -----------------
def main():
    st.title("Fill-in-the-Blank Quiz Generator")
    st.sidebar.title("Menu")
    option = st.sidebar.radio("Choose an option", ["Generate Questions", "Take Quiz"])

    # Generate Questions
    if option == "Generate Questions":
        pdf_docs = st.sidebar.file_uploader(
            "Upload PDF Files", accept_multiple_files=True
        )
        num_questions = st.sidebar.number_input(
            "Number of questions:", min_value=1, max_value=50, value=5, step=1
        )

        if st.sidebar.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    questions_text = generate_fill_in_blank_questions(
                        text_chunks, num_questions
                    )

                    if questions_text:
                        st.session_state.questions_json = load_fill_in_blank_questions(
                            questions_text
                        )
                        if st.session_state.questions_json:
                            st.success("Questions Generated Successfully!")

                            # Generate and offer PDF for download
                            pdf_buffer = generate_pdf(st.session_state.questions_json)
                            st.download_button(
                                label="Download Quiz as PDF",
                                data=pdf_buffer,
                                file_name="fill_in_the_blank_quiz.pdf",
                                mime="application/pdf",
                            )
                        else:
                            st.error(
                                "Questions were generated but parsing failed. Check format."
                            )
                    else:
                        st.error("No valid questions generated. Please try again.")
            else:
                st.error("Please upload at least one PDF file.")

    # Take Quiz
    if option == "Take Quiz":
        if st.session_state.questions_json:
            st.write("### Quiz")
            conduct_fill_in_blank_quiz()
            if (
                "quiz_submitted" in st.session_state
                and st.session_state.quiz_submitted
            ):
                calculate_fill_in_blank_score()
        else:
            st.warning("No questions available. Please generate them first.")


if __name__ == "__main__":
    main()
