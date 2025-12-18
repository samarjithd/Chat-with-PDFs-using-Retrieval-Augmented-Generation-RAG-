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
# import random

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Step 1: Functions to process PDFs

# def hash_file(file):
#     """Generate a hash for the uploaded file."""
#     hasher = hashlib.md5()
#     for chunk in iter(lambda: file.read(4096), b""):
#         hasher.update(chunk)
#     file.seek(0)  # Reset file pointer after hashing
#     return hasher.hexdigest()

# def get_pdf_text(pdf_docs):
#     """Extract text from uploaded PDF files."""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     """Split text into manageable chunks."""
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Step 2: Generate questions

# def generate_questions(text_chunks, num_questions):
#     """Generate the specified number of questions."""
#     model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)

#     prompt_template = f"""
#     Based on the following context , generate{num_questions} multiple-choice questions and
#     the options should not be too large.
#     Each question must strictly follow this format
#     otherwise the parsing of the questions will be
#     difficult:
#     question: <question_text>
#     option: <option1>
#     option: <option2>
#     option: <option3>
#     option: <option4>
#     answer: <correct_option>

#     Context:
#     {{context}}

#     Questions:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
#     context = " ".join(text_chunks[:3])  # Limit context for the prompt
#     formatted_prompt = prompt.format(context=context)

#     input_message = HumanMessage(content=formatted_prompt)
#     response = model([input_message])
#     return response.content if hasattr(response, "content") else response

# # Step 3: Parse questions into JSON

# # def load_questions_from_text(questions_text):
# #     """Generate questions JSON from structured text."""
# #     questions_json = []
# #     question_blocks = questions_text.strip().split("\n\n")

# #     for block in question_blocks:
# #         question_text = None
# #         options = []
# #         correct_answer = None

# #         lines = block.strip().split("\n")
# #         for line in lines:
# #             if line.startswith("question:"):
# #                 question_text = line.replace("question:", "").strip()
# #             elif line.startswith("option:"):
# #                 options.append(line.replace("option:", "").strip())
# #             elif line.startswith("answer:"):
# #                 correct_answer = line.replace("answer:", "").strip()

# #         # Validate and append to JSON
# #         if question_text and correct_answer and len(options) == 4:
# #             random.shuffle(options)  # Shuffle options
# #             questions_json.append({
# #                 "type": "mcq",
# #                 "question": question_text,
# #                 "options": options,
# #                 "answer": correct_answer
# #             })
# #         else:
# #             st.warning(f"Skipping invalid question block:\n{block}")
    
# #     return questions_json


#  # Only if using Streamlit for warnings

# def load_questions_from_text(questions_text):
#     """Generate questions JSON from structured text."""
#     questions_json = []
#     question_blocks = questions_text.strip().split("\n\n")

#     for block in question_blocks:
#         question_text = None
#         options = []
#         correct_answer = None

#         lines = block.strip().split("\n")
#         for line in lines:
#             if line.startswith("question:") and question_text is None:
#                 question_text = line.replace("question:", "").strip()
#             elif line.startswith("option:"):
#                 option_text = line.replace("option:", "").strip()
#                 if option_text not in options:  # Avoid duplicate options
#                     options.append(option_text)
#             elif line.startswith("answer:") and correct_answer is None:
#                 correct_answer = line.replace("answer:", "").strip()

#         # Validate and append to JSON
#         if question_text and correct_answer and len(options) == 4:
#             random.shuffle(options)  # Shuffle options to randomize order
#             questions_json.append({
#                 "type": "mcq",
#                 "question": question_text,
#                 "options": options,
#                 "answer": correct_answer
#             })
#         else:
#             st.warning(f"Skipping invalid question block:\n{block}")

#     return questions_json


# # Step 4: Generate downloadable PDF

# def generate_pdf(questions_json):
#     """Generate a PDF containing the quiz questions and the answer key."""
#     buffer = BytesIO()
#     pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
#     width, height = letter
#     y_position = height - 50  # Start below the top margin

#     # Write Quiz Questions
#     pdf_canvas.setFont("Helvetica-Bold", 14)
#     pdf_canvas.drawString(50, y_position, "Quiz Questions")
#     pdf_canvas.setFont("Helvetica", 10)
#     y_position -= 30

#     for i, question in enumerate(questions_json):
#         if y_position < 50:  # Add a new page if space is insufficient
#             pdf_canvas.showPage()
#             y_position = height - 50

#         pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['question']}")
#         y_position -= 15
#         for option in question['options']:
#             pdf_canvas.drawString(70, y_position, f"- {option}")
#             y_position -= 15
#         y_position -= 10

#     # Add Answer Key
#     if y_position < 100:
#         pdf_canvas.showPage()
#         y_position = height - 50

#     pdf_canvas.setFont("Helvetica-Bold", 14)
#     pdf_canvas.drawString(50, y_position, "Answer Key")
#     pdf_canvas.setFont("Helvetica", 10)
#     y_position -= 30

#     for i, question in enumerate(questions_json):
#         if y_position < 50:  # Add a new page if space is insufficient
#             pdf_canvas.showPage()
#             y_position = height - 50

#         pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['answer']}")
#         y_position -= 15

#     pdf_canvas.save()
#     buffer.seek(0)
#     return buffer

# # Step 5: Conduct quiz

# def conduct_quiz(questions):
#     """Display the quiz and collect user answers."""
#     user_answers = {}
#     for i, q in enumerate(questions):
#         st.subheader(f"Question {i+1}")
#         st.write(q["question"])
#         if q["type"] == "mcq":
#             user_answers[i] = st.radio(
#                 "Choose an answer:", q["options"], key=f"q{i}"
#             )
#         elif q["type"] == "fill":
#             user_answers[i] = st.text_input("Your answer:", key=f"q{i}")
#     return user_answers

# def calculate_score(questions, user_answers):
#     """Calculate the quiz score and return correct answers."""
#     score = 0
#     correct_answers = []

#     for i, q in enumerate(questions):
#         correct_answer = q["answer"]
#         if user_answers.get(i) and user_answers[i].strip().lower() == correct_answer.strip().lower():
#             score += 1
#         correct_answers.append((q["question"], correct_answer))
    
#     return score, correct_answers

# # Step 6: Main Streamlit application

# def main():
#     st.title("Instant Quiz Generator")
#     st.sidebar.title("Menu")
#     option = st.sidebar.radio("Choose an option", ["Generate Questions", "Take Quiz"])

#     # Generate Questions
#     if option == "Generate Questions":
#         pdf_docs = st.sidebar.file_uploader(
#             "Upload your PDF Files and Click on the Submit & Process Button",
#             accept_multiple_files=True
#         )
#         num_questions = st.sidebar.number_input("Number of questions:", min_value=1, max_value=50, value=5, step=1)

#         if st.sidebar.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     questions_text = generate_questions(text_chunks, num_questions)
#                     print(questions_text)

#                     if questions_text:
#                         questions_json = load_questions_from_text(questions_text)
#                         st.session_state.questions_json = questions_json
#                         st.success("Questions Generated Successfully!")

#                         # Generate and offer PDF for download
#                         pdf_buffer = generate_pdf(questions_json)
#                         st.download_button(
#                             label="Download Questions and Answer Key as PDF",
#                             data=pdf_buffer,
#                             file_name="quiz_with_answers.pdf",
#                             mime="application/pdf"
#                         )
#                     else:
#                         st.error("No valid questions generated. Please try again.")
#             else:
#                 st.error("Please upload at least one PDF file.")

#     # Take Quiz
#     if option == "Take Quiz":
#         if "questions_json" in st.session_state:
#             st.write("### Quiz")
#             user_answers = conduct_quiz(st.session_state.questions_json)

#             if st.button("Submit Quiz"):
#                 score, correct_answers = calculate_score(st.session_state.questions_json, user_answers)
#                 st.success(f"You scored {score}/{len(st.session_state.questions_json)}!")
#                 st.write("### Correct Answers:")
#                 for q, correct in correct_answers:
#                     st.write(f"- **{q}**: {correct}")
#         else:
#             st.info("Generate questions first to take the quiz.")


 
# if __name__ == "__main__":
#     main()



import os
import random
import hashlib
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import requests  # 👈 NEW: for Groq API calls


# Load environment variables
load_dotenv()  # loads GROQ_API_KEY, etc.


# ----------------- Groq helper -----------------

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
                "content": (
                    "You are an assistant that generates multiple-choice questions "
                    "in a very strict text format so they can be parsed by code."
                ),
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


# ----------------- Step 1: Functions to process PDFs -----------------

def hash_file(file):
    """Generate a hash for the uploaded file."""
    hasher = hashlib.md5()
    for chunk in iter(lambda: file.read(4096), b""):
        hasher.update(chunk)
    file.seek(0)  # Reset file pointer after hashing
    return hasher.hexdigest()


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# ----------------- Step 2: Generate questions -----------------

def generate_questions(text_chunks, num_questions):
    """Generate the specified number of questions using Groq."""
    # Limit context size to avoid insane prompts
    context = " ".join(text_chunks[:3])

    prompt = f"""
Based on the following context, generate EXACTLY {num_questions} multiple-choice questions.
The options should NOT be very long.

You MUST STRICTLY follow this format for each question (NO extra text, NO numbering, NO bullet points):

question: <question_text>
option: <option1>
option: <option2>
option: <option3>
option: <option4>
answer: <exact_correct_option_text>

Rules:
- 'question:' and 'option:' and 'answer:' must start at the beginning of the line.
- There must be EXACTLY 4 'option:' lines per question.
- The 'answer:' line must contain EXACTLY one of the option texts (copy-paste).
- Separate each question block by a blank line.
- Do NOT add explanations, JSON, or any other text.

Context:
{context}

Questions:
"""

    questions_text = call_groq(prompt)
    return questions_text


# ----------------- Step 3: Parse questions into JSON -----------------

def load_questions_from_text(questions_text):
    """Generate questions JSON from structured text."""
    questions_json = []
    question_blocks = questions_text.strip().split("\n\n")

    for block in question_blocks:
        question_text = None
        options = []
        correct_answer = None

        lines = block.strip().split("\n")
        for line in lines:
            if line.startswith("question:") and question_text is None:
                question_text = line.replace("question:", "").strip()
            elif line.startswith("option:"):
                option_text = line.replace("option:", "").strip()
                if option_text not in options:  # Avoid duplicate options
                    options.append(option_text)
            elif line.startswith("answer:") and correct_answer is None:
                correct_answer = line.replace("answer:", "").strip()

        # Validate and append to JSON
        if question_text and correct_answer and len(options) == 4:
            random.shuffle(options)  # Shuffle options to randomize order
            questions_json.append({
                "type": "mcq",
                "question": question_text,
                "options": options,
                "answer": correct_answer
            })
        else:
            st.warning(f"Skipping invalid question block:\n{block}")

    return questions_json


# ----------------- Step 4: Generate downloadable PDF -----------------

def generate_pdf(questions_json):
    """Generate a PDF containing the quiz questions and the answer key."""
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y_position = height - 50  # Start below the top margin

    # Write Quiz Questions
    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(50, y_position, "Quiz Questions")
    pdf_canvas.setFont("Helvetica", 10)
    y_position -= 30

    for i, question in enumerate(questions_json):
        if y_position < 50:  # Add a new page if space is insufficient
            pdf_canvas.showPage()
            y_position = height - 50

        pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['question']}")
        y_position -= 15
        for option in question['options']:
            pdf_canvas.drawString(70, y_position, f"- {option}")
            y_position -= 15
        y_position -= 10

    # Add Answer Key
    if y_position < 100:
        pdf_canvas.showPage()
        y_position = height - 50

    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(50, y_position, "Answer Key")
    pdf_canvas.setFont("Helvetica", 10)
    y_position -= 30

    for i, question in enumerate(questions_json):
        if y_position < 50:  # Add a new page if space is insufficient
            pdf_canvas.showPage()
            y_position = height - 50

        pdf_canvas.drawString(50, y_position, f"Q{i + 1}: {question['answer']}")
        y_position -= 15

    pdf_canvas.save()
    buffer.seek(0)
    return buffer


# ----------------- Step 5: Conduct quiz -----------------

def conduct_quiz(questions):
    """Display the quiz and collect user answers."""
    user_answers = {}
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}")
        st.write(q["question"])
        if q["type"] == "mcq":
            user_answers[i] = st.radio(
                "Choose an answer:", q["options"], key=f"q{i}"
            )
        elif q["type"] == "fill":
            user_answers[i] = st.text_input("Your answer:", key=f"q{i}")
    return user_answers


def calculate_score(questions, user_answers):
    """Calculate the quiz score and return correct answers."""
    score = 0
    correct_answers = []

    for i, q in enumerate(questions):
        correct_answer = q["answer"]
        if user_answers.get(i) and user_answers[i].strip().lower() == correct_answer.strip().lower():
            score += 1
        correct_answers.append((q["question"], correct_answer))

    return score, correct_answers


# ----------------- Step 6: Main Streamlit application -----------------

def main():
    st.title("Instant Quiz Generator")
    st.sidebar.title("Menu")
    option = st.sidebar.radio("Choose an option", ["Generate Questions", "Take Quiz"])

    # Generate Questions
    if option == "Generate Questions":
        pdf_docs = st.sidebar.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        num_questions = st.sidebar.number_input(
            "Number of questions:", min_value=1, max_value=50, value=5, step=1
        )

        if st.sidebar.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    questions_text = generate_questions(text_chunks, num_questions)
                    print(questions_text)

                    if questions_text:
                        questions_json = load_questions_from_text(questions_text)
                        st.session_state.questions_json = questions_json
                        st.success("Questions Generated Successfully!")

                        # Generate and offer PDF for download
                        pdf_buffer = generate_pdf(questions_json)
                        st.download_button(
                            label="Download Questions and Answer Key as PDF",
                            data=pdf_buffer,
                            file_name="quiz_with_answers.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("No valid questions generated. Please try again.")
            else:
                st.error("Please upload at least one PDF file.")

    # Take Quiz
    if option == "Take Quiz":
        if "questions_json" in st.session_state and st.session_state.questions_json:
            st.write("### Quiz")
            user_answers = conduct_quiz(st.session_state.questions_json)

            if st.button("Submit Quiz"):
                score, correct_answers = calculate_score(
                    st.session_state.questions_json, user_answers
                )
                st.success(f"You scored {score}/{len(st.session_state.questions_json)}!")
                st.write("### Correct Answers:")
                for q, correct in correct_answers:
                    st.write(f"- **{q}**: {correct}")
        else:
            st.info("Generate questions first to take the quiz.")


if __name__ == "__main__":
    main()
