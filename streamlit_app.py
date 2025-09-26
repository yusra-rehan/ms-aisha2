import streamlit as st
from openai import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import io
from docx import Document as DocxDocument

# Show title and description.
st.title("📄 MS AISHA (AI - Aritifically Intelligent Student Helping Assitant)")
st.write(
    "Please start by uploading your homework and any reference materials you have. Then, ask a question about your homework. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else st.text_input("Enter your OpenAI API key:", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.

    st.header("Step 1: Upload Homework and Study Material")
    uploaded_homework = st.file_uploader(
        "Upload your homework (required)", type=("txt", "md", "pdf", "docx"), key="homework"
    )
    uploaded_study = st.file_uploader(
        "Upload your study/reference material (optional)", type=("txt", "md", "pdf", "docx"), key="study"
    )

    if uploaded_homework:
        def read_file(file, filetype):
            if filetype == "docx":
                doc = DocxDocument(io.BytesIO(file.read()))
                return "\n".join([para.text for para in doc.paragraphs])
            else:
                return file.read().decode(errors="ignore")

        homework_text = read_file(uploaded_homework, uploaded_homework.type.split("/")[-1] if hasattr(uploaded_homework, 'type') else uploaded_homework.name.split('.')[-1])
        study_text = ""
        if uploaded_study:
            study_text = read_file(uploaded_study, uploaded_study.type.split("/")[-1] if hasattr(uploaded_study, 'type') else uploaded_study.name.split('.')[-1])

        st.header("Step 2: Let's Work on Your Homework!")
        st.write("This AI tutor will help you solve your homework by giving hints and teaching you concepts. It will not give you the direct answer, but will guide you until you understand. When you're ready, submit your answer for feedback!")

        if 'tutor_history' not in st.session_state:
            st.session_state.tutor_history = []
        if 'student_history' not in st.session_state:
            st.session_state.student_history = []
        if 'awaiting_answer' not in st.session_state:
            st.session_state.awaiting_answer = False
        if 'last_hint' not in st.session_state:
            st.session_state.last_hint = ""

        # Step 2a: Tutor gives a hint or asks a guiding question
        if not st.session_state.awaiting_answer:
            prompt = f"You are a friendly, encouraging middle school tutor. Your job is to help the student solve their homework by giving hints, asking guiding questions, and teaching concepts. Never give the direct answer. Use the following study material as reference if available. After each hint, ask the student to try the next step or explain their thinking. If the student seems stuck, break the problem down further.\n\nHomework:\n{homework_text}\n\nStudy Material:\n{study_text}\n\nStart by giving a hint or asking a question to help the student begin."
            messages = [
                {"role": "system", "content": "You are a middle school tutor. Never give direct answers. Only give hints, explanations, and ask questions to help the student learn. Always check if the student understands before moving on."},
                {"role": "user", "content": prompt}
            ]
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True,
            )
            st.session_state.last_hint = ""
            for chunk in stream:
                st.session_state.last_hint += chunk.choices[0].delta.content or ""
            st.markdown(st.session_state.last_hint)
            st.session_state.awaiting_answer = True
            st.session_state.tutor_history.append(st.session_state.last_hint)

        # Step 2b: Student responds
        if st.session_state.awaiting_answer:
            student_input = st.text_area("Your turn! Type your answer, thought process, or next step:", key="student_input")
            if st.button("Submit Answer/Step"):
                st.session_state.student_history.append(student_input)
                # Tutor evaluates student's response and gives next hint or feedback
                tutor_followup = f"The student responded: {student_input}\n\nHomework: {homework_text}\n\nStudy Material: {study_text}\n\nAs a tutor, do NOT give the answer. Instead, evaluate if the student is on the right track, give feedback, and provide the next hint or question. If the student is correct, encourage them and ask for the next step. If they are done, ask them to submit their final answers for review. If they are stuck, break it down further."
                messages = [
                    {"role": "system", "content": "You are a middle school tutor. Never give direct answers. Only give hints, explanations, and ask questions to help the student learn. Always check if the student understands before moving on."},
                    {"role": "user", "content": tutor_followup}
                ]
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=True,
                )
                tutor_reply = ""
                for chunk in stream:
                    tutor_reply += chunk.choices[0].delta.content or ""
                st.markdown(tutor_reply)
                st.session_state.tutor_history.append(tutor_reply)
                st.session_state.awaiting_answer = True

            st.write("\nWhen you are finished with your homework, paste your final answers below and click 'Submit Final Answers' for review.")
            final_answers = st.text_area("Paste your final answers here:", key="final_answers")
            if st.button("Submit Final Answers") and final_answers.strip():
                # Tutor reviews the final answers
                review_prompt = f"The student has submitted their final answers: {final_answers}\n\nHomework: {homework_text}\n\nStudy Material: {study_text}\n\nAs a tutor, review the answers. Give feedback, corrections, and encouragement. If the answers are correct, praise the student. If not, give hints for improvement. Do NOT give the direct answer."
                messages = [
                    {"role": "system", "content": "You are a middle school tutor. Never give direct answers. Only give hints, explanations, and ask questions to help the student learn. Always check if the student understands before moving on."},
                    {"role": "user", "content": review_prompt}
                ]
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=True,
                )
                review_reply = ""
                for chunk in stream:
                    review_reply += chunk.choices[0].delta.content or ""
                st.markdown(review_reply)
                st.session_state.tutor_history.append(review_reply)
                st.session_state.awaiting_answer = False
