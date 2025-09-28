import streamlit as st
from openai import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import io
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None


st.set_page_config(page_title="MS AISHA — AI Student Helper (RAG)", page_icon="📄", layout="centered")

st.title("📄 MS AISHA (Artificially Intelligent Student Helping Assistant)")
# Show title and description with wider styling
st.markdown(
    """
    <div style='text-align: center; width: 100%;'>
        <h1 style='font-size: 3em; margin-bottom: 0.2em;'>📄 Ms AISHA</h1>
        <h2 style='font-size: 2em; font-weight: 400; margin-top: 0;'>Artificially Intelligent Student Helping Assistant</h2>
    </div>
    """,
    unsafe_allow_html=True
)
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

def read_any(file):
    if file is None:
        return ""
    name = (getattr(file, "name", "") or "").lower()
    mime = (getattr(file, "type", "") or "").lower()
    ext = name.split(".")[-1] if "." in name else ""

    def as_text(b):
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return b.decode(errors="ignore")

    # DOCX
    if ext == "docx" or "wordprocessingml.document" in mime:
        if DocxDocument is None:
            return "[Install python-docx to read .docx]"
        buf = io.BytesIO(file.read())
        doc = DocxDocument(buf)
        return "\n".join(p.text for p in doc.paragraphs)

    # PDF
    if ext == "pdf" or "pdf" in mime:
        if PyPDF2 is None:
            return "[Install PyPDF2 to read .pdf]"
        buf = io.BytesIO(file.read())
        reader = PyPDF2.PdfReader(buf)
        pages = []
        for i in range(len(reader.pages)):
            try:
                txt = reader.pages[i].extract_text() or ""
                pages.append(f"[p.{i+1}] {txt}")
            except Exception:
                pages.append(f"[p.{i+1}]")
        return "\n\n".join(pages)

    # Plain text / md fallback
    return as_text(file.read())

if uploaded_homework:
    homework_text = read_any(uploaded_homework) if uploaded_homework else ""
    study_text = ""
    if uploaded_study:
        study_text = read_any(uploaded_study) if uploaded_study else ""

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
    

