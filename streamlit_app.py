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




st.set_page_config(page_title="MS AISHA — AI Student Helper", page_icon="📄", layout="centered")

st.title("📄 MS AISHA (Artificially Intelligent Student Helping Assistant)")
st.write(
    "Upload your homework and any reference materials, then ask a question. "
    "Add your OpenAI API key in the sidebar (or via `st.secrets['OPENAI_API_KEY']`). "
    "The tutor gives hints and guiding questions — **never direct answers**."
)

# --- API key (no hard-coding) ---
with st.sidebar:
    st.header("🔐 API")
  #  api_key = st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else st.text_input("Enter your OpenAI API key:", type="password")

    st.caption("Tip: put `OPENAI_API_KEY = \"...\"` in `.streamlit/secrets.toml` for convenience.")
    st.divider()
    st.write("Made for middle-school learners. Upload files and interact step-by-step.")

if not api_key:
    st.info("Add your OpenAI API key in the sidebar to continue.", icon="🗝️")
    st.stop()

client = OpenAI(api_key=api_key)

# --- File uploaders ---
st.header("Step 1: Upload Homework and Study Material")
uploaded_homework = st.file_uploader("Upload your homework (required)", type=("txt", "md", "pdf", "docx"), key="homework")
uploaded_study = st.file_uploader("Upload your study/reference material (optional)", type=("txt", "md", "pdf", "docx"), key="study")

def read_any(file):
    if file is None:
        return ""
    name = (getattr(file, "name", "") or "").lower()
    mime = (getattr(file, "type", "") or "").lower()

    ext = ""
    if "." in name:
        ext = name.split(".")[-1]

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
                pages.append(reader.pages[i].extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages)

    # Plain text / md fallback
    return as_text(file.read())

homework_text = read_any(uploaded_homework) if uploaded_homework else ""
study_text = read_any(uploaded_study) if uploaded_study else ""

if uploaded_homework:
    st.header("Step 2: Let’s Work on Your Homework!")
    st.write(
        "The tutor will give **hints**, ask **guiding questions**, and check your understanding. "
        "When you’re ready, submit your own answer to get feedback."
    )

    # --- Session state ---
    if "awaiting_answer" not in st.session_state:
        st.session_state.awaiting_answer = False
    if "history" not in st.session_state:
        st.session_state.history = []  # [(role, content), ...]

    SYSTEM_INSTRUCTIONS = (
        "You are a warm, encouraging middle-school tutor.\n"
        "Rules:\n"
        "1) Never give the direct answer.\n"
        "2) Use hints, guiding questions, and short explanations.\n"
        "3) Break problems into steps; check understanding before moving on.\n"
        "4) Encourage the student to show their thinking.\n"
        "5) If the student is correct, praise them and ask for the next step.\n"
        "6) If they are done, invite them to paste final answers for review.\n"
    )

    # --- Start turn: tutor gives a first hint ---
    if not st.session_state.awaiting_answer and homework_text.strip():
        prompt = (
            "Start by giving a helpful hint or a question to get the student going. "
            "Use the study material only if relevant.\n\n"
            f"Homework:\n{homework_text}\n\n"
            f"Study Material:\n{study_text}\n"
        )
        st.subheader("Tutor")
        placeholder = st.empty()
        streamed = ""

        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=0.2,
            )
            for chunk in stream:
                piece = chunk.choices[0].delta.content or ""
                streamed += piece
                placeholder.markdown(streamed)

        st.session_state.history.append(("assistant", streamed))
        st.session_state.awaiting_answer = True

    # --- Student input ---
    if st.session_state.awaiting_answer:
        st.subheader("Your Turn")
        student_input = st.text_area("Type your answer, thought process, or next step:")
        col1, col2 = st.columns(2)

        if col1.button("Submit Answer/Step", type="primary") and student_input.strip():
            st.session_state.history.append(("user", student_input))

            followup = (
                f"The student responded:\n{student_input}\n\n"
                f"Homework:\n{homework_text}\n\n"
                f"Study Material:\n{study_text}\n\n"
                "Evaluate if the student is on track. Do NOT give the answer. "
                "Give feedback and the next hint or question. If correct, encourage and ask for the next step. "
                "If finished, ask for final answers for review. If stuck, break it down further."
            )

            st.subheader("Tutor")
            placeholder = st.empty()
            streamed = ""
            with st.spinner("Thinking..."):
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": followup},
                    ],
                    stream=True,
                    temperature=0.2,
                )
                for chunk in stream:
                    piece = chunk.choices[0].delta.content or ""
                    streamed += piece
                    placeholder.markdown(streamed)

            st.session_state.history.append(("assistant", streamed))
            st.session_state.awaiting_answer = True

        st.markdown("---")
        final_answers = st.text_area("Paste your **final answers** here for review:")
        if col2.button("Submit Final Answers") and final_answers.strip():
            review = (
                f"The student has submitted final answers:\n{final_answers}\n\n"
                f"Homework:\n{homework_text}\n\n"
                f"Study Material:\n{study_text}\n\n"
                "Review the answers. Provide feedback, corrections, and encouragement. "
                "If correct, praise. If not, give hints for improvement. **Never give the direct answer.**"
            )

            st.subheader("Tutor Review")
            placeholder = st.empty()
            streamed = ""
            with st.spinner("Reviewing..."):
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": review},
                    ],
                    stream=True,
                    temperature=0.2,
                )
                for chunk in stream:
                    piece = chunk.choices[0].delta.content or ""
                    streamed += piece
                    placeholder.markdown(streamed)

            st.session_state.history.append(("assistant", streamed))
            st.session_state.awaiting_answer = False
else:
    st.info("Please upload your homework to begin.", icon="📎")
