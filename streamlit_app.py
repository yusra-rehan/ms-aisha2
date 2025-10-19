import streamlit as st
from openai import OpenAI

import io
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None
import re
from urllib.parse import urlparse
try:
    from googleapiclient.discovery import build
except Exception:
    build = None




st.set_page_config(page_title="MS AISHA — AI Student Helper", page_icon="📄", layout="centered")

st.title("📄 🎓Ms. AISHA ✏️ 🤖") 
st.title("(Artificially Intelligent Student Helping Agent)")
st.write(
    "Upload your homework and any reference materials, then ask a question. "
    "The tutor gives hints and guiding questions — **never direct answers**."
)

# --- API key (no hard-coding) ---
with st.sidebar:
    st.header("🔐 API")
    # api_key = st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else st.text_input("Enter your OpenAI API key:", type="password")
    # YouTube API key (optional) — used only if you enable automatic YouTube search below
    youtube_api_key = st.secrets.get("youtube_api_key") if "youtube_api_key" in st.secrets else st.text_input("YouTube API key (optional):", type="password")
    enable_youtube_search = st.checkbox("Allow agent to fetch YouTube links (uses YouTube Data API)", value=False)

    st.divider()
    st.write("Made for middle-school learners. Upload files and interact step-by-step.")
    debug_ui = st.checkbox("Show debug info", value=False)

if not api_key:
    st.info("Add your OpenAI API key in the sidebar to continue.", icon="🗝️")
    st.stop()

client = OpenAI(api_key=api_key)

# construct youtube client if requested and key present
youtube_client = None
if enable_youtube_search:
    if not youtube_api_key:
        st.sidebar.warning("You enabled YouTube search but did not provide a YouTube API key in the sidebar or in Streamlit secrets.")
    elif build is None:
        st.sidebar.warning("google-api-python-client is not installed. Add it to requirements.txt to enable YouTube searches.")
    else:
        try:
            youtube_client = build("youtube", "v3", developerKey=youtube_api_key)
        except Exception as e:
            st.sidebar.error(f"Failed to create YouTube client: {e}")

# Debug UI: show raw assistant output and youtube client status
if 'history' in st.session_state and debug_ui:
    st.sidebar.markdown("**Debug**")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.sidebar.write("Last message (role):", last[0])
        st.sidebar.text_area("Last assistant raw output", value=last[1], height=200)
    else:
        st.sidebar.write("No conversation history yet.")
    st.sidebar.write("YouTube client:", "present" if youtube_client else "not present")

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


### Media rendering helpers
ALLOWED_HOSTS =  {"youtube.com","youtu.be","vimeo.com","imgur.com"}

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        if ALLOWED_HOSTS:
            host = parsed.netloc.lower()
            return any(h in host for h in ALLOWED_HOSTS)
        return True
    except Exception:
        return False

def render_model_output(text: str, container=None):
    """Render text and any MEDIA tokens found in `text` into the Streamlit app."""
    
    # show the textual content (remove token lines)
    cleaned = re.sub(r'^(VIDEO:.*|IMAGE:.*|LINK:.*)$', '', text, flags=re.MULTILINE).strip()
    # Also remove inline Search: tokens from the cleaned text
    cleaned = re.sub(r'Search:\s*[^.!?\n]+', '', cleaned)
    
    if cleaned:
        if container:
            container.markdown(cleaned)
        else:
            st.markdown(cleaned)

    # videos
    for m in re.findall(r'^VIDEO:(\S+)$', text, flags=re.MULTILINE):
        url = m.strip()
        if is_valid_url(url):
            st.video(url)
        else:
            st.warning(f"Video URL appears invalid or blocked: {url}")

    # images
    for m in re.findall(r'^IMAGE:(\S+)$', text, flags=re.MULTILINE):
        url = m.strip()
        if is_valid_url(url):
            st.image(url, use_column_width=True)
        else:
            st.warning(f"Image URL appears invalid or blocked: {url}")

    # links: LINK:<url>|<label>
    for m in re.findall(r'^LINK:(\S+)\|(.+)$', text, flags=re.MULTILINE):
        url, label = m
        url = url.strip()
        label = label.strip()
        if is_valid_url(url):
            st.markdown(f"[{label}]({url})")
        else:
            st.warning(f"Link appears invalid or blocked: {url}")

    # search tokens: Search: <query> -> perform youtube search if enabled
    # Changed to match Search: anywhere in text, not just on its own line
    for m in re.findall(r'Search:\s*([^.!?\n]+)', text):
        query = m.strip()
        if not query:
            continue
        if youtube_client is None:
            st.info(f"Suggested search: '{query}'. Enable YouTube search in the sidebar to fetch results automatically.")
            continue
        # perform a simple YouTube search and render the top video
        try:
            res = youtube_client.search().list(q=query, part="snippet", type="video", maxResults=1).execute()
            items = res.get("items", [])
            if not items:
                st.info(f"No YouTube results found for: {query}")
                continue
            video_id = items[0]["id"]["videoId"]
            video_url = f"https://youtu.be/{video_id}"
            st.video(video_url)
            st.markdown(f"[Watch on YouTube]({video_url})")
        except Exception as e:
            st.warning(f"YouTube search failed for '{query}': {e}")

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
    "7) ALWAYS recommend at least one helpful video for the topic using the format below.\n"
    "\nMedia and links formatting rules:\n"
    "- For video recommendations, use: Search: <query>\n"
    "  Example: Search: fractions to decimals for middle school\n"
    "- ALWAYS include at least one Search: token in your response to help students with visual learning.\n"
    "- Make search queries specific and middle-school appropriate.\n"
    "- You can include multiple Search: tokens for different aspects of the problem.\n"
    "- Keep search queries concise (5-8 words max).\n"
    "- Place Search: tokens naturally in your explanations where videos would be most helpful.\n"
)

    # --- Start turn: tutor gives a first hint ---
    if not st.session_state.awaiting_answer and homework_text.strip():
        prompt = (
            "Start by giving a helpful hint or a question to get the student going. "
            "If you recommend videos or images, include them using the MEDIA tokens (VIDEO:/IMAGE:/LINK) specified in system instructions. "
            "If you cannot provide a reliable link, provide a short search query or video title the student can use to find a tutorial. "
            "Ask the student if they know how to start; if not, provide the next hint or a suggested video (as a token). "
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
        # after stream finishes, render potential media tokens
        render_model_output(streamed, container=placeholder)

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
                "If not correct, try to understand where the student is going wrong and provide helpful hints. "
                "You may include VIDEO:/IMAGE:/LINK tokens with full https URLs if you have reliable links; otherwise include suggested search queries or video titles the student can search for. "
                "If still not correct, try to draw simple diagrams or provide simpler explanations. "
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
            render_model_output(streamed, container=placeholder)

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
            render_model_output(streamed, container=placeholder)
            st.session_state.history.append(("assistant", streamed))
            st.session_state.awaiting_answer = False
else:
    st.info("Please upload your homework to begin.", icon="📎")
