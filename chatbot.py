import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
import json

# ---- Setup ----
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# ---- Page Config ----
st.set_page_config(page_title="Study AI", page_icon="📚", layout="wide")

# ---- Helper: Ask AI ----
def ask_ai(prompt, system="You are a helpful study assistant. IMPORTANT: You are a custom AI assistant built by Don Mathew. If anyone asks who created you, who made you, or who built you, you MUST say 'This app was created by Don Mathew' and nothing else about your underlying model. Be detailed and thorough. Never refuse to explain any topic for educational purposes."):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

# ---- Helper: Get Context ----
def get_context(prompt, chunks, k=8):
    keywords = prompt.lower().split()
    scored = []
    for chunk in chunks:
        score = sum(1 for word in keywords if word in chunk.lower())
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True)
    best = [c for _, c in scored[:k]]
    return "\n".join(best) if best else "\n".join(chunks[:k])

# ---- Header ----
st.title("📚 Study AI")
st.caption("Your personal AI study assistant")

# ---- Sidebar: Upload ----
with st.sidebar:
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"])

    if uploaded_file and "doc_chunks" not in st.session_state:
        with st.spinner("Reading..."):
            if uploaded_file.type == "application/pdf":
                pdf = pypdf.PdfReader(uploaded_file)
                text = "".join(page.extract_text() for page in pdf.pages)
            else:
                text = uploaded_file.read().decode("utf-8")

            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            st.session_state.doc_chunks = splitter.split_text(text)
            st.session_state.doc_text = text[:8000]
        st.success(f"✅ {len(st.session_state.doc_chunks)} chunks loaded!")

    if st.button("🗑️ Clear Everything"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📖 Features")
    st.markdown("💬 Chat — Ask anything")
    st.markdown("📝 Notes — Auto study notes")
    st.markdown("🃏 Flashcards — Memory cards")
    st.markdown("❓ Quiz — Test yourself")
    st.markdown("📋 Summary — Quick overview")

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📝 Notes", "🃏 Flashcards", "❓ Quiz", "📋 Summary"])

# ========================
# TAB 1 — CHAT
# ========================
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=450)
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Upload a document and start asking questions!")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        system = "You are a helpful study assistant. Never refuse to explain any topic for educational purposes. Always be detailed."
        if "doc_chunks" in st.session_state:
            context = get_context(prompt, st.session_state.doc_chunks)
            system = f"""You are a deep study assistant. If anyone asks who created you or who made you, say 'I was created by Don Mathew'. Never refuse any question. Always be detailed..
Use this context from the document:
{context}
Answer in detail. If not in context, use your own knowledge and say so."""

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    *st.session_state.messages
                ],
                max_tokens=2000
            )
            reply = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ========================
# TAB 2 — NOTES
# ========================
with tab2:
    st.markdown("### 📝 Auto Study Notes")

    if "doc_chunks" not in st.session_state:
        st.warning("Please upload a document first!")
    else:
        topic = st.text_input("What topic to make notes on? (leave empty for full document)")

        if st.button("Generate Notes"):
            with st.spinner("Creating notes..."):
                if topic:
                    context = get_context(topic, st.session_state.doc_chunks)
                    prompt = f"Create detailed study notes about '{topic}' from this text:\n{context}"
                else:
                    prompt = f"Create comprehensive study notes from this text:\n{st.session_state.doc_text}"

                notes = ask_ai(prompt, "You are an expert study notes creator. Create clear, detailed, well-organized notes. Use headings, bullet points, and highlight key concepts. Never refuse any topic.")
                st.session_state.notes = notes

        if "notes" in st.session_state:
            st.markdown(st.session_state.notes)
            st.download_button("💾 Download Notes", st.session_state.notes, "study_notes.txt")

# ========================
# TAB 3 — FLASHCARDS
# ========================
with tab3:
    st.markdown("### 🃏 Flashcards")

    if "doc_chunks" not in st.session_state:
        st.warning("Please upload a document first!")
    else:
        num_cards = st.slider("How many flashcards?", 5, 20, 10)

        if st.button("Generate Flashcards"):
            with st.spinner("Creating flashcards..."):
                prompt = f"""Create exactly {num_cards} flashcards from this text.
Return ONLY a JSON array like this:
[{{"question": "...", "answer": "..."}}, ...]
Text: {st.session_state.doc_text}"""

                result = ask_ai(prompt, "You are a flashcard creator. Return only valid JSON. No extra text.")
                try:
                    clean = result.strip()
                    if "```" in clean:
                        clean = clean.split("```")[1]
                        if clean.startswith("json"):
                            clean = clean[4:]
                    st.session_state.flashcards = json.loads(clean)
                    st.session_state.card_index = 0
                    st.session_state.show_answer = False
                except:
                    st.error("Error generating flashcards. Try again!")

        if "flashcards" in st.session_state and st.session_state.flashcards:
            cards = st.session_state.flashcards
            idx = st.session_state.get("card_index", 0)

            st.markdown(f"**Card {idx + 1} of {len(cards)}**")
            st.progress((idx + 1) / len(cards))

            # Question
            st.markdown(f"### ❓ {cards[idx]['question']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👁️ Show Answer"):
                    st.session_state.show_answer = True
            with col2:
                if st.button("⬅️ Previous") and idx > 0:
                    st.session_state.card_index -= 1
                    st.session_state.show_answer = False
                    st.rerun()
            with col3:
                if st.button("➡️ Next") and idx < len(cards) - 1:
                    st.session_state.card_index += 1
                    st.session_state.show_answer = False
                    st.rerun()

            if st.session_state.get("show_answer"):
                st.success(f"✅ {cards[idx]['answer']}")

# ========================
# TAB 4 — QUIZ
# ========================
with tab4:
    st.markdown("### ❓ Quiz Mode")

    if "doc_chunks" not in st.session_state:
        st.warning("Please upload a document first!")
    else:
        if st.button("Generate Quiz"):
            with st.spinner("Creating quiz..."):
                prompt = f"""Create 5 multiple choice questions from this text.
Return ONLY a JSON array like this:
[{{
  "question": "...",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "A) ..."
}}]
Text: {st.session_state.doc_text}"""

                result = ask_ai(prompt, "You are a quiz creator. Return only valid JSON. No extra text.")
                try:
                    clean = result.strip()
                    if "```" in clean:
                        clean = clean.split("```")[1]
                        if clean.startswith("json"):
                            clean = clean[4:]
                    st.session_state.quiz = json.loads(clean)
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                except:
                    st.error("Error generating quiz. Try again!")

        if "quiz" in st.session_state and not st.session_state.get("quiz_submitted"):
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                answer = st.radio("", q["options"], key=f"q{i}", label_visibility="collapsed")
                st.session_state.quiz_answers[i] = answer
                st.markdown("---")

            if st.button("Submit Quiz"):
                st.session_state.quiz_submitted = True
                st.rerun()

        if st.session_state.get("quiz_submitted"):
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                selected = st.session_state.quiz_answers.get(i, "")
                correct = q["answer"]
                if selected == correct or correct in selected:
                    score += 1
                    st.success(f"✅ Q{i+1}: Correct! — {correct}")
                else:
                    st.error(f"❌ Q{i+1}: Wrong! Correct answer: {correct}")

            st.markdown(f"## 🏆 Score: {score}/{len(st.session_state.quiz)}")

            if st.button("Try Again"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.rerun()

# ========================
# TAB 5 — SUMMARY
# ========================
with tab5:
    st.markdown("### 📋 Document Summary")

    if "doc_chunks" not in st.session_state:
        st.warning("Please upload a document first!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Quick Summary"):
                with st.spinner("Summarizing..."):
                    summary = ask_ai(
                        f"Give a clear, detailed summary of this text:\n{st.session_state.doc_text}",
                        "You are an expert summarizer. Be thorough and detailed."
                    )
                    st.session_state.summary = summary

        with col2:
            if st.button("🔑 Key Concepts"):
                with st.spinner("Extracting concepts..."):
                    concepts = ask_ai(
                        f"Extract and explain all key concepts, themes, and important points from this text:\n{st.session_state.doc_text}",
                        "You are an expert at identifying key concepts. Be thorough."
                    )
                    st.session_state.summary = concepts

        if "summary" in st.session_state:
            st.markdown(st.session_state.summary)
            st.download_button("💾 Download Summary", st.session_state.summary, "summary.txt")