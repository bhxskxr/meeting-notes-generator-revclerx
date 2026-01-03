import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# ==========================================
# SETUP: API KEY
# ==========================================
# ⚠️ REPLACE THIS WITH YOUR ACTUAL KEY BEFORE RUNNING
os.environ["GOOGLE_API_KEY"] = "AIzaSyCrb4hsdrHX8TGOzQSToU09p57oHSbjbCk"

# ==========================================
# CORE LOGIC: MAP-REDUCE STRATEGY
# ==========================================

# 1. Configure the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# 2. Define the "Map" Prompt (Processing individual chunks)
# We ask the AI to extract raw data from this specific slice of text.
map_prompt_template = """
You are analyzing a segment of a long meeting transcript.
Extract the following details from this specific segment:
1. Key discussion points
2. Decisions made (if any)
3. Action items with owners (if any)

If nothing significant happens in this segment, just say "No key details."

SEGMENT TEXT:
"{text}"

RAW EXTRACTION:
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
map_chain = map_prompt | llm | StrOutputParser()

# 3. Define the "Reduce" Prompt (Merging everything)
# We ask the AI to take the list of extractions and create the final report.
reduce_prompt_template = """
You are an expert Executive Assistant.
Below are raw notes extracted from different parts of a meeting.
Your job is to merge them into one cohesive, structured report.
Remove duplicates and organize logically.

RAW NOTES FROM CHUNKS:
"{text}"

-----------------------
GENERATE FINAL REPORT IN THIS FORMAT:
### 1. Concise Summary
(3-5 sentences summarizing the whole meeting)

### 2. Key Discussion Points
(Bulleted list)

### 3. Decisions Made
(Explicit agreements)

### 4. Action Items
(Format: [Owner]: [Task])
"""
reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["text"])
reduce_chain = reduce_prompt | llm | StrOutputParser()

# 4. The Main Processing Function (The "Chunking" Logic)
def process_transcript_with_chunking(full_text):
    # A. Split the text into chunks
    # chunk_size=4000 means roughly 1-2 pages of text per chunk.
    # chunk_overlap=200 ensures we don't cut a sentence in half and lose context at the edge.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.create_documents([full_text])
    
    st.info(f"ℹ️ Text is too long. Split into {len(chunks)} chunks for processing.")
    
    # B. Map Phase: Process each chunk
    partial_notes = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        # Update progress bar
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress, text=f"Analyzing chunk {i+1} of {len(chunks)}...")
        
        # Run the Map Chain
        response = map_chain.invoke({"text": chunk.page_content})
        partial_notes.append(response)
    
    # C. Reduce Phase: Combine results
    combined_text = "\n---\n".join(partial_notes)
    st.caption("Merging all chunks into final report...")
    final_report = reduce_chain.invoke({"text": combined_text})
    
    return final_report

# ==========================================
# USER INTERFACE (STREAMLIT)
# ==========================================
st.set_page_config(page_title="AI Meeting Notes", layout="wide")
st.title("📝 AI Meeting Notes (Chunking Enabled)")
st.markdown("""
This tool uses a **Map-Reduce** strategy to handle long transcripts:
1. **Splits** text into manageable chunks.
2. **Extracts** insights from each chunk.
3. **Merges** results into a final report.
""")

# Text Area for input
transcript_text = st.text_area("Paste Transcript Here:", height=300)

if st.button("Generate Notes"):
    if transcript_text:
        try:
            # Check length to decide if we need chunking or not
            # (We force chunking here to demonstrate the feature as requested)
            final_output = process_transcript_with_chunking(transcript_text)
            
            st.markdown("### 🏁 Final Meeting Minutes")
            st.markdown(final_output)
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please paste some text first.")