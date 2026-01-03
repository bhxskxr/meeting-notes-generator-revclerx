# meeting-notes-generator-revclerx
AI meeting assistant that converts raw meeting transcrpts into structred summaries and actionable insights using an LLM.

# 📝 LLM-Based Meeting Notes Generator & Action Extractor

An AI-powered assistant that converts raw, noisy meeting transcripts into structured summaries, decision logs, and actionable tasks. 

Built with **Python**, **Streamlit**, **LangChain**, and **Google Gemini 1.5 Flash**.

## 🚀 Project Overview
This tool solves the challenge of processing long, unstructured meeting text. It uses a **Map-Reduce architecture** to handle context limits, ensuring that even lengthy transcripts (1 hour+) are processed accurately without losing information.

### Key Features
* **Structured Output:** Generates clear headers for Summary, Discussion Points, Decisions, and Action Items.
* **Noise Handling:** Automatically filters filler words ("um", "ah") and informal chatter.
* **Long Context Support:** Implements **Chunking (Map-Reduce)** to split long texts into segments, analyze them in parallel, and synthesize a final report.
* **Action Extraction:** Identifies tasks and assigns owners automatically.

---

## ⚙️ Technical Implementation
* **LLM:** Google Gemini 1.5 Flash (Optimized for speed and long context).
* **Framework:** LangChain (for prompt engineering and chains).
* **UI:** Streamlit (for instant web interface).
* **Context Strategy:** * **Splitting:** Uses `RecursiveCharacterTextSplitter` to break text into 4,000-character chunks with overlap.
    * **Mapping:** Extracts partial insights from each chunk.
    * **Reducing:** Merges partial insights into a cohesive final report.

---

## 🛠️ Setup & Installation Guide

Follow these steps to run the project locally.

### 1. Prerequisites
* Python 3.10 or higher installed.
* A Google Cloud API Key (for Gemini).

### 2. Clone the Repository
Open your terminal/command prompt and run:
```bash
git clone [https://github.com/YOUR_USERNAME/meeting-notes-generator.git](https://github.com/YOUR_USERNAME/meeting-notes-generator.git)
cd meeting-notes-generator

