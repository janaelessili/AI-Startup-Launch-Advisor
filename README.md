# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

---

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Jana Yasser Elessili                 |
| Project Name     | AI Startup Launch Advisor            |
| GitHub Username  | janaelessili                         |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en) |

---

# 📖 Project Overview

**AI Startup Launch Advisor** is an AI-powered startup evaluation system that helps users analyze and validate startup ideas.

The system takes a startup idea as input, along with details such as industry, target market, target customers, budget level, and startup stage. It then combines **RAG**, **Chains**, **Tavily Web Search Tool**, and **Structured Output Parsing** to generate a complete startup evaluation report.

The final report includes startup score, problem analysis, solution summary, target customers, real competitors, market size analysis, SWOT analysis, business model, MVP plan, pricing strategy, risks, validation questions, next steps, and source links.

---

# ✨ Features

* 🚀 Startup idea evaluation based on structured business criteria.
* 🧠 RAG-based business knowledge retrieval using FAISS vector database.
* 🔗 Multi-step chains for idea understanding, RAG retrieval, web research, and final evaluation.
* 🌐 Real-time web search using Tavily API to find competitors, market size, pricing examples, and industry trends.
* 📊 Structured startup report generated using LangChain `StructuredOutputParser`.
* 🧩 SWOT analysis, MVP plan, pricing strategy, risks, and next steps.
* 🔗 Source links for competitors and market research.
* 💻 Interactive Streamlit UI.
* 📥 Downloadable JSON report.

---

# 🛠️ Technologies Used

* Python
* Streamlit
* LangChain
* StructuredOutputParser
* Prompt Engineering
* Tavily Search API
* Hugging Face Transformers
* Qwen2.5 Instruct Model
* Sentence Transformers
* FAISS Vector Database
* Torch
* JSON Parsing

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/AI-Startup-Launch-Advisor.git
cd AI-Startup-Launch-Advisor
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 API Key Setup

This project uses **Tavily Search API** for real web search.

Create a folder named `.streamlit`, then create a file inside it named `secrets.toml`.

```text
.streamlit/
└── secrets.toml
```

Inside `secrets.toml`, add your Tavily API key:

```toml
TAVILY_API_KEY = "PASTE_YOUR_TAVILY_API_KEY_HERE"
```

> ⚠️ Do not upload your real API key to GitHub.

---

# 🚀 Usage

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, for example:

```text
http://localhost:8501
```

or:

```text
http://127.0.0.1:8501
```

---

# 🧪 How It Works

The project follows this pipeline:

```text
User Startup Idea
        ↓
Idea Understanding Chain
        ↓
RAG Business Knowledge Retrieval
        ↓
Tavily Web Search Tool
        ↓
Startup Evaluation Chain
        ↓
Structured Output Parser
        ↓
Final JSON Report
```

---

# 🧠 RAG Component

The project includes a local business knowledge base covering:

* Startup validation
* Problem-solution fit
* Target customers
* Competitor analysis
* SWOT analysis
* MVP planning
* Business models
* Pricing strategy
* Market sizing
* Risk analysis
* Go-to-market strategy

The knowledge base is split into chunks, converted into embeddings using Sentence Transformers, and stored in a FAISS vector database. Relevant chunks are retrieved based on the user's startup idea and used as business context during evaluation.

---

# 🔗 Chains Used

The system is divided into multiple chains:

## 1. Idea Understanding Chain

Extracts and organizes the startup idea details.

## 2. RAG Retrieval Chain

Retrieves relevant business knowledge from the local FAISS vector database.

## 3. Web Research Chain

Generates search queries and uses Tavily API to collect real web results.

## 4. Final Evaluation Chain

Combines the user input, retrieved RAG context, and web search results to generate the final startup evaluation.

---

# 🧾 Output Parser

The project uses LangChain `StructuredOutputParser` to generate a clean JSON output with the following fields:

* Startup Score
* Overall Judgment
* Startup Summary
* Problem
* Solution
* Target Customers
* Retrieved Business Context
* Real Competitors
* Market Size Analysis
* SWOT Analysis
* Business Model
* MVP Plan
* Pricing Strategy
* Risk Analysis
* Validation Questions
* Next Steps
* Source Links

---

# 📈 Results

The system successfully generates a structured startup evaluation report based on:

* User-provided startup idea
* Retrieved business knowledge from RAG
* Real web search results from Tavily
* Structured JSON parsing

The output helps founders understand the strengths, weaknesses, market opportunities, risks, competitors, and practical next steps for their startup idea.

---

# 🔮 Future Improvements

* Add PDF export for the final startup report.
* Add user accounts and saved startup evaluations.
* Improve competitor ranking using match scores.
* Add financial projections and revenue estimation.
* Add pitch deck generation.
* Add support for Arabic startup ideas.
* Deploy the Streamlit app online.

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
