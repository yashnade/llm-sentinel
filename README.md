# 🧠 LLM-Sentinel: AI Model Evaluation & Observability System

**LLM-Sentinel** is a modular framework for evaluating and monitoring the performance of AI and LLM models — whether they are **local (Ollama)**, **custom-built**, or **served via APIs**.  
It combines **Langfuse** for observability, **LangChain** for orchestration, and **Streamlit** for interactive analytics, allowing researchers and developers to benchmark, compare, and track model performance over time.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-Enabled-green" alt="LangChain">
  <img src="https://img.shields.io/badge/Langfuse-Integrated-purple" alt="Langfuse">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-orange" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

---

## 🚀 Key Features

| Feature | Description |
|----------|-------------|
| 🧩 **Multi-Mode Evaluation** | Supports evaluation of **local**, **manual**, and **API-based** models. |
| ⚙️ **Automated Metrics** | Calculates **faithfulness**, **relevance**, and **latency** for every model response. |
| 💾 **Local Database Storage** | All results are stored persistently in a **SQLite** database for offline analysis. |
| 📊 **Interactive Dashboard** | Streamlit-based dashboard provides visual analytics including bar, line, and latency charts. |
| 🌐 **Langfuse Integration** | Real-time observability, trace logging, and metadata visualization for LLM runs. |
| 🧮 **Static Reports** | `plot_dashboard.py` generates shareable static charts (`.png`) for reports or presentations. |
| 🔐 **Secure Environment Management** | API keys and configurations handled safely via `.env` variables. |

---

## 🏗️ Project Structure


