import os
import time
import json
import requests
import argparse
import base64
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langfuse import get_client, observe
from evaluation.metrics import evaluate_hallucination_and_relevance
from evaluation.db import init_db, save_evaluation

# ------------------------------------------------------------
# 1️⃣ Load environment variables and initialize Langfuse client
# ------------------------------------------------------------
load_dotenv()
lf = get_client()

# Increase OTLP timeout for slow networks
os.environ["OTEL_EXPORTER_OTLP_TIMEOUT"] = os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "20")

# Initialize local SQLite DB
init_db()


# ------------------------------------------------------------
# 2️⃣ Helper: Send scores + metadata to Langfuse (modern Basic Auth)
# ------------------------------------------------------------
def send_scores(trace_id: str, scores: list, model_name: str = None, sample_id: str = None):
    """Attach metadata to Langfuse trace (skip /scores endpoint)."""
    try:
        base_url = os.getenv("LANGFUSE_HOST")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not all([base_url, public_key, secret_key]):
            print("⚠️ Missing Langfuse environment vars.")
            return

        # Build Basic Auth header
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {token}"
        }

        # --- Attach metadata (only) ---
        metadata_payload = {
            "traceId": trace_id,
            "metadata": {
                "model_name": model_name or "unknown_model",
                "sample_id": sample_id or "unknown_sample",
                "latency": next((s["value"] for s in scores if s.get("name") == "latency"), None),
                "faithfulness": next((s["value"] for s in scores if s.get("name") == "faithfulness"), None),
                "relevance": next((s["value"] for s in scores if s.get("name") == "relevance"), None),
                "timestamp": int(time.time())
            }
        }

        resp_meta = requests.post(
            f"{base_url}/api/public/traces",
            headers=headers,
            data=json.dumps(metadata_payload),
            timeout=10
        )

        if resp_meta.status_code == 200:
            print(" Metadata successfully attached to trace .")
        else:
            print(f"⚠️ Langfuse (metadata) responded with {resp_meta.status_code}: {resp_meta.text}")

    except Exception as e:
        print(f"⚠️ Failed to send metadata to Langfuse: {e}")

# ------------------------------------------------------------
# 3️⃣ Observed function for LLM execution and evaluation
# ------------------------------------------------------------
@observe(name="LLMSentinel-TestRun")
def execute_and_observe_llm(
    query: str,
    context: str,
    mode: str = "ollama",
    model_name: str = "model_default",
    sample_id: str = "sample_default",
    api_url: str = None
):
    """
    mode: 'ollama' | 'manual' | 'api'
    - ollama: run local ChatOllama target model
    - manual: prompt for user to paste model output
    - api: fetch from local HTTP model endpoint specified by api_url
    """
    start_time = time.time()

    # --------------------------------------------------------
    # Select mode
    # --------------------------------------------------------
    if mode == "ollama":
        TARGET_MODEL = ChatOllama(
            model=os.getenv("TARGET_MODEL_NAME", "llama3"),
            temperature=float(os.getenv("TARGET_MODEL_TEMP", "0.7"))
        )

        @observe(name="Model-Call", as_type="generation")
        def do_model_call(prompt):
            return TARGET_MODEL.invoke(prompt).content

        prompt = f"Using the following context, answer the query:\n\nQuery: {query}\n\nContext: {context}"
        model_output = do_model_call(prompt)

    elif mode == "api":
        if not api_url:
            raise ValueError("api_url must be provided for mode='api'")
        try:
            resp = requests.post(api_url, json={"query": query, "context": context}, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            model_output = body.get("output") or body.get("result") or str(body)
        except Exception as e:
            print(f"⚠️ API call failed: {e}")
            model_output = ""

    elif mode == "manual":
        print("\n🧠 Manual Evaluation Mode Active — paste your custom model output below:\n")
        model_output = input("Paste model output (end with Enter):\n\n")

    else:
        raise ValueError("Invalid mode. Use 'ollama', 'manual', or 'api'.")

    # --------------------------------------------------------
    # Measure latency
    # --------------------------------------------------------
    latency = time.time() - start_time
    print(f"\n✅ Model Output (first 200 chars): {model_output[:200]}...")
    print(f"⏱️ Latency: {latency:.2f}s")

    # Retrieve or create trace ID
    trace_id = lf.get_current_trace_id() or f"trace-{mode}-{int(time.time())}"
    print(f"🧭 Trace ID: {trace_id}")

    # --------------------------------------------------------
    # Evaluate using Judge (metrics.py)
    # --------------------------------------------------------
    print("\n🔍 Running LLM-as-a-Judge evaluation (metrics.py)...")
    eval_scores = evaluate_hallucination_and_relevance(query, model_output, context)

    # Prepare validated numeric scores
    scores = [
        {"name": "latency", "value": float(latency)},
        {"name": "faithfulness", "value": float(eval_scores.get("faithfulness_score", 0))},
        {"name": "relevance", "value": float(eval_scores.get("relevance_score", 0))},
    ]

    # --------------------------------------------------------
    # Send to Langfuse
    # --------------------------------------------------------
    send_scores(trace_id, scores, model_name=model_name, sample_id=sample_id)

    # --------------------------------------------------------
    # Save locally to DB
    # --------------------------------------------------------
    record = {
        "trace_id": trace_id,
        "model_name": model_name,
        "sample_id": sample_id,
        "query": query,
        "context": context,
        "faithfulness": int(eval_scores.get("faithfulness_score", 0)),
        "relevance": int(eval_scores.get("relevance_score", 0)),
        "latency": float(latency),
        "created_at": int(time.time()),
    }

    try:
        save_evaluation(record)
        print("💾 Saved evaluation to local DB.")
    except Exception as e:
        print(f"⚠️ Failed to save evaluation to DB: {e}")

    # --------------------------------------------------------
    # Print trace link for Langfuse
    # --------------------------------------------------------
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    trace_url = f"{os.getenv('LANGFUSE_HOST')}/project/{project_name}/traces/{trace_id}"
    print(f"🔗 View Trace in Langfuse: {trace_url}")

    return {
        "model_output": model_output,
        "faithfulness": record["faithfulness"],
        "relevance": record["relevance"],
        "latency": latency,
        "trace_url": trace_url,
    }


# ------------------------------------------------------------
# 4️⃣ Entry point (CLI)
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ollama or custom model using LLM-as-a-Judge.")
    parser.add_argument("--mode", type=str, default="ollama", choices=["ollama", "manual", "api"],
                        help="Choose evaluation mode.")
    parser.add_argument("--model-name", type=str, default=os.getenv("DEFAULT_MODEL_NAME", "model_default"),
                        help="Name for the evaluated model (e.g., 'my_local_v1').")
    parser.add_argument("--sample-id", type=str, default="sample_default",
                        help="Sample/query identifier (e.g., 'q1' or 'smoke-01').")
    parser.add_argument("--api-url", type=str, default=None,
                        help="If mode=api, URL of model endpoint (e.g., http://localhost:5000/predict)")
    args = parser.parse_args()

    SAMPLE_QUERY = (
        "Explain the concept of quantum entanglement in simple terms, "
        "but do NOT use the word 'spooky'."
    )
    SAMPLE_CONTEXT = (
        "Quantum entanglement is a phenomenon where two or more particles become "
        "linked, or correlated, in such a way that measuring a property of one "
        "instantaneously influences the corresponding property of the others, "
        "regardless of the distance separating them."
    )

    execute_and_observe_llm(
        SAMPLE_QUERY,
        SAMPLE_CONTEXT,
        mode=args.mode,
        model_name=args.model_name,
        sample_id=args.sample_id,
        api_url=args.api_url
    )
