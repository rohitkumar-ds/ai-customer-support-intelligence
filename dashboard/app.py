from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

st.set_page_config(
    page_title="AI Customer Support Intelligence",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Customer Support Intelligence Platform")
st.caption("Classical NLP → Transformers → Embeddings → RAG → grounded support")


# ============================================================
# SHARED SERVICES
# ============================================================

@st.cache_resource
def get_service():
    from src.api.main import build_service
    return build_service()


def get_repository():
    from src.database.connection import get_session
    from src.database.repository import TicketRepository
    session = get_session()
    return session, TicketRepository(session)


def show_analysis(result):
    cols = st.columns(5)
    values = [
        ("Intent", result.get("intent")),
        ("Sentiment", result.get("sentiment")),
        ("Priority", result.get("priority")),
        ("Department", result.get("department")),
        ("Confidence", result.get("confidence", result.get("intent_confidence"))),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, str(value))

    st.write("**Entities**")
    st.json(result.get("entities", {}))


# ============================================================
# NAVIGATION
# ============================================================

tabs = st.tabs([
    "Home",
    "AI Support Chat",
    "Ticket Analyzer",
    "Knowledge Search",
    "Ticket History",
    "Analytics",
    "Model Comparison",
    "RAG Evaluation",
])


# ============================================================
# HOME
# ============================================================

with tabs[0]:
    st.subheader("System overview")
    st.markdown(
        "**Customer → analysis → retrieval → grounded generation → "
        "verification → SQLite**"
    )
    st.info(
        "The default LLM provider is MockLLM for offline demos. "
        "Set LLM_PROVIDER=gemini and GEMINI_API_KEY for live generation."
    )

    cols = st.columns(4)
    cols[0].metric("NLP", "Intent + Sentiment")
    cols[1].metric("Retrieval", "FAISS")
    cols[2].metric("Generation", "RAG + LLM")
    cols[3].metric("Storage", "SQLite")


# ============================================================
# AI SUPPORT CHAT
# ============================================================

with tabs[1]:
    st.subheader("AI Support Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    message = st.text_area(
        "Customer message",
        placeholder="How long does a refund take?",
        key="chat_message",
    )

    if st.button("Analyze & Respond", key="chat"):
        if not message.strip():
            st.warning("Enter a customer message.")
        else:
            try:
                classifier, sentiment, rag = get_service()
                from src.services.support_service import SupportService

                session, repository = get_repository()
                try:
                    service = SupportService(
                        classifier,
                        sentiment,
                        rag,
                        repository,
                    )

                    start = time.perf_counter()

                    result = service.chat(
                        message,
                        "streamlit",
                        st.session_state.chat_history,
                    )

                    result["total_latency_ms"] = round(
                        (time.perf_counter() - start) * 1000,
                        2,
                    )

                    session.commit()
                finally:
                    session.close()

                st.session_state.chat_history.extend([
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": result["answer"]},
                ])

                st.success(result["answer"])
                show_analysis(result)

                st.write("**Sources**")
                sources = result.get("sources", [])

                if sources:
                    st.dataframe(
                        pd.DataFrame(sources),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No knowledge source was retrieved.")

                if result.get("escalate"):
                    st.warning(
                        "Human escalation: "
                        + result.get("escalation_reason", "")
                    )

                st.caption(
                    f"Analysis: {result.get('analysis_latency_ms', result.get('latency_ms', 0))} ms · "
                    f"Retrieval: {result.get('retrieval_latency_ms', 0)} ms · "
                    f"Generation: {result.get('generation_latency_ms', 0)} ms · "
                    f"Total: {result.get('total_latency_ms', 0)} ms"
                )

            except Exception as exc:
                st.error(f"Request failed: {exc}")


# ============================================================
# TICKET ANALYZER
# ============================================================

with tabs[2]:
    st.subheader("Ticket Analyzer")

    message = st.text_area(
        "Message to classify",
        placeholder="I was charged twice for the same transaction.",
        key="analyze",
    )

    if st.button("Analyze ticket", key="ticket"):
        if not message.strip():
            st.warning("Enter a message.")
        else:
            try:
                classifier, sentiment, _ = get_service()
                from src.services.support_service import SupportService

                result = SupportService(
                    classifier,
                    sentiment,
                    None,
                ).analyze(message)

                show_analysis(result)

                with st.expander("Raw analysis output"):
                    st.json(result)

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


# ============================================================
# KNOWLEDGE SEARCH
# ============================================================

with tabs[3]:
    st.subheader("Knowledge Search")

    query = st.text_input(
        "Search support knowledge",
        placeholder="How long before a refund appears on my card?",
        key="knowledge_query",
    )

    top_k = st.slider(
        "Number of results",
        1,
        5,
        3,
    )

    if st.button("Search", key="search"):
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            try:
                from src.retrieval.retriever import Retriever

                results = Retriever().search(
                    query,
                    top_k=top_k,
                    threshold=0.55,
                )

                if not results:
                    st.warning(
                        "No sufficiently similar knowledge was found."
                    )
                else:
                    for i, result in enumerate(results, 1):
                        st.markdown(
                            f"### Result {i} · "
                            f"similarity {result['similarity']:.4f}"
                        )
                        st.write(
                            f"**Document:** {result['document']}  "
                            f"**Page:** {result.get('page')}"
                        )
                        st.write(result["text"])
                        st.divider()

            except Exception as exc:
                st.error(f"Knowledge search failed: {exc}")


# ============================================================
# TICKET HISTORY
# ============================================================

with tabs[4]:
    st.subheader("Ticket History")

    try:
        session, repository = get_repository()
        try:
            tickets = repository.list_tickets(200)
        finally:
            session.close()

        if not tickets:
            st.info("No tickets stored yet.")
        else:
            data = []

            for ticket in tickets:
                data.append({
                    "ID": ticket.id,
                    "Created": ticket.created_at,
                    "Message": ticket.message,
                    "Status": ticket.status,
                    "Intent": getattr(ticket, "intent", None),
                    "Priority": getattr(ticket, "priority", None),
                    "Confidence": ticket.confidence,
                    "Escalated": ticket.escalated,
                })

            st.dataframe(
                pd.DataFrame(data),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        st.error(f"Database error: {exc}")


# ============================================================
# ANALYTICS
# ============================================================

with tabs[5]:
    st.subheader("Analytics")

    try:
        session, repository = get_repository()

        try:
            analytics = repository.get_analytics_summary()
        finally:
            session.close()

        cols = st.columns(3)

        cols[0].metric(
            "Total tickets",
            analytics["total_tickets"],
        )

        cols[1].metric(
            "Escalated",
            analytics["escalated"],
        )

        cols[2].metric(
            "Avg intent confidence",
            analytics["average_confidence"],
        )

        distributions = [
            ("Intent distribution", "intent_distribution"),
            ("Sentiment distribution", "sentiment_distribution"),
            ("Priority distribution", "priority_distribution"),
            ("Department distribution", "department_distribution"),
        ]

        for title, key in distributions:
            st.write(f"**{title}**")

            values = analytics.get(key, {})

            if values:
                st.bar_chart(pd.Series(values))
            else:
                st.info("No data available.")

    except Exception as exc:
        st.error(f"Analytics error: {exc}")


# ============================================================
# MODEL COMPARISON
# ============================================================

with tabs[6]:
    st.subheader("TF-IDF vs Transformer")

    report_dir = PROJECT_ROOT / "artifacts" / "evaluation_reports"

    reports = [
        ("classical_model_report.json", "TF-IDF + Logistic Regression"),
        ("transformer_model_report.json", "DistilBERT"),
    ]

    rows = []

    for filename, model_name in reports:
        path = report_dir / filename

        if path.exists():
            data = json.loads(
                path.read_text(encoding="utf-8")
            )

            metrics = data.get("global_metrics", {})

            rows.append({
                "Model": model_name,
                **metrics,
            })

    if rows:
        frame = pd.DataFrame(rows)

        columns = [
            "Model",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "macro_f1",
            "weighted_f1",
        ]

        columns = [
            column
            for column in columns
            if column in frame.columns
        ]

        st.dataframe(
            frame[columns],
            use_container_width=True,
            hide_index=True,
        )

        if "accuracy" in frame.columns:
            st.write("**Accuracy comparison**")
            st.bar_chart(
                frame.set_index("Model")["accuracy"]
            )
    else:
        st.info(
            "No measured comparison results yet. "
            "Run both training/evaluation pipelines."
        )


# ============================================================
# RAG EVALUATION
# ============================================================

with tabs[7]:
    st.subheader("RAG Evaluation")

    path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation_reports"
        / "rag_retrieval_report.json"
    )

    if path.exists():
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        metrics = data.get(
            "metrics",
            data.get("global_metrics", {}),
        )

        if metrics:
            cols = st.columns(3)

            for col, key in zip(
                cols,
                [
                    "recall_at_k",
                    "precision_at_k",
                    "mrr",
                ],
            ):
                if key in metrics:
                    col.metric(
                        key.replace("_", " ").title(),
                        metrics[key],
                    )

        with st.expander(
            "Evaluation details",
            expanded=True,
        ):
            st.json(data)

    else:
        st.info(
            "Run scripts/evaluate.py after building the FAISS index."
        )