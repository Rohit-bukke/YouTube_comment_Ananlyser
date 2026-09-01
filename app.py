"""
Interactive Streamlit Web Dashboard for YouTube Sentiment Analysis & NLP Intelligence.
"""

import json
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.predictor import SentimentPredictor
from src.services.youtube_service import YouTubeService
from src.utils.config_manager import load_config
from src.utils.logger import get_logger

# Page configuration
st.set_page_config(
    page_title="YouTube Sentiment & Audience Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .sentiment-pos {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .sentiment-neu {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .sentiment-neg {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor_instance():
    config = load_config()
    return SentimentPredictor(config=config)


@st.cache_data
def load_metrics_data():
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


predictor = get_predictor_instance()
metrics_data = load_metrics_data()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/youtube-play.png", width=64)
st.sidebar.title("NLP Intelligence")
st.sidebar.markdown("Production-grade sentiment analysis powered by **Calibrated Linear Support Vector Machine** with sublinear TF-IDF features.")

if metrics_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 Model Status")
    st.sidebar.success(f"**Best Model**: {metrics_data.get('best_model', 'N/A')}")
    st.sidebar.metric(
        label="Macro F1-Score",
        value=f"{metrics_data.get('best_score', 0):.2%}",
    )
    st.sidebar.metric(
        label="Trained Samples",
        value=f"{metrics_data.get('train_samples', 0):,}",
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Navigate tabs below to test live YouTube URLs, single comments, batch CSV datasets, or view model benchmarks.")

# App Header
st.markdown("<div class='main-title'>📊 YouTube Sentiment & Audience Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time comment classification, emotion polarity indexing, and audience analytics micro-platform.</div>", unsafe_allow_html=True)

# Tabs
tab_yt, tab_single, tab_batch, tab_benchmarks = st.tabs([
    "📺 Live YouTube Analyzer",
    "💬 Single Comment Tester",
    "📁 Batch File Processing",
    "📈 Model Benchmarks & Metrics",
])


# ==========================================
# TAB 1: LIVE YOUTUBE ANALYZER
# ==========================================
with tab_yt:
    st.markdown("### 📺 Analyze YouTube Video Sentiment")
    st.markdown("Enter any YouTube video link to automatically fetch audience comments and analyze overall viewer reception.")

    col1, col2 = st.columns([3, 1])
    with col1:
        yt_url = st.text_input(
            "YouTube Video URL or ID:",
            value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            placeholder="https://www.youtube.com/watch?v=...",
        )
    with col2:
        max_comments = st.slider("Max Comments to Analyze:", min_value=10, max_value=200, value=50, step=10)

    api_key = st.text_input("Optional YouTube Data API v3 Key (Leave blank to use built-in smart extractor):", type="password")

    if st.button("🚀 Analyze Video Comments", type="primary"):
        if not predictor.is_ready:
            st.error("⚠️ Model is not loaded. Please train the model first by running `python -m src.models.trainer`.")
        else:
            with st.spinner("Extracting and analyzing YouTube comments..."):
                try:
                    yt_service = YouTubeService(api_key=api_key if api_key else None)
                    video_id = yt_service.extract_video_id(yt_url)

                    if not video_id:
                        st.error("Invalid YouTube URL. Please verify the URL format.")
                    else:
                        comments_data = yt_service.get_comments(yt_url, max_comments=max_comments)
                        if not comments_data:
                            st.warning("No comments could be retrieved for this video.")
                        else:
                            texts = [c["text"] for c in comments_data]
                            predictions = predictor.predict_batch(texts)

                            for c_dict, pred in zip(comments_data, predictions):
                                c_dict["cleaned_text"] = pred["cleaned_text"]
                                c_dict["prediction"] = pred["prediction"]
                                c_dict["sentiment"] = pred["label"]
                                c_dict["confidence"] = pred["confidence"]

                            df_results = pd.DataFrame(comments_data)

                            # Metrics Summary
                            total_c = len(df_results)
                            pos_count = int((df_results["sentiment"] == "Positive").sum())
                            neu_count = int((df_results["sentiment"] == "Neutral").sum())
                            neg_count = int((df_results["sentiment"] == "Negative").sum())

                            pos_pct = (pos_count / total_c) * 100
                            neu_pct = (neu_count / total_c) * 100
                            neg_pct = (neg_count / total_c) * 100
                            positivity_idx = ((pos_count - neg_count) / total_c) * 100

                            st.success(f"✅ Successfully analyzed **{total_c}** comments for Video ID: `{video_id}`")

                            # KPI Cards
                            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                            kpi1.metric("Positivity Index", f"{positivity_idx:+.1f}%", help="Calculated as (Positive - Negative) / Total")
                            kpi2.metric("🟢 Positive Comments", f"{pos_count} ({pos_pct:.1f}%)")
                            kpi3.metric("🟡 Neutral Comments", f"{neu_count} ({neu_pct:.1f}%)")
                            kpi4.metric("🔴 Negative Comments", f"{neg_count} ({neg_pct:.1f}%)")

                            st.markdown("---")

                            # Charts
                            col_chart1, col_chart2 = st.columns(2)
                            with col_chart1:
                                st.markdown("#### Sentiment Distribution")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                colors = ["#22C55E", "#F59E0B", "#EF4444"]
                                labels = ["Positive", "Neutral", "Negative"]
                                counts = [pos_count, neu_count, neg_count]
                                ax.pie(
                                    counts,
                                    labels=labels,
                                    autopct="%1.1f%%",
                                    startangle=140,
                                    colors=colors,
                                    wedgeprops={"edgecolor": "white", "linewidth": 2},
                                )
                                ax.axis("equal")
                                st.pyplot(fig)
                                plt.close(fig)

                            with col_chart2:
                                st.markdown("#### Top Liked Audience Comments")
                                top_pos = df_results[df_results["sentiment"] == "Positive"].sort_values("likes", ascending=False)
                                top_neg = df_results[df_results["sentiment"] == "Negative"].sort_values("likes", ascending=False)

                                if not top_pos.empty:
                                    st.success(f"**Top Positive Comment** (+{top_pos.iloc[0]['likes']} 👍):\n\n*\"{top_pos.iloc[0]['text']}\"* — @{top_pos.iloc[0]['author']}")
                                if not top_neg.empty:
                                    st.error(f"**Top Negative Comment** (+{top_neg.iloc[0]['likes']} 👍):\n\n*\"{top_neg.iloc[0]['text']}\"* — @{top_neg.iloc[0]['author']}")

                            # Data Table
                            st.markdown("#### Detailed Comments Breakdown")
                            filter_sentiment = st.multiselect(
                                "Filter by Sentiment:",
                                options=["Positive", "Neutral", "Negative"],
                                default=["Positive", "Neutral", "Negative"],
                            )
                            filtered_df = df_results[df_results["sentiment"].isin(filter_sentiment)]
                            st.dataframe(
                                filtered_df[["author", "sentiment", "confidence", "likes", "text"]],
                                use_container_width=True,
                                height=300,
                            )

                            # CSV Download
                            csv = filtered_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Download Analysis CSV",
                                data=csv,
                                file_name=f"youtube_sentiment_{video_id}.csv",
                                mime="text/csv",
                            )
                except Exception as e:
                    st.error(f"Analysis failed: {e}")


# ==========================================
# TAB 2: SINGLE COMMENT TESTER
# ==========================================
with tab_single:
    st.markdown("### 💬 Real-Time Comment Sentiment Tester")
    st.markdown("Type or paste any comment to see real-time NLP classification, confidence score, and probability distribution.")

    sample_col1, sample_col2, sample_col3 = st.columns(3)
    sample_text = ""
    if sample_col1.button("Sample 1: Positive Review"):
        sample_text = "This machine learning tutorial is absolutely wonderful and saved me hours of research!"
    if sample_col2.button("Sample 2: Constructive / Neutral"):
        sample_text = "Could you please clarify how the learning rate was chosen in the training script?"
    if sample_col3.button("Sample 3: Negative Critique"):
        sample_text = "Completely broken code, outdated dependencies and terribly explained."

    comment_input = st.text_area(
        "Enter Comment Text:",
        value=sample_text if sample_text else "This project is brilliantly engineered and production-ready!",
        height=100,
    )

    if st.button("🔍 Classify Sentiment", type="primary"):
        if not predictor.is_ready:
            st.error("Model is not loaded. Train the model first.")
        else:
            with st.spinner("Classifying..."):
                res = predictor.predict(comment_input)

                lbl = res["label"]
                conf = res["confidence"]
                probs = res["probabilities"]
                latency = res["latency_ms"]

                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.markdown("#### Prediction Result")
                    if lbl == "Positive":
                        st.markdown(f"<span class='sentiment-pos' style='font-size: 1.4rem;'>🟢 Positive</span>", unsafe_allow_html=True)
                    elif lbl == "Neutral":
                        st.markdown(f"<span class='sentiment-neu' style='font-size: 1.4rem;'>🟡 Neutral</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='sentiment-neg' style='font-size: 1.4rem;'>🔴 Negative</span>", unsafe_allow_html=True)

                    st.markdown(f"**Confidence**: `{conf:.2%}`")
                    st.markdown(f"**Inference Latency**: `{latency} ms`")
                    st.markdown(f"**Cleaned Tokens**: `{res['cleaned_text']}`")

                with col_res2:
                    st.markdown("#### Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Sentiment": list(probs.keys()),
                        "Probability": list(probs.values()),
                    })
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    sns.barplot(
                        x="Probability",
                        y="Sentiment",
                        data=prob_df,
                        palette=["#EF4444", "#F59E0B", "#22C55E"],
                        ax=ax,
                    )
                    ax.set_xlim(0, 1.0)
                    for p in ax.patches:
                        ax.annotate(f"{p.get_width():.1%}", (p.get_width() + 0.02, p.get_y() + 0.5), va="center")
                    st.pyplot(fig)
                    plt.close(fig)


# ==========================================
# TAB 3: BATCH FILE PROCESSING
# ==========================================
with tab_batch:
    st.markdown("### 📁 High-Throughput Batch Processing")
    st.markdown("Upload a CSV dataset of comments to run bulk sentiment inference and generate analytics.")

    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_upload)}** rows. Preview:")
            st.dataframe(df_upload.head(4), use_container_width=True)

            text_columns = list(df_upload.columns)
            selected_col = st.selectbox("Select the Text / Comment column:", text_columns)

            if st.button("⚡ Process Batch Dataset", type="primary"):
                with st.spinner("Processing comments in batch..."):
                    texts = df_upload[selected_col].astype(str).tolist()
                    batch_res = predictor.predict_batch(texts)

                    df_upload["predicted_sentiment"] = [r["label"] for r in batch_res]
                    df_upload["confidence"] = [r["confidence"] for r in batch_res]
                    df_upload["cleaned_text"] = [r["cleaned_text"] for r in batch_res]

                    st.success(f"Processed {len(texts)} comments in {batch_res[0].get('batch_latency_ms', 0):.2f} ms (avg {batch_res[0].get('avg_item_latency_ms', 0):.3f} ms/comment)!")

                    st.dataframe(df_upload.head(10), use_container_width=True)

                    csv_out = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Processed CSV",
                        data=csv_out,
                        file_name="processed_sentiment_dataset.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


# ==========================================
# TAB 4: MODEL BENCHMARKS & METRICS
# ==========================================
with tab_benchmarks:
    st.markdown("### 📈 Model Benchmarks & Experimental Results")
    st.markdown("Comprehensive statistical evaluation and multi-model benchmarking on 36,200+ real-world social comments.")

    if metrics_data:
        models_dict = metrics_data.get("models", {})
        table_rows = []
        for name, m in models_dict.items():
            lat = m.get("latency", {}).get("avg_latency_ms", 0.0)
            table_rows.append({
                "Model Architecture": name,
                "Accuracy": f"{m['accuracy']:.2%}",
                "Macro F1": f"{m['f1_macro']:.2%}",
                "Weighted F1": f"{m['f1_weighted']:.2%}",
                "Macro Precision": f"{m['precision_macro']:.2%}",
                "Macro Recall": f"{m['recall_macro']:.2%}",
                "Avg Latency (ms)": f"{lat:.3f} ms",
            })

        bench_df = pd.DataFrame(table_rows)
        st.dataframe(bench_df, use_container_width=True)

        best_model_name = metrics_data.get("best_model")
        if best_model_name and best_model_name in models_dict:
            best_metrics = models_dict[best_model_name]

            col_cm, col_report = st.columns(2)
            with col_cm:
                st.markdown(f"#### Confusion Matrix ({best_model_name})")
                cm = best_metrics["confusion_matrix"]
                target_names = best_metrics["target_names"]

                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=target_names,
                    yticklabels=target_names,
                    ax=ax_cm,
                )
                ax_cm.set_ylabel("True Sentiment")
                ax_cm.set_xlabel("Predicted Sentiment")
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with col_report:
                st.markdown("#### Classification Report Details")
                clf_report = best_metrics.get("classification_report", {})
                report_rows = []
                for label in target_names:
                    if label in clf_report:
                        d = clf_report[label]
                        report_rows.append({
                            "Class": label,
                            "Precision": f"{d['precision']:.2%}",
                            "Recall": f"{d['recall']:.2%}",
                            "F1-Score": f"{d['f1-score']:.2%}",
                            "Support": d["support"],
                        })
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
    else:
        st.warning("Benchmark metrics not found. Please run the training pipeline first.")
