import altair as alt
import pandas as pd
import streamlit as st

from src import data_loader
from src.config import (
    CUSTOMERS_CSV,
    DEFAULT_MODEL_NAME,
    PRODUCT_CATALOG_CSV,
    PURCHASE_HISTORY_CSV,
    VALID_STYLES,
)
from src.email_generator import EmailGenerator
from src.pipeline import generate_campaign
from src.preprocessing import build_customer_profiles, segment_customers

st.set_page_config(page_title="Everdale Email Studio", page_icon="✉️", layout="wide")

SENTIMENT_COLORS = {"Positive": "#0ca30c", "Neutral": "#898781", "Negative": "#d03b3b"}
SEGMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BASE_BLUE = "#2a78d6"


@st.cache_resource(show_spinner=False)
def load_generator(model_name: str) -> EmailGenerator:
    return EmailGenerator(model_name=model_name)


@st.cache_data
def load_bundled_sample_data():
    customers_df = data_loader.load_customer_data(CUSTOMERS_CSV)
    purchases_df = data_loader.load_purchase_data(PURCHASE_HISTORY_CSV)
    catalog_df = data_loader.load_product_catalog(PRODUCT_CATALOG_CSV)
    return customers_df, purchases_df, catalog_df


def render_email_card(record):
    st.markdown(f"#### {record['subject']}")
    st.caption(f"To: {record['first_name']} ({record['city']}) - Style: {record['style'].title()}")
    st.write(record["greeting"])
    st.write(record["body"])
    st.markdown(f"**You might also like:** {record['recommendation']}")
    st.info(record["offer"])
    st.button(record["cta"], key=f"cta_{record['customer_id']}", disabled=True)


def load_data_from_sidebar():
    st.subheader("Data source")
    use_uploaded = st.checkbox("Upload my own CSVs instead of the sample data", value=False)

    if not use_uploaded:
        return load_bundled_sample_data()

    customers_file = st.file_uploader("Customer data (CSV)", type="csv")
    purchases_file = st.file_uploader("Purchase history (CSV)", type="csv")

    if not customers_file or not purchases_file:
        st.caption("Using bundled sample data until both files are uploaded.")
        return load_bundled_sample_data()

    try:
        customers_df = data_loader.load_customer_data(customers_file)
        purchases_df = data_loader.load_purchase_data(purchases_file)
        catalog_df = data_loader.load_product_catalog(PRODUCT_CATALOG_CSV)
        return customers_df, purchases_df, catalog_df
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Couldn't read your files: {exc}")
        st.stop()


def main():
    st.title("Everdale Email Studio")
    st.caption("Personalized marketing email generator, powered by a local open-source LLM.")

    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input(
            "Hugging Face model",
            value=DEFAULT_MODEL_NAME,
            help="Any instruction-tuned causal LM works - Phi-3, Gemma, Mistral and Llama instruct variants have all been tested.",
        )
        style = st.selectbox("Writing style", VALID_STYLES, index=1, format_func=str.title)
        max_customers = st.slider(
            "Customers to process in batch mode", 1, 40, 5,
            help="Local models are slow on a CPU, so batch mode is capped by default. Raise it if you have a GPU.",
        )
        st.divider()
        customers_df, purchases_df, catalog_df = load_data_from_sidebar()

    profiles_df = build_customer_profiles(customers_df, purchases_df)

    tab_single, tab_batch, tab_analytics = st.tabs(["Single customer", "Batch generate", "Analytics"])

    with tab_single:
        st.subheader("Generate one email")
        customer_labels = profiles_df["customer_id"] + " - " + profiles_df["first_name"] + " " + profiles_df["last_name"]
        selected_label = st.selectbox("Pick a customer", customer_labels)
        selected_id = selected_label.split(" - ")[0]

        if st.button("Generate email", type="primary"):
            with st.spinner("Loading the model and writing the email - first run can take a while..."):
                generator = load_generator(model_name)
                one_customer_df = customers_df[customers_df["customer_id"] == selected_id]
                try:
                    records = generate_campaign(one_customer_df, purchases_df, catalog_df, style, generator)
                except RuntimeError as exc:
                    st.error(str(exc))
                    records = []

            if records:
                record = records[0]
                with st.container(border=True):
                    render_email_card(record)

                score_cols = st.columns(4)
                score_cols[0].metric("Word count", record["word_count"])
                score_cols[1].metric("Readability", record["readability_score"])
                score_cols[2].metric("Sentiment", record["sentiment_label"])
                score_cols[3].metric("Personalization", f"{record['personalization_score']}%")

    with tab_batch:
        st.subheader("Generate for multiple customers")
        st.write(f"This will process the first **{max_customers}** customers using the **{style}** style.")

        if st.button("Generate emails for all customers", type="primary"):
            progress_bar = st.progress(0.0, text="Starting up...")

            def update_progress(done, total):
                progress_bar.progress(done / total, text=f"Generated {done}/{total} emails")

            generator = load_generator(model_name)
            try:
                records = generate_campaign(
                    customers_df, purchases_df, catalog_df, style, generator,
                    limit=max_customers, progress_callback=update_progress,
                )
                st.session_state["batch_records"] = records
            except RuntimeError as exc:
                st.error(str(exc))

            progress_bar.empty()

        if "batch_records" in st.session_state:
            results_df = pd.DataFrame(st.session_state["batch_records"])
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download as CSV", data=csv_bytes, file_name="generated_emails.csv", mime="text/csv")

    with tab_analytics:
        st.subheader("Customer segments")
        st.caption("Rough KMeans grouping on age, total spend and order count - just to see if natural segments show up.")

        segmented_df = segment_customers(profiles_df)
        segment_chart = alt.Chart(segmented_df).mark_circle(size=90, opacity=0.8).encode(
            x=alt.X("age:Q", title="Age"),
            y=alt.Y("total_spent:Q", title="Total spent"),
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=[0, 1, 2], range=SEGMENT_COLORS),
                legend=alt.Legend(title="Segment"),
            ),
            tooltip=["first_name", "age", "total_spent", "total_orders", "loyalty_tier"],
        )
        st.altair_chart(segment_chart, use_container_width=True)

        st.divider()
        st.subheader("Generated campaign metrics")

        if "batch_records" not in st.session_state:
            st.info("Generate a batch of emails in the 'Batch generate' tab to see campaign metrics here.")
            return

        results_df = pd.DataFrame(st.session_state["batch_records"])

        metric_cols = st.columns(4)
        metric_cols[0].metric("Avg. word count", round(results_df["word_count"].mean(), 1))
        metric_cols[1].metric("Avg. readability", round(results_df["readability_score"].mean(), 1))
        metric_cols[2].metric("Avg. sentiment", round(results_df["sentiment_polarity"].mean(), 3))
        metric_cols[3].metric("Avg. personalization", f"{round(results_df['personalization_score'].mean(), 1)}%")

        left, right = st.columns(2)

        with left:
            st.markdown("**Sentiment distribution**")
            sentiment_counts = results_df["sentiment_label"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment_label", "count"]
            chart = alt.Chart(sentiment_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("sentiment_label:N", title=None, sort=list(SENTIMENT_COLORS.keys())),
                y=alt.Y("count:Q", title="Emails"),
                color=alt.Color(
                    "sentiment_label:N",
                    scale=alt.Scale(domain=list(SENTIMENT_COLORS.keys()), range=list(SENTIMENT_COLORS.values())),
                    legend=None,
                ),
                tooltip=["sentiment_label", "count"],
            )
            st.altair_chart(chart, use_container_width=True)

        with right:
            st.markdown("**Personalization score by customer**")
            chart = alt.Chart(results_df).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=BASE_BLUE
            ).encode(
                x=alt.X("first_name:N", title=None, sort="-y"),
                y=alt.Y("personalization_score:Q", title="Score"),
                tooltip=["first_name", "personalization_score"],
            )
            st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
