import os
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
import openai

st.set_page_config(page_title="Data Analytics Summary", layout="wide")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_data():
    return pd.read_excel("crime_data.xlsx")

df = load_data()

st.title("Comprehensive Data Analytics Summary")

st.write("""
This application provides a detailed and data-driven narrative summary of the underlying crime dataset, 
offering valuable insights, highlighting notable patterns, and revealing rare findings.
""")

if st.button("Generate Summary Report"):
    # Gather basic info
    num_rows = len(df)
    num_columns = len(df.columns)
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Get top values safely
    top_stations = df['reporting_station'].value_counts(dropna=True).head(5) if 'reporting_station' in df.columns else pd.Series([], dtype=int)
    top_classifications = df['classification'].value_counts(dropna=True).head(5) if 'classification' in df.columns else pd.Series([], dtype=int)
    top_charges = df['charge_type'].value_counts(dropna=True).head(5) if 'charge_type' in df.columns else pd.Series([], dtype=int)
    top_nationalities = df['accused_nationality'].value_counts(dropna=True).head(5) if 'accused_nationality' in df.columns else pd.Series([], dtype=int)
    age_stats = df['accused_age'].describe() if 'accused_age' in df.columns and pd.api.types.is_numeric_dtype(df['accused_age']) else None

    rare_categories_info = {}
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=True)
        rare = counts[counts < 5].index.tolist()
        if rare:
            rare_categories_info[col] = rare

    # Define colors
    primary_color = "#3da47d"
    secondary_color = "gray"
    # Extra colors for charts with more categories
    extra_colors = ["#a6cee3","#b2df8a","#fb9a99","#fdbf6f","#cab2d6"]

    # Function to create a bar chart with a single or multiple categories
    def make_bar_chart(data, x_label, y_label, title, single_color=True):
        chart_data = data.reset_index()
        chart_data.columns = [x_label, y_label]
        if single_color:
            chart = alt.Chart(chart_data).mark_bar(color=primary_color).encode(
                x=alt.X(x_label, sort='-y'),
                y=y_label,
                tooltip=[x_label, y_label]
            ).properties(title=title)
        else:
            # For categorical bars with multiple categories, use a custom scale
            # But since these top_* are single-dimension counts, single_color suffices.
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(x_label, sort='-y'),
                y=y_label,
                tooltip=[x_label, y_label],
                color=alt.value(primary_color)
            ).properties(title=title)
        return chart

    # Charts
    station_chart = make_bar_chart(top_stations, 'reporting_station', 'count', "Top 5 Reporting Stations") if not top_stations.empty else None
    charge_chart = make_bar_chart(top_charges, 'charge_type', 'count', "Top 5 Charge Types") if not top_charges.empty else None
    nationality_chart = make_bar_chart(top_nationalities, 'accused_nationality', 'count', "Top 5 Nationalities") if not top_nationalities.empty else None

    # Pie chart for classifications with custom color range
    if not top_classifications.empty:
        classification_data = top_classifications.reset_index()
        classification_data.columns = ['classification', 'count']
        domain = classification_data['classification'].tolist()
        # Construct a color range with primary, secondary, and extras
        color_range = [primary_color, secondary_color] + extra_colors[:max(0, len(domain)-2)]
        classification_chart = alt.Chart(classification_data).mark_arc().encode(
            theta='count:Q',
            color=alt.Color('classification:N', scale=alt.Scale(domain=domain, range=color_range)),
            tooltip=['classification', 'count']
        ).properties(title="Top 5 Classifications")
    else:
        classification_chart = None

    # Prompt with placeholders for chart insertion
    prompt = f"""
    You are a data expert analyzing a crime dataset with {num_rows} rows and {num_columns} columns.
    Categorical columns: {categorical_cols}
    Numerical columns: {numerical_cols}

    Findings:
    - Top 5 reporting stations: {top_stations.to_dict() if not top_stations.empty else "None"}
    - Top 5 classifications: {top_classifications.to_dict() if not top_classifications.empty else "None"}
    - Top 5 charge types: {top_charges.to_dict() if not top_charges.empty else "None"}
    - Nationalities: {top_nationalities.to_dict() if not top_nationalities.empty else "None"}
    - Age stats: {age_stats.to_dict() if age_stats is not None else "None"}
    - Rare categories: {rare_categories_info if rare_categories_info else "None"}

    In your response:
    - Provide a very thorough, detailed, and lengthy analysis that goes beyond generic statements.
    - Highlight subtle patterns, anomalies, or rare insights not obvious at first glance.
    - Integrate the concept of these visual findings into the narrative.
    - Include the following placeholders in your narrative where you'd like the charts to appear:
      - <<CHART_STATIONS>> for Top 5 Reporting Stations bar chart
      - <<CHART_CLASSIFICATIONS>> for Top 5 Classifications pie chart
      - <<CHART_CHARGES>> for Top 5 Charge Types bar chart
      - <<CHART_NATIONALITIES>> for Top 5 Nationalities bar chart
    - Use these placeholders exactly (e.g., "<<CHART_STATIONS>>") so we can insert the charts in those spots.

    Make the explanation rich in detail, non-generic, and insightful. Leverage the data to draw in-depth conclusions,
    even if speculative, about what these distributions imply in a real-world crime analysis context.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful and detail-oriented data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=7000
    )

    summary_text = response.choices[0].message["content"].strip()

    # Now we will display the summary text and insert charts where placeholders appear
    # We’ll split the text by the placeholders and insert charts accordingly.
    def insert_chart(text, placeholder, chart):
        if chart is not None and placeholder in text:
            parts = text.split(placeholder)
            return parts[0], parts[1], chart
        return None

    # We'll handle this step by step:
    container = st.container()
    final_text = summary_text

    # Process each placeholder in order
    for placeholder, chart in [
        ("<<CHART_STATIONS>>", station_chart),
        ("<<CHART_CLASSIFICATIONS>>", classification_chart),
        ("<<CHART_CHARGES>>", charge_chart),
        ("<<CHART_NATIONALITIES>>", nationality_chart),
    ]:
        if placeholder in final_text and chart is not None:
            before, after = final_text.split(placeholder, 1)
            # Write the text before the placeholder
            container.write(before)
            # Show the chart
            container.altair_chart(chart, use_container_width=True)
            # Continue with the text after the placeholder
            final_text = after

    # After processing all placeholders, write any remaining text
    container.write(final_text)
