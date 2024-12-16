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

    # Identify top values
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
    # Extra colors
    extra_colors = ["#a6cee3","#b2df8a","#fb9a99","#fdbf6f","#cab2d6"]

    # Helper functions to create charts
    def make_bar_chart(data, x_label, y_label, title):
        chart_data = data.reset_index()
        chart_data.columns = [x_label, y_label]
        chart = alt.Chart(chart_data).mark_bar(color=primary_color).encode(
            x=alt.X(x_label, sort='-y'),
            y=y_label,
            tooltip=[x_label, y_label]
        ).properties(title=title)
        return chart

    def make_pie_chart(counts, title):
        pie_data = counts.reset_index()
        pie_data.columns = ['category', 'count']
        domain = pie_data['category'].tolist()
        color_range = [primary_color, secondary_color] + extra_colors[:max(0, len(domain)-2)]
        chart = alt.Chart(pie_data).mark_arc().encode(
            theta='count:Q',
            color=alt.Color('category:N', scale=alt.Scale(domain=domain, range=color_range), legend=None),
            tooltip=['category', 'count']
        ).properties(title=title)
        return chart

    # Time series from report_datetime if available
    timeseries_chart = None
    if 'report_datetime' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['report_datetime']):
            df['report_datetime'] = pd.to_datetime(df['report_datetime'], errors='coerce')

        # If conversion successful and we have valid datetimes
        if df['report_datetime'].notna().any():
            # Aggregate by date (ignoring time)
            df['report_date'] = df['report_datetime'].dt.date
            daily_counts = df.groupby('report_date').size().reset_index(name='count')
            timeseries_chart = alt.Chart(daily_counts).mark_line(color=primary_color).encode(
                x='report_date:T',
                y='count:Q',
                tooltip=['report_date:T', 'count:Q']
            ).properties(title="Incidents Over Time (Daily)")

    # Map chart if we have lat/lon
    map_chart = None
    if 'incident_lat' in df.columns and 'incident_lang' in df.columns:
        map_data = df[['incident_lat', 'incident_lang']].dropna()
        if not map_data.empty:
            map_chart = map_data.rename(columns={'incident_lat':'lat','incident_lang':'lon'})

    # Create charts
    station_chart = make_bar_chart(top_stations, 'reporting_station', 'count', "Top 5 Reporting Stations") if not top_stations.empty else None
    charge_chart = make_bar_chart(top_charges, 'charge_type', 'count', "Top 5 Charge Types") if not top_charges.empty else None
    nationality_chart = make_bar_chart(top_nationalities, 'accused_nationality', 'count', "Top 5 Nationalities") if not top_nationalities.empty else None
    classification_chart = make_pie_chart(top_classifications, "Top 5 Classifications") if not top_classifications.empty else None

    # Updated prompt to include placeholders for map and timeseries
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
    - Time series available: {"Yes" if timeseries_chart else "No"}
    - Map data available: {"Yes" if map_chart is not None else "No"}

    In your response:
    - Provide a very thorough, detailed, and lengthy analysis that goes beyond generic statements.
    - Highlight subtle patterns, anomalies, or rare insights not obvious at first glance.
    - Integrate these visual findings into the narrative.
    - Include the following placeholders where you'd like the charts to appear:
      - <<CHART_STATIONS>> for Top 5 Reporting Stations bar chart
      - <<CHART_CLASSIFICATIONS>> for Top 5 Classifications pie chart
      - <<CHART_CHARGES>> for Top 5 Charge Types bar chart
      - <<CHART_NATIONALITIES>> for Top 5 Nationalities bar chart
      - <<CHART_MAP>> for the map of incidents (if available)
      - <<CHART_TIMESERIES>> for the line chart of incidents over time (if available)

    Make the explanation rich in detail, non-generic, and insightful. Leverage the data to draw in-depth conclusions,
    even if speculative, about what these distributions and trends might mean in a real-world crime analysis context.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and detail-oriented data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=15000
    )

    summary_text = response.choices[0].message["content"].strip()

    container = st.container()
    final_text = summary_text

    # Function to insert Altair charts
    def insert_alt_chart(text, placeholder, chart):
        if chart is not None and placeholder in text:
            parts = text.split(placeholder, 1)
            container.write(parts[0])
            container.altair_chart(chart, use_container_width=True)
            return parts[1]
        return text

    # Function to insert map
    def insert_map(text, placeholder, map_data):
        if map_data is not None and placeholder in text:
            parts = text.split(placeholder, 1)
            container.write(parts[0])
            container.map(map_data)
            return parts[1]
        return text

    # Insert charts in order
    placeholders = [
        ("<<CHART_STATIONS>>", station_chart, 'alt'),
        ("<<CHART_CLASSIFICATIONS>>", classification_chart, 'alt'),
        ("<<CHART_CHARGES>>", charge_chart, 'alt'),
        ("<<CHART_NATIONALITIES>>", nationality_chart, 'alt'),
        ("<<CHART_MAP>>", map_chart, 'map'),
        ("<<CHART_TIMESERIES>>", timeseries_chart, 'alt')
    ]

    for ph, ch, ctype in placeholders:
        if ph in final_text:
            if ctype == 'alt':
                final_text = insert_alt_chart(final_text, ph, ch)
            elif ctype == 'map':
                final_text = insert_map(final_text, ph, ch)

    # After all placeholders processed, write the remainder of the text
    container.write(final_text)
