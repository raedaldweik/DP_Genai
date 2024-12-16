import os
import random
import pandas as pd
import streamlit as st
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from dotenv import load_dotenv
from langdetect import detect
import altair as alt
import openai
from audio_recorder_streamlit import audio_recorder
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

# NEW IMPORTS FOR SQL AGENT
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

st.set_page_config(page_title="Crime Analytics Assistant", layout="wide")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_data():
    return pd.read_excel("crime_data.xlsx")  # Ensure this file exists and path is correct

df = load_data()

SUGGESTED_PROMPTS_POOL = [
    "Assess the level of crime in each reporting station",
    "Show me a bar chart of accused_age",
    "Describe the distribution of incidents by nationality",
    "Show me a map of incidents",
    "Summarize the most common charge types",
    "Which classification has the highest number of incidents?",
    "How many incidents are there in total?",
    "Show me a pie chart of classification",
    "Describe the top 5 reporting stations",
    "Compare the number of incidents between two stations"
]

def get_new_suggestions():
    return random.sample(SUGGESTED_PROMPTS_POOL, 4)

if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

if 'suggested_prompts' not in st.session_state:
    st.session_state.suggested_prompts = get_new_suggestions()

st.title("Crime Analytics Assistant")

# Sidebar Filters
st.sidebar.header("Filters")
all_nationalities = ["All"] + df["accused_nationality"].dropna().unique().tolist()
filter_nationality = st.sidebar.selectbox("Nationality:", all_nationalities)

all_stations = ["All"] + df["reporting_station"].dropna().unique().tolist()
filter_station = st.sidebar.selectbox("Reporting Station:", all_stations)

all_charge_types = ["All"] + df["charge_type"].dropna().unique().tolist()
filter_charge = st.sidebar.selectbox("Charge Type:", all_charge_types)

all_classifications = ["All"] + df["classification"].dropna().unique().tolist()
filter_classification = st.sidebar.selectbox("Classification:", all_classifications)

all_report_status = ["All"] + df["report_status"].dropna().unique().tolist()
filter_report_status = st.sidebar.selectbox("Report Status:", all_report_status)

# Apply filters to the dataset
filtered_df = df.copy()
if filter_nationality != "All":
    filtered_df = filtered_df[filtered_df["accused_nationality"] == filter_nationality]

if filter_station != "All":
    filtered_df = filtered_df[filtered_df["reporting_station"] == filter_station]

if filter_charge != "All":
    filtered_df = filtered_df[filtered_df["charge_type"] == filter_charge]

if filter_classification != "All":
    filtered_df = filtered_df[filtered_df["classification"] == filter_classification]

if filter_report_status != "All":
    filtered_df = filtered_df[filtered_df["report_status"] == filter_report_status]

# Prompt bar with audio icon aligned
input_container = st.container()
with input_container:
    prompt_col1, prompt_col2 = st.columns([0.95, 0.05])
    with prompt_col1:
        user_query = st.text_input("", value=st.session_state.user_query, placeholder="Ask your question...")
    with prompt_col2:
        st.markdown("<style>div[data-testid='stHorizontalBlock'] > div {display:flex; align-items:center; justify-content:center;}</style>", unsafe_allow_html=True)
        record_audio = st.button("🎙️", help="Record your question")

audio_bytes = None
if record_audio:
    st.info("Recording... Click stop when done.")
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.success("Recorded successfully! Processing...")

processed_query = None
if user_query.strip():
    processed_query = user_query
elif audio_bytes:
    if audio_bytes:
        r = sr.Recognizer()
        with sr.AudioFile(BytesIO(audio_bytes)) as source:
            audio = r.record(source)
        try:
            recognized_text = r.recognize_google(audio, language="en-US")
            processed_query = recognized_text
        except:
            processed_query = ""

st.write("**Suggested Prompts:**")
for prompt in st.session_state.suggested_prompts:
    if st.button(prompt):
        st.session_state.user_query = prompt
        st.experimental_rerun()

def detect_language(query):
    try:
        return detect(query)
    except:
        return "en"

def identify_chart_request(query):
    q = query.lower()
    if "bar chart" in q or "مخطط أعمدة" in q:
        return "bar"
    if "line chart" in q or "مخطط خطي" in q:
        return "line"
    if "pie chart" in q or "مخطط دائري" in q:
        return "pie"
    if "scatter plot" in q or "مخطط مبعثر" in q:
        return "scatter"
    if "map" in q or "خريطة" in q:
        return "map"
    return None

# Color scheme
PRIMARY_COLOR = "#3da47d"
SECONDARY_COLOR = "gray"

def create_bar_chart(data: pd.Series, title: str):
    chart_data = data.reset_index()
    chart_data.columns = ['category', 'count']
    return alt.Chart(chart_data).mark_bar(color=PRIMARY_COLOR).encode(
        x=alt.X('category:N', sort='-y'),
        y='count:Q',
        tooltip=['category', 'count']
    ).properties(title=title)

def create_line_chart(data: pd.Series, title: str):
    chart_data = pd.DataFrame({'value': data.values, 'index': range(len(data))})
    return alt.Chart(chart_data).mark_line(color=PRIMARY_COLOR).encode(
        x='index:Q',
        y='value:Q',
        tooltip=['index', 'value']
    ).properties(title=title)

def create_pie_chart(counts: pd.Series, title: str):
    pie_data = pd.DataFrame({'category': counts.index, 'count': counts.values})
    domain = pie_data['category'].tolist()
    color_range = [PRIMARY_COLOR, SECONDARY_COLOR]
    return alt.Chart(pie_data).mark_arc().encode(
        theta='count:Q',
        color=alt.Color('category:N', scale=alt.Scale(domain=domain, range=color_range), legend=None),
        tooltip=['category', 'count']
    ).properties(title=title)

def create_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    return alt.Chart(df).mark_circle(color=PRIMARY_COLOR).encode(
        x=alt.X(x_col, type='quantitative'),
        y=alt.Y(y_col, type='quantitative'),
        tooltip=[x_col, y_col]
    ).properties(title=title).interactive()

class CustomPandasAgent:
    def __init__(self, llm, df):
        self.llm = llm
        engine = create_engine("sqlite:///crime.db")
        self.db = SQLDatabase(engine=engine)
        # Use OPENAI_FUNCTIONS agent type
        self.sql_agent = create_sql_agent(
            llm=self.llm, 
            db=self.db, 
            agent_type=AgentType.OPENAI_FUNCTIONS, 
            verbose=False
        )

    def run(self, query: str) -> str:
        # Use run() instead of invoke()
        response = self.sql_agent.run(query)
        return response.strip()

def ask_agent(query, df, lang):
    user_prompt_lang = "Please respond in Arabic." if lang == "ar" else "Please respond in English."
    detailed_instructions = (
        "Please provide a very detailed, thorough, and explanatory answer. "
        "Include insights, interpretations, and contextual understanding, "
        "and elaborate on the reasoning behind the results."
    )
    full_query = f"{user_prompt_lang}\n\n{detailed_instructions}\n\n{query}"

    llm = ChatOpenAI(openai_api_key=openai.api_key, model_name="gpt-4o", temperature=0)
    agent = CustomPandasAgent(llm, df)
    response = agent.run(full_query)
    return response

if processed_query:
    query_lang = detect_language(processed_query)
    st.write(f"Your question is: {processed_query}")

    chart_type = identify_chart_request(processed_query)

    if chart_type is None:
        answer = ask_agent(processed_query, filtered_df, query_lang)
        st.write(answer)
    else:
        known_columns = list(filtered_df.columns)
        mentioned_col = None
        for c in known_columns:
            if c.lower() in processed_query.lower():
                mentioned_col = c
                break

        if chart_type == "map":
            if 'incident_lat' in filtered_df.columns and 'incident_lang' in filtered_df.columns:
                map_data = filtered_df[['incident_lat', 'incident_lang']].dropna()
                if not map_data.empty:
                    st.map(map_data.rename(columns={'incident_lat':'lat','incident_lang':'lon'}))
                else:
                    st.write("No valid data for the map.")
            else:
                st.write("No geographic data found.")
        else:
            if mentioned_col is None:
                col_query = f"Which column should I use to create a {chart_type} chart based on the user's query: {processed_query}? Just provide the column name."
                col_answer = ask_agent(col_query, filtered_df, query_lang)
                for c in known_columns:
                    if c.lower() in col_answer.lower():
                        mentioned_col = c
                        break
            if mentioned_col is None:
                st.write("I couldn't determine the column to plot. Please specify a column name in your query.")
            else:
                counts = filtered_df[mentioned_col].value_counts()
                if chart_type == "bar":
                    chart = create_bar_chart(counts, f"Bar Chart of {mentioned_col}")
                    st.altair_chart(chart, use_container_width=True)
                elif chart_type == "line":
                    if pd.api.types.is_numeric_dtype(filtered_df[mentioned_col]):
                        chart = create_line_chart(filtered_df[mentioned_col].dropna(), f"Line Chart of {mentioned_col}")
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.write("The chosen column is not numeric, cannot produce a meaningful line chart.")
                elif chart_type == "pie":
                    pie_chart = create_pie_chart(counts, f"Pie Chart of {mentioned_col}")
                    st.altair_chart(pie_chart, use_container_width=True)
                elif chart_type == "scatter":
                    numeric_columns = filtered_df.select_dtypes(include=['int64','float64']).columns.tolist()
                    if mentioned_col not in numeric_columns:
                        st.write("The chosen column is not numeric, cannot produce a scatter plot.")
                    else:
                        other_nums = [col for col in numeric_columns if col != mentioned_col]
                        if not other_nums:
                            st.write("Not enough numeric columns for a scatter plot.")
                        else:
                            second_column = other_nums[0]
                            scatter_df = filtered_df.dropna(subset=[mentioned_col, second_column])
                            scatter_chart = create_scatter_chart(scatter_df, mentioned_col, second_column, f"Scatter Plot of {mentioned_col} vs {second_column}")
                            st.altair_chart(scatter_chart, use_container_width=True)

    st.session_state.suggested_prompts = get_new_suggestions()
