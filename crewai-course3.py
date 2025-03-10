import streamlit as st
import subprocess
from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool
from crewai.tools import BaseTool
from enum import Enum
from dotenv import load_dotenv, find_dotenv
import os


class AGENT(Enum):
    DOC = "https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
    # ON = "https://service.opsnow.com/docs/console/en/user-guide-console-en.html"
    SERPER = "SERPERAPI"


# class LLM(Enum):
#     EXA7B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:latest"
#     EXA4B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:latest"
#     BLLOSSOM = "ollama/llama-3.2-korean-bllossom-3b-q8:latest"
#     DEEPSEEK = "ollama/deepseek-r1:1.5b"
#     DEEPSEEK8B = "ollama/deepseek-r1:8b"
#     UNCENSOR = "ollama/hf.co:/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF:latest"


class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = (
        "Analyzes the sentiment of text "
        "to ensure positive and engaging communication."
    )

    def _run(self, text: str) -> str:
        # Your custom code tool goes here
        return "positive"


def get_llms(tool="ollama"):
    try:
        result = subprocess.run(
            [tool, "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()

        models = []
        for line in lines[1:]:
            # Remove any leading/trailing whitespace
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                model_name = f"ollama/{parts[0]}"
                models.append(model_name)

        if models:
            return models
        else:
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error occured while running 'ollama list': {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occured: {e}")


# main
load_dotenv(find_dotenv(), override=True)

SERPER_KEY = os.getenv(AGENT.SERPER.value)

# set streamlit
st.set_page_config(layout="centered")
st.title("Tools for a Customer Outreach - CrewAI")

if models := get_llms():
    sales_model = st.selectbox("Sales Representative", models)
    lead_sales_model = st.selectbox("Lead Sales Representative", models)
else:
    st.error("Can't load any models.")
    exit()

lead_name = st.text_input(
    label="Lead Name", value="DeepLearningAI", max_chars=100
)
industry = st.text_input(
    label="Industry", value="Online Learning Platform", max_chars=100
)
decision_maker = st.text_input(
    label="Decision Maker", value="Andrew Ng", max_chars=100
)
position = st.text_input(label="Position", value="CEO", max_chars=100)
milestone = st.text_input(
    label="MileStone", value="product launch", max_chars=100
)
button = st.button("생성하기")

if button:
    if all([lead_name, industry, decision_maker, position, milestone]):
        sales_rep_agent = Agent(
            role="Sales Representative",
            goal="Identify high-value leads that match "
            "our ideal customer profile",
            backstory=(
                "As a part of the dynamic sales team at CrewAI, "
                "your mission is to scour "
                "the digital landscape for potential leads. "
                "Armed with cutting-edge tools "
                "and a strategic mindset, you analyze data, "
                "trends, and interactions to "
                "unearth opportunities that others might overlook. "
                "Your work is crucial in paving the way "
                "for meaningful engagements and driving the company's growth."
            ),
            allow_delegation=False,
            verbose=True,
            llm=sales_model,
        )

        lead_sales_rep_agent = Agent(
            role="Lead Sales Representative",
            goal="Nurture leads with personalized, compelling communications",
            backstory=(
                "Within the vibrant ecosystem of CrewAI's sales department, "
                "you stand out as the bridge between potential clients "
                "and the solutions they need."
                "By creating engaging, personalized messages, "
                "you not only inform leads about our offerings "
                "but also make them feel seen and heard."
                "Your role is pivotal in converting interest "
                "into action, guiding leads through the journey "
                "from curiosity to commitment."
            ),
            allow_delegation=False,
            verbose=True,
            llm=lead_sales_model,
        )
        directory_read_tool = DirectoryReadTool(directory="./instructions")
        file_read_tool = FileReadTool()
        search_tool = SerperDevTool()

        sentiment_analysis_tool = SentimentAnalysisTool()

        lead_profiling_task = Task(
            description=(
                "Conduct an in-depth analysis of {lead_name}, "
                "a company in the {industry} sector "
                "that recently showed interest in our solutions. "
                "Utilize all available data sources "
                "to compile a detailed profile, "
                "focusing on key decision-makers, recent business "
                "developments, and potential needs "
                "that align with our offerings. "
                "This task is crucial for tailoring "
                "our engagement strategy effectively.\n"
                "Don't make assumptions and "
                "only use information you absolutely sure about."
            ),
            expected_output=(
                "A comprehensive report on {lead_name}, "
                "including company background, "
                "key personnel, recent milestones, and identified needs. "
                "Highlight potential areas where "
                "our solutions can provide value, "
                "and suggest personalized engagement strategies."
            ),
            tools=[directory_read_tool, file_read_tool, search_tool],
            agent=sales_rep_agent,
        )

        personalized_outreach_task = Task(
            description=(
                "Using the insights gathered from "
                "the lead profiling report on {lead_name}, "
                "craft a personalized outreach campaign "
                "aimed at {key_decision_maker}, "
                "the {position} of {lead_name}. "
                "The campaign should address their recent {milestone} "
                "and how our solutions can support their goals. "
                "Your communication must resonate "
                "with {lead_name}'s company culture and values, "
                "demonstrating a deep understanding of "
                "their business and needs.\n"
                "Don't make assumptions and only "
                "use information you absolutely sure about."
            ),
            expected_output=(
                "A series of personalized email drafts "
                "tailored to {lead_name}, "
                "specifically targeting {key_decision_maker}."
                "Each draft should include "
                "a compelling narrative that connects our solutions "
                "with their recent achievements and future goals. "
                "Ensure the tone is engaging, professional, "
                "and aligned with {lead_name}'s corporate identity."
            ),
            tools=[sentiment_analysis_tool, search_tool],
            agent=lead_sales_rep_agent,
        )

        crew = Crew(
            agents=[sales_rep_agent, lead_sales_rep_agent],
            tasks=[lead_profiling_task, personalized_outreach_task],
            verbose=True,
            memory=True,
            embedder={
                "provider": "ollama",
                "config": {"model": "mxbai-embed-large:latest"},
            },
        )

        # inputs = {
        #     "lead_name": "DeepLearningAI",
        #     "industry": "Online Learning Platform",
        #     "key_decision_maker": "Andrew Ng",
        #     "position": "CEO",
        #     "milestone": "product launch"
        # }

        inputs = {
            "lead_name": lead_name,
            "industry": industry,
            "key_decision_maker": decision_maker,
            "position": position,
            "milestone": milestone,
        }

        with st.spinner("답변 생성 중..."):
            result = crew.kickoff(inputs=inputs)

        st.markdown(result)
