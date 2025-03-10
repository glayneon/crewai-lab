import streamlit as st
from crewai import Crew, Agent, Task, Process
from crewai_tools import SerperDevTool
import os
from enum import Enum


class LLM(Enum):
    EXA7B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:latest"
    EXA4B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:latest"
    BLLOSSOM = "ollama/llama-3.2-korean-bllossom-3b-q8:latest"
    DEEPSEEK = "ollama/deepseek-r1:1.5b"


st.title("CrewAI Test-Lab")

models = [e.value for e in LLM]

selected_model = st.selectbox("사용할 모델을 고르세요.", models)

topic_prompt = st.text_area(label="궁금한 토픽 단어를 입력하세요.")
button = st.button(label="Generate")

if button:
    if topic_prompt:
        researcher = Agent(
            role="{topic} Senior Researcher",
            goal="""Uncover groundbreaking technologies in {topic} for year 2024. Only use in Korean language.""",
            backstory="""Driven by curiosity, you explore and share the latest innovations.""",
            llm=selected_model,
        )

        researcher_task = Task(
            description="""Identify the next big trend in {topic} with pros and cons. Only use in Korean language.""",
            expected_output="""A 3-paragraph report on emerging {topic} technologies.""",
            agent=researcher,
        )

        crew = Crew(
            agents=[researcher],
            tasks=[researcher_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff(inputs={"topic": topic_prompt})
        st.markdown(result)
