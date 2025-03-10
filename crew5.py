import streamlit as st
from crewai import Agent, Task, Crew
from enum import Enum


class LLM(Enum):
    EXA7B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:latest"
    EXA4B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:latest"
    BLLOSSOM = "ollama/llama-3.2-korean-bllossom-3b-q8:latest"
    DEEPSEEK = "ollama/deepseek-r1:1.5b"
    DEEPSEEK8B = "ollama/deepseek-r1:8b"
    UNCENSOR = "ollama/hf.co:/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF:latest"


class PROMPT(Enum):
    TITLE = "Travel Itinerary Planner"
    PLANNER = "Trip Planner"
    PLANNER_GOAL = (
        "Create a detailed travel itinerary based on the destination."
    )
    PLANNER_STORY = "This agent specializes in planning exciting trips."
    PLANNER_DESC = "Plan a trip itinerary for {destination}"
    PLANNER_EXPECT = "A detailed itinerary including activities and timings. Only use in Korean language."

    LOCAL_EXPERT = "Local Expert"
    LOCAL_GOAL = "Provide insights and recommendations about the destination."
    LOCAL_STORY = "This agent knows all the best spots in town!"
    LOCAL_DESC = "Provide local insights about {destination}."
    LOCAL_EXPECT = "Recommendations for places to visit and eat. Only use in Korean language."


# main
st.title(PROMPT.TITLE.value)

models = [e.value for e in LLM]

planner_prompt = st.text_area(label="여행 목적지: ")
planner_model = st.selectbox("플래너 모델", models)
local_expert_model = st.selectbox("현지 전문가 모델", models)
button = st.button("여행지 생성 및 리뷰 생성")

if button:
    if planner_prompt:
        planner = Agent(
            role=PROMPT.PLANNER.value,
            goal=PROMPT.PLANNER_GOAL.value,
            backstory=PROMPT.PLANNER_STORY.value,
            llm=planner_model,
        )
        local_expert = Agent(
            role=PROMPT.LOCAL_EXPERT.value,
            goal=PROMPT.LOCAL_GOAL.value,
            backstory=PROMPT.LOCAL_STORY.value,
            llm=local_expert_model,
        )

        # Create tasks for agants
        planner_task = Task(
            description=PROMPT.PLANNER_DESC.value,
            expected_output=PROMPT.PLANNER_EXPECT.value,
            agent=planner,
        )

        local_expert_task = Task(
            description=PROMPT.LOCAL_DESC.value,
            expected_output=PROMPT.LOCAL_EXPERT.value,
            agent=local_expert,
        )

        travel_itinerary = Crew(
            agents=[planner, local_expert],
            tasks=[planner_task, local_expert_task],
        )

        final_result = travel_itinerary.kickoff(
            inputs={"destination": planner_prompt}
        )

        st.subheader("여행 계획")
        st.markdown(local_expert_task.output)

        st.subheader("현지 인사이트")
        st.markdown(final_result)
