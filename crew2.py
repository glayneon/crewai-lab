import streamlit as st
from crewai import Agent, Task, Crew
from enum import Enum


class LLM(Enum):
    EXA7B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:latest"
    EXA4B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:latest"
    BLLOSSOM = "ollama/llama-3.2-korean-bllossom-3b-q8:latest"
    DEEPSEEK = "ollama/deepseek-r1:1.5b"


st.title("이야기 생성 및 편집 Crew AI")

models = [m.value for m in LLM]

# Input for story prompt
story_prompt = st.text_area(label="간단한 이야기 키워드 :")
writer_model = st.selectbox("이야기를 쓸 모델", models)
editor_model = st.selectbox("이야기를 편집할 모델", models)
button = st.button("이야기 생성")

if button:
    if story_prompt:
        # Create the agent for story generation
        writer = Agent(
            role="Writer",
            goal="Generate a creative story based on the given prompt. Only use in Korean language.",
            backstory="""This agent is a skilled writer with a vivid imagination. 
            They will craft an engaging story using the provided prompt as inspiration.""",
            llm=writer_model,
        )

        # Create the agent for story reviewing and editing
        editor = Agent(
            role="Editor",
            goal="Review and refine the generated story for clarity, coherence, and flow. Only use in Korean language.",
            backstory="""This agent is an experienced editor who will carefully review the story, 
            pvovide feedback, and make necessary revisions to improve the overall quality.""",
            llm=editor_model,
        )

        # Create tasks for the agents
        write_story = Task(
            description=f"Write a short story based on the prompt: {story_prompt}",
            expected_output="A creative short story with a beginning, middle, and end. Only use Korean language.",
            agent=writer,
        )
        edit_story = Task(
            description="Review and edit the generated story to enhance clarity, coherence, and flow.",
            expected_output="The edited story with improved quality. Only use Korean language",
            agent=editor,
        )

        # Create a Crew with the agents and tasks
        story_crew = Crew(
            agents=[writer, editor], tasks=[write_story, edit_story]
        )

        # Execute the story generation and editing
        final_story = story_crew.kickoff()

        # Display the generated story
        col1, col2 = st.columns(2)
        with col1:
            st.header("생성된 이야기")
            st.write(f"Model: {writer_model}")
            st.markdown(write_story.output)
        with col2:
            st.header("편집된 이야기")
            st.write(f"Model: {editor_model}")
            st.markdown(final_story)
