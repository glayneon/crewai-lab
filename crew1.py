import streamlit as st
from crewai import Agent, Task, Crew

# download lg-exaone model in advance
st.title("Hangul Exaone-LLM with CrewAI")

# Input for prompt
prompt = st.text_input(label="Enter your prompt:")
button = st.button("Generate")

if button:
    if prompt:
        # Create an agent with role, goal..
        agent = Agent(
            role="Assistant",
            goal="Provide helpful responses based on user input",
            backstory="This agent assists users by generating responses using a Exaone LLM.",
            # llm="ollama/llama-3.2-korean-bllossom-3b-q8:latest",
            llm="ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:latest",
        )

        # Create a task associated with the agent
        task = Task(
            description=f"Generate a response based on user input: {prompt}",
            expected_output="The generated response will be a list and only use Korean language.",
            agent=agent,
        )

        # Create a Crew with the agent and the task
        crew = Crew(agents=[agent], tasks=[task])

        # Execute the crew to get results
        result = crew.kickoff()

        # Display the generated response
        st.markdown(result)
