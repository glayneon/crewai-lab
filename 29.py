import streamlit as st
from crewai import Agent, Task, Crew

st.title("Personal Finance Advisor")

# Input for financial goal
financial_goal = st.text_input(label="Enter your financial goal (e.g., saving for a house):")
button = st.button("Get Advice")

if button:
    if financial_goal:
        # Create agents for financial planning and investment advice
        planner = Agent(
            role="Financial Planner",
            goal="Create a savings plan to achieve the financial goal.",
            backstory="This agent helps users manage their finances effectively.",
            llm="ollama/llama3.1"
        )

        investor = Agent(
            role="Investment Advisor",
            goal="Provide investment options based on user’s risk tolerance.",
            backstory="This agent specializes in finding suitable investment opportunities.",
            llm="ollama/llama3.1"
        )

        # Create tasks for the agents
        create_plan = Task(
            description=f"Develop a savings plan for achieving: {financial_goal}.",
            expected_output="A detailed savings plan with steps to follow.",
            agent=planner
        )

        suggest_investments = Task(
            description=f"Suggest investment options based on user's financial situation.",
            expected_output="A list of recommended investments.",
            agent=investor
        )

        # Create a Crew with the agents and tasks
        finance_crew = Crew(agents=[planner, investor], tasks=[create_plan, suggest_investments])

        # Execute the crew to provide financial advice
        final_advice = finance_crew.kickoff()

        # Display the savings plan and investment options
        st.subheader("Savings Plan:")
        st.markdown(create_plan.output)

        st.subheader("Investment Options:")
        st.markdown(final_advice)