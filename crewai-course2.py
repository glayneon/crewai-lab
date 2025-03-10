import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from enum import Enum


class CREWAI(Enum):
    # DOC = "https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
    ON = "https://service.opsnow.com/docs/console/en/user-guide-console-en.html"


class LLM(Enum):
    EXA7B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:latest"
    EXA4B = "ollama/hf.co:/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:latest"
    BLLOSSOM = "ollama/llama-3.2-korean-bllossom-3b-q8:latest"
    DEEPSEEK = "ollama/deepseek-r1:1.5b"
    DEEPSEEK8B = "ollama/deepseek-r1:8b"
    UNCENSOR = "ollama/hf.co:/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF:latest"


# main
st.set_page_config(layout="wide")
st.title("Multi-Agent Customer Support Automation")

support_model = st.selectbox("Support Model", [e.value for e in LLM])
qa_model = st.selectbox("QA Model", [e.value for e in LLM])

customer = st.text_input(label="Customer", max_chars=100)
person = st.text_input(label="Person", max_chars=100)
inquiry = st.text_area(label="Inquiry")
button = st.button("생성하기")

if button:
    if all([customer, person, inquiry]):
        support_agent = Agent(
            role="Senior Support Representative",
            goal="Be the most friendly and helpful "
            "support representative in your team",
            backstory=(
                "You work at Opsnow (https://opsnow.com) and "
                " are now working on providing "
                "support to {customer}, a super important customer "
                " for your company."
                "You need to make sure that you provide the best support!"
                "Make sure to provide full complete answers, "
                " and make no assumptions."
            ),
            allow_delegation=False,
            verbose=True,
            llm=support_model,
        )
        support_quality_assurance_agent = Agent(
            role="Support Quality Assurance Specialist",
            goal="Get recognition for providing the "
            "best support quality assurance in your team",
            backstory=(
                "You work at Opsnow (https://opsnow.com) and "
                "are now working with your team "
                "on a request from {customer} ensuring that "
                "the support representative is "
                "providing the best support possible.\n"
                "You need to make sure that the support representative "
                "is providing full"
                "complete answers, and make no assumptions."
            ),
            verbose=True,
            llm=qa_model,
        )

        docs_scrape_tool = ScrapeWebsiteTool(website_url=CREWAI.ON.value)

        inquiry_resolution = Task(
            description=(
                "{customer} just reached out with a super important ask:\n"
                "{inquiry}\n\n"
                "{person} from {customer} is the one that reached out. "
                "Make sure to use everything you know "
                "to provide the best support possible."
                "You must strive to provide a complete "
                "and accurate response to the customer's inquiry."
            ),
            expected_output=(
                "A detailed, informative response to the "
                "customer's inquiry that addresses "
                "all aspects of their question.\n"
                "The response should include references "
                "to everything you used to find the answer, "
                "including external data or solutions. "
                "Ensure the answer is complete, "
                "leaving no questions unanswered, and maintain a helpful and friendly "
                "tone throughout."
            ),
            tools=[docs_scrape_tool],
            agent=support_agent,
        )

        quality_assurance_review = Task(
            description=(
                "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
                "Ensure that the answer is comprehensive, accurate, and adheres to the "
                "high-quality standards expected for customer support.\n"
                "Verify that all parts of the customer's inquiry "
                "have been addressed "
                "thoroughly, with a helpful and friendly tone.\n"
                "Check for references and sources used to "
                " find the information, "
                "ensuring the response is well-supported and "
                "leaves no questions unanswered."
            ),
            expected_output=(
                "A final, detailed, and informative response "
                "ready to be sent to the customer.\n"
                "This response should fully address the "
                "customer's inquiry, incorporating all "
                "relevant feedback and improvements.\n"
                "Don't be too formal, we are a chill and cool company "
                "but maintain a professional and friendly tone throughout."
            ),
            agent=support_quality_assurance_agent,
        )

        crew = Crew(
            agents=[support_agent, support_quality_assurance_agent],
            tasks=[inquiry_resolution, quality_assurance_review],
            verbose=True,
            memory=True,
            embedder={
                "provider": "ollama",
                "config": {"model": "mxbai-embed-large:latest"},
            },
        )

        inputs = {
            "customer": customer,
            "person": person,
            "inquiry": inquiry,
        }
        with st.spinner("답변 생성 중..."):
            result = crew.kickoff(inputs=inputs)

        st.markdown(result)
