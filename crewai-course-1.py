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


# main
st.set_page_config(layout="wide")
st.title("블로그 포스팅 Multi-Agent AI")

# Choose the right models for planner, writer, editor
content_model = st.selectbox("컨텐츠 플래너", [e.value for e in LLM])
writer_model = st.selectbox("컨텐츠 라이터", [e.value for e in LLM])
editor_model = st.selectbox("컨텐츠 에디터", [e.value for e in LLM])

planner_prompt = st.text_area(label="블로깅 키워드 ")
button = st.button("블로깅 포스트 생성")

if button:
    if planner_prompt:
        planner = Agent(
            role="Content Planner",
            goal="Plan engaging and factually accurate content on {topic}",
            backstory="You're working on planning a blog article "
            "about the topic: {topic}."
            "You collect information that helps the "
            "audience learn something "
            "and make informed decisions. "
            "Your work is the basis for "
            "the Content Writer to write an article on this topic.",
            allow_delegation=False,
            verbose=True,
            llm=content_model,
        )

        writer = Agent(
            role="Content Writer",
            goal="Write insightful and factually accurate "
            "opinion piece about the topic: {topic}",
            backstory="You're working on a writing "
            "a new opinion piece about the topic: {topic}. "
            "You base your writing on the work of "
            "the Content Planner, who provides an outline "
            "and relevant context about the topic. "
            "You follow the main objectives and "
            "direction of the outline, "
            "as provide by the Content Planner. "
            "You also provide objective and impartial insights "
            "and back them up with information "
            "provide by the Content Planner. "
            "You acknowledge in your opinion piece "
            "when your statements are opinions "
            "as opposed to objective statements.",
            allow_delegation=False,
            verbose=True,
            llm=writer_model,
        )

        editor = Agent(
            role="Editor",
            goal="Edit a given blog post to align with "
            "the writing style of the organization. ",
            backstory="You are an editor who receives a blog post "
            "from the Content Writer. "
            "Your goal is to review the blog post "
            "to ensure that it follows journalistic best practices,"
            "provides balanced viewpoints "
            "when providing opinions or assertions, "
            "and also avoids major controversial topics "
            "or opinions when possible.",
            allow_delegation=False,
            verbose=True,
            llm=editor_model,
        )

        # Create tasks for agants
        plan = Task(
            description=(
                "1. Prioritize the latest trends, key players, "
                "and noteworthy news on {topic}.\n"
                "2. Identify the target audience, considering "
                "their interests and pain points.\n"
                "3. Develop a detailed content outline including "
                "an introduction, key points, and a call to action.\n"
                "4. Include SEO keywords and relevant data or sources."
            ),
            expected_output="A comprehensive content plan document "
            "with an outline, audience analysis, "
            "SEO keywords, and resources."
            "Only use Korean language.",
            agent=planner,
        )

        write = Task(
            description=(
                "1. Use the content plan to craft a compelling "
                "blog post on {topic}.\n"
                "2. Incorporate SEO keywords naturally.\n"
                "3. Sections/Subtitles are properly named "
                "in an engaging manner.\n"
                "4. Ensure the post is structured with an "
                "engaging introduction, insightful body, "
                "and a summarizing conclusion.\n"
                "5. Proofread for grammatical errors and "
                "alignment with the brand's voice.\n"
            ),
            expected_output="A well-written blog post "
            "in markdown format, ready for publication, "
            "each section should have 2 or 3 paragraphs."
            "Only use Korean language.",
            agent=writer,
        )

        edit = Task(
            description=(
                "Proofread the given blog post for "
                "grammatical errors and "
                "alignment with the brand's voice."
            ),
            expected_output="A well-written blog post in markdown format, "
            "ready for publication, "
            "each section should have 2 or 3 paragraphs."
            "Only use Korean language.",
            agent=editor,
        )

        crew = Crew(
            agents=[planner, writer, editor],
            tasks=[plan, write, edit],
        )

        with st.spinner("답변 생성 중..."):
            result = crew.kickoff(inputs={"topic": planner_prompt})

        st.subheader("최종본")
        st.markdown(result)

        with st.expander("플래너 글 보기"):
            st.markdown(plan.output)

        with st.expander("라이터 글 보기"):
            st.markdown(edit.output)
