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
    CHEF = "Chef"
    CHEF_GOAL = "Generate a creative recipe based on the given idea. Only use in Korean language."
    CHEF_STORY = (
        "This agent is a skilled chef who loves to create delicious recipes."
    )
    CHEF_DESC = "Create a recipe based on the idea: {topic}"
    CHEF_EXPECT = "A detailed recipe with ingredients and instructions. Only use in Korean language."

    NUTRI = "Nutritionist"
    NUTRI_GOAL = "Analyze the generated recipe for nutritional value. Only use in Korean language."
    NUTRI_STORY = "This agent is a nutrition expert who evaluates recipes for health benefits."
    NUTRI_DESC = "Evaluate the nutritional value of the generated recipe."
    NUTRI_EXPERT = (
        "Nutritional analysis of the recipe. Only use in Korean language."
    )


# main
st.title("레시피 생성 및 영양 분석")

models = [e.value for e in LLM]

recipe_prompt = st.text_area(label="레시피 키워드: ")
recipe_model = st.selectbox("레시피를 생성할 모델", models)
analysis_model = st.selectbox("레시피에 대한 영양 분석을 수행할 모델", models)
button = st.button("레시피 생성 및 영양 분석")

if button:
    if recipe_prompt:
        chef = Agent(
            role=PROMPT.CHEF.value,
            goal=PROMPT.CHEF_GOAL.value,
            backstory=PROMPT.CHEF_STORY.value,
            llm=recipe_model,
        )
        nutritionist = Agent(
            role=PROMPT.NUTRI.value,
            goal=PROMPT.NUTRI_GOAL.value,
            backstory=PROMPT.NUTRI_STORY,
            llm=analysis_model,
        )

        # Create tasks for agants
        generate_recipe = Task(
            description=PROMPT.CHEF_DESC.value,
            expected_output=PROMPT.CHEF_EXPECT.value,
            agent=chef,
        )

        analyze_nutrition = Task(
            description=PROMPT.NUTRI_DESC.value,
            expected_output=PROMPT.NUTRI_EXPERT.value,
            agent=nutritionist,
        )

        recipe_crew = Crew(
            agents=[chef, nutritionist],
            tasks=[generate_recipe, analyze_nutrition],
        )

        final_result = recipe_crew.kickoff(inputs={"topic": recipe_prompt})

        st.subheader("생성된 레시피")
        st.markdown(generate_recipe.output)

        st.subheader("영양 분석")
        st.markdown(final_result)
