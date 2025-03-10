import streamlit as st
import subprocess
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv, find_dotenv
import os
from pydantic import BaseModel
from loguru import logger
from enum import Enum


class AGENT(Enum):
    SERPER = "SERPERAPI"


class BLACKLIST_LLM(Enum):
    NOMIC = "nomic-embed"
    MXBAI = "mxbai-embed"


def not_embeding(modelname=None):
    blacklist = [e.value for e in BLACKLIST_LLM]
    for i in blacklist:
        if i not in modelname:
            return True
        else:
            return False


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
            if parts and not_embeding(parts[0]):
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
st.title("Event Planning - CrewAI")

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

if models := get_llms():
    chosen_model = st.selectbox("Select Model", models)
else:
    st.error("Can't import any models.")
    exit()
