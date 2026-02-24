from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from huggingface_hub import list_models

import random
from utils import get_weather_info, get_hub_stats
from retriever import extract_text

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

search_tool = DuckDuckGoSearchRun()

weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)


# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)
