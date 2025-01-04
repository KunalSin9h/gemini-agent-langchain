# Agent using ReAct Reasoning Framework
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.tools import GooglePlacesTool
from langchain_community.utilities import SerpAPIWrapper

@tool
def search(query: str):
    """Use the SerpAPI to run a Google Search."""
    search = SerpAPIWrapper()
    return search.run(query)

@tool
def places(query: str):
    """Use the Google Places API to run a Google Places Query"""
    places = GooglePlacesTool()
    return places.run(query)

