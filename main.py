# Agent using ReAct Reasoning Framework
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

# SERPAPI_API_KEY
load_dotenv()

@tool
def search(query: str):
    """Use the SerpAPI to run a Google Search."""
    search = SerpAPIWrapper()
    return search.run(query)

@tool
def research(query: str):
    """Use arxiv to search for any query in arxiv e-paper archives"""
    arxiv = ArxivAPIWrapper(
            arxiv_search=True,
            arxiv_exceptions=True,
            load_all_available_meta=True,
            ARXIV_MAX_QUERY_LENGTH= 300,
            top_k_results=3,
            load_max_docs=3,
            doc_content_chars_max=40000
    )
    return arxiv.run(query)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
tools = [search, research]

query = input("")

agent = create_react_agent(model, tools)
input = { "messages": [("human", query)]}

for s in agent.stream(input, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()

