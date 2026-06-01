from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
import datetime
import os

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

load_dotenv()

# initializing the language model
llm = ChatOpenAI(model="gpt-4")

# initializing the search tool
search_tool = TavilySearchResults(search_depth = "basic")

# defining a custom tool to get the current system time
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool, get_system_time]

# creating the agent with the language model and the tools
agent = create_agent(model = llm, tools=tools)

# running the agent with a user query
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Use the get_system_time tool to get today's date. Then search for SpaceX's last launch and how many days ago was that from this instant"
            }
        ]
    }
)
print(response["messages"][-1].content)