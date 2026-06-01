from typing import Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from chains import generation_chain, reflection_chain


REFLECT = "reflect"
GENERATE = "generate"

class State(TypedDict):
    messages: Annotated[list, add_messages]


def generate_node(state: State):
    return {"messages": generation_chain.invoke({"messages": state["messages"]})}


def reflect_node(state: State):
    response = reflection_chain.invoke({"messages": state["messages"]})
    # Wrap as HumanMessage so the generator treats it as user feedback
    return {"messages": [HumanMessage(content=response.content)]}


def should_continue(state: State):
    return END if len(state["messages"]) > 6 else REFLECT


graph = StateGraph(State)
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)
graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke({"messages": [HumanMessage(content="AI Agents taking over content creation")]})
print(response["messages"][-1].content)