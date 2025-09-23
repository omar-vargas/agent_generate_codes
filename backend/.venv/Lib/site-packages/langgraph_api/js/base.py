from langchain_core.runnables import Runnable

from langgraph_api.schema import Config


class BaseRemotePregel(Runnable):
    name: str = "LangGraph"

    graph_id: str

    # Config passed from get_graph()
    config: Config
