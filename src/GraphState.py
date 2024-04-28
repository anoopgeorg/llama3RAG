from typing import TypedDict
from typing import List


class GraphState(TypedDict):
    query: str
    action: str
    route: str
    web_search_flag: bool
    web_context: List[str]
    context: List[str]
    generation: str
