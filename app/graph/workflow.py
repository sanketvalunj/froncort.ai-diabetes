from langgraph.graph import END, StateGraph

from app.graph.nodes import make_nodes
from app.models.state import AgentState
from app.utils.logger import get_logger

log = get_logger(__name__)


def build_workflow(settings, llm_client):
    nodes = make_nodes(settings, llm_client)
    graph = StateGraph(AgentState)
    graph.add_node("filter_node",     nodes["filter_node"])
    graph.add_node("retrieval_node",  nodes["retrieval_node"])
    graph.add_node("evaluation_node", nodes["evaluation_node"])
    graph.add_node("ranking_node",    nodes["ranking_node"])
    graph.add_node("report_node",     nodes["report_node"])
    graph.set_entry_point("filter_node")
    graph.add_edge("filter_node",     "retrieval_node")
    graph.add_edge("retrieval_node",  "evaluation_node")
    graph.add_edge("evaluation_node", "ranking_node")
    graph.add_edge("ranking_node",    "report_node")
    graph.add_edge("report_node",     END)
    compiled = graph.compile()
    log.info("workflow_compiled", nodes=list(nodes.keys()))
    return compiled


def run_workflow(settings, llm_client, initial_state: AgentState) -> AgentState:
    workflow = build_workflow(settings, llm_client)
    log.info("workflow_start", patient_id=initial_state.patient.id,
             total_trials=len(initial_state.all_trials),
             trace_id=initial_state.trace_id)
    result = workflow.invoke(initial_state)
    if isinstance(result, AgentState):
        final_state = result
    else:
        final_state = AgentState.model_validate(result)
    log.info("workflow_complete", patient_id=final_state.patient.id,
             ranked_trials=len(final_state.ranked_trials),
             trace_id=final_state.trace_id)
    return final_state
