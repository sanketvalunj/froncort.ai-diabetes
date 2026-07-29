from app.models.state import AgentState
from app.services.evaluation_service import EvaluationService
from app.services.filtering_service import FilteringService
from app.services.ranking_service import RankingService
from app.services.report_service import ReportService
from app.services.retrieval_service import RetrievalService
from app.utils.logger import log_node_execution


def make_nodes(settings, llm_client):
    filtering_svc  = FilteringService(settings)
    retrieval_svc  = RetrievalService(settings)
    evaluation_svc = EvaluationService(settings, llm_client)
    ranking_svc    = RankingService(settings)
    report_svc     = ReportService(settings)

    def filter_node(state: dict) -> dict:
        agent_state = AgentState.model_validate(state)
        with log_node_execution("filter_node", agent_state.trace_id):
            updated = filtering_svc.run(agent_state)
        return updated.model_dump()

    def retrieval_node(state: dict) -> dict:
        agent_state = AgentState.model_validate(state)
        with log_node_execution("retrieval_node", agent_state.trace_id):
            updated = retrieval_svc.run(agent_state)
        return updated.model_dump()

    def evaluation_node(state: dict) -> dict:
        agent_state = AgentState.model_validate(state)
        with log_node_execution("evaluation_node", agent_state.trace_id):
            updated = evaluation_svc.run(agent_state)
        return updated.model_dump()

    def ranking_node(state: dict) -> dict:
        agent_state = AgentState.model_validate(state)
        with log_node_execution("ranking_node", agent_state.trace_id):
            updated = ranking_svc.run(agent_state)
        return updated.model_dump()

    def report_node(state: dict) -> dict:
        agent_state = AgentState.model_validate(state)
        with log_node_execution("report_node", agent_state.trace_id):
            updated = report_svc.run(agent_state)
        return updated.model_dump()

    return {
        "filter_node":     filter_node,
        "retrieval_node":  retrieval_node,
        "evaluation_node": evaluation_node,
        "ranking_node":    ranking_node,
        "report_node":     report_node,
    }
