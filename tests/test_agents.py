import pytest
from kvprism.graph.state import GraphState, PipelineInput, Technology, PerspectiveResult
from kvprism.agents import market_node, stakeholder_node, domain_node

@pytest.fixture
def mock_state() -> GraphState:
    # state.py의 Technology 및 PipelineInput 규격 충족
    req = PipelineInput(
        domain="LLM Serving",
        scenario="Long-context inference",
        technologies=[
            Technology(
                technology_id="turboquant",
                name="TurboQuant",
                kind="software",
                paper_doc_id="TurboQuant.pdf",
                paper_title="TurboQuant: Online Vector Quantization"
            ),
            Technology(
                technology_id="itme",
                name="ITME",
                kind="hardware",
                paper_doc_id="ITME.pdf",
                paper_title="ITME: Inference Tiered Memory Expansion"
            )
        ]
    )
    return {
        "request": req,
        "sources": [],
        "retry_count": 0,
        "retry_targets": []
    }

def test_perspective_nodes_schema(mock_state):
    # 1. market_node 검증
    market_out = market_node(mock_state)
    assert "market_eval" in market_out
    assert isinstance(market_out["market_eval"], PerspectiveResult)
    
    # 2. stakeholder_node 검증
    stakeholder_out = stakeholder_node(mock_state)
    assert "stakeholder_eval" in stakeholder_out
    assert isinstance(stakeholder_out["stakeholder_eval"], PerspectiveResult)

    # 3. domain_node 검증
    domain_out = domain_node(mock_state)
    assert "domain_eval" in domain_out
    assert isinstance(domain_out["domain_eval"], PerspectiveResult)