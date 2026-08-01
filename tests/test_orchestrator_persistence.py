from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from deepresearch.agent.orchestrator import run_research
from deepresearch.backends.local_corpus import LocalCorpusBackend
from deepresearch.config import RunConfig
from deepresearch.schemas import RunStatus
from deepresearch.store import db
from deepresearch.telemetry.otel_setup import init_telemetry

DOCS = [
    {"doc_id": "d1", "title": "Paris", "text": "Paris is the capital of France and its largest city."},
    {"doc_id": "d2", "title": "Berlin", "text": "Berlin is the capital of Germany, known for its history."},
]


class StubChatModel:
    """First turn requests one `search` call (so the real search/fetch tools
    actually run and get recorded -- what this file's persistence tests
    check for); second turn answers. The per-hop finalize output itself is
    fully controlled by make_stub_llm."""

    def __init__(self) -> None:
        self._turn = 0

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self._turn += 1
        if self._turn == 1:
            msg = AIMessage(
                content="", tool_calls=[{"name": "search", "args": {"query": "capital"}, "id": "call_1", "type": "tool_call"}]
            )
        else:
            msg = AIMessage(content="done", tool_calls=[])
        msg.usage_metadata = {"input_tokens": 5, "output_tokens": 5}
        return msg


@pytest.mark.asyncio
async def test_run_research_persists_runs_trajectories_and_tool_calls(tmp_path, make_stub_llm):
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_test.db'}"

    config = RunConfig(
        database_url=db_url,
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        cache_enabled=False,
        rerank_enabled=False,  # keep it fast — reranker model isn't the point of this test
    )
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = make_stub_llm(seed=1)

    result = await run_research(
        "What is the capital of France?", config=config, search_backend=backend, llm=llm,
        chat_model=StubChatModel(), benchmark_name="test",
    )

    assert result.status == RunStatus.COMPLETED

    from sqlalchemy import select
    from deepresearch.store.models import runs, trajectories, tool_calls

    engine = db.get_engine(db_url)
    async with engine.begin() as conn:
        run_rows = (await conn.execute(select(runs).where(runs.c.run_id == result.run_id))).fetchall()
        traj_rows = (await conn.execute(select(trajectories).where(trajectories.c.run_id == result.run_id))).fetchall()
        tool_rows = (await conn.execute(select(tool_calls).where(tool_calls.c.run_id == result.run_id))).fetchall()

    assert len(run_rows) == 1
    assert run_rows[0]._mapping["status"] == "completed"
    assert run_rows[0]._mapping["benchmark_name"] == "test"

    stages = {row._mapping["stage"] for row in traj_rows}
    # A single independent (leaf) node: plan -> subagent(agent_step + final-
    # ize_finding, recorded by the subagent's own inner ReAct subgraph) ->
    # synthesis -> reflection. No "verify" stage — nothing depends on the
    # lone node, so its finding is marked verified for free (no paid
    # grounding call).
    assert stages == {"plan", "subagent", "agent_step", "finalize_finding", "synthesis", "reflection"}

    tool_names = {row._mapping["tool_name"] for row in tool_rows}
    assert "search" in tool_names
    assert "fetch" in tool_names
    # every tool_call's span_id must reference a real trajectory span_id (FK)
    traj_span_ids = {row._mapping["span_id"] for row in traj_rows}
    assert all(row._mapping["span_id"] in traj_span_ids for row in tool_rows)


@pytest.mark.asyncio
async def test_run_research_on_event_fires_for_every_stage(tmp_path, make_stub_llm):
    """The SSE streaming endpoint (api/streaming.py) has no other hook into
    run_research()'s progress — on_event must fire once per stage, in order,
    with the stages this graph actually produces."""
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_events_test.db'}"
    config = RunConfig(
        database_url=db_url, checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        cache_enabled=False, rerank_enabled=False,
    )
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = make_stub_llm(seed=1)

    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    result = await run_research(
        "What is the capital of France?", config=config, search_backend=backend, llm=llm,
        chat_model=StubChatModel(), on_event=on_event,
    )

    assert result.status == RunStatus.COMPLETED
    assert events[0] == {"type": "run_started", "run_id": result.run_id, "question": "What is the capital of France?"}

    stage_events = [e for e in events[1:] if e["type"] == "stage_complete"]
    assert len(stage_events) >= 3  # plan, subagent, synthesis (+ reflection)
    stages_in_order = [e["stage"] for e in stage_events]
    assert stages_in_order[0] == "plan"
    assert stages_in_order[-1] == "reflection"
    assert "subagent" in stages_in_order
    assert all(isinstance(e["latency_ms"], (int, float)) for e in stage_events)


@pytest.mark.asyncio
async def test_run_research_events_are_enriched_and_include_nested_subagent_stages(tmp_path, make_stub_llm):
    """Phase 1 of the live-trace UI work: on_event must carry paired
    stage_start/stage_complete events (correlated by stage_id, for a
    spinner-to-checkmark UI), stage_complete must carry the actual `output`
    (not just tokens/cost/latency -- previously the only thing a live trace
    could show was "subagent:n1" and a dollar amount), and nested subagent
    ReAct-loop stages (agent_step, finalize_finding -- each their own
    compiled subgraph, invoked via ainvoke() in agent/subagent.py) must
    reach on_event at all, which requires astream(..., subgraphs=True) in
    orchestrator.py -- without it these stages are silently invisible to any
    stream consumer even though they were always written to the trajectories
    table (recorder is a plain shared object, unaffected by streaming)."""
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_enriched_events_test.db'}"
    config = RunConfig(
        database_url=db_url, checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        cache_enabled=False, rerank_enabled=False,
    )
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = make_stub_llm(seed=1)

    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    result = await run_research(
        "What is the capital of France?", config=config, search_backend=backend, llm=llm,
        chat_model=StubChatModel(), on_event=on_event,
    )
    assert result.status == RunStatus.COMPLETED

    starts = [e for e in events if e["type"] == "stage_start"]
    completes = [e for e in events if e["type"] == "stage_complete"]
    assert starts  # previously didn't exist at all
    stage_types = {e["stage"] for e in completes}
    # agent_step/finalize_finding only reach on_event with subgraphs=True.
    assert {"plan", "subagent", "agent_step", "finalize_finding", "synthesis", "reflection"} <= stage_types

    # Every stage_complete's stage_id was announced by a matching stage_start
    # first -- the pairing a spinner-to-checkmark UI depends on.
    start_ids = {e["stage_id"] for e in starts}
    for c in completes:
        assert c["stage_id"] in start_ids

    plan_complete = next(e for e in completes if e["stage"] == "plan")
    assert plan_complete["output"]["sub_questions"][0]["id"] == "n1"

    subagent_complete = next(e for e in completes if e["stage"] == "subagent")
    assert subagent_complete["node_id"] == "n1"
    assert subagent_complete["wave"] == 0
    assert subagent_complete["output"]["node_id"] == "n1"
    assert subagent_complete["output"]["answer"]  # the actual finding, not just a dollar figure

    agent_step_complete = next(e for e in completes if e["stage"] == "agent_step")
    assert agent_step_complete["node_id"] == "n1_r0"  # AgentContext.source_id_prefix


@pytest.mark.asyncio
async def test_run_research_bypass_cache_flag_means_no_stats(tmp_path, make_stub_llm):
    init_telemetry()
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'orch_test2.db'}"
    config = RunConfig(
        database_url=db_url, checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        cache_enabled=False, rerank_enabled=False,
    )
    backend = LocalCorpusBackend.from_dicts(DOCS)
    llm = make_stub_llm(seed=2)

    result = await run_research(
        "Capital of Germany?", config=config, search_backend=backend, llm=llm, chat_model=StubChatModel(),
    )

    assert result.cache_stats.hit_rate == 0.0
    assert result.cache_stats.estimated_dollars_saved == 0.0
