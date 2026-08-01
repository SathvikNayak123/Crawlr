"""synthesize() (agent/synthesis.py): report_style picks the prompt file and
token budget -- "concise" (default, what eval/ci_baselines is scored
against) vs. "report" (the live UI's structured multi-section markdown,
docs/DESIGN.md Phase 3)."""

from __future__ import annotations

import pytest

from deepresearch.agent.synthesis import SynthesisDraft, synthesize
from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMUsage
from deepresearch.schemas import WorkerNotes


class RecordingLLM:
    """Records every complete_structured call it receives; always returns a
    fixed SynthesisDraft regardless of input."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_structured(self, *, model, system, user_content, response_model, max_tokens=4096):
        self.calls.append({"model": model, "system": system, "user_content": user_content, "max_tokens": max_tokens})
        usage = LLMUsage(input_tokens=10, output_tokens=10, cost_usd=0.0)
        return SynthesisDraft(text="answer [src_1]", cited_source_ids=["src_1"]), usage


NOTES = [WorkerNotes(sub_question_id="n1", sub_question="Q1?", claims=[])]


@pytest.mark.asyncio
async def test_concise_is_the_default_style():
    llm = RecordingLLM()
    await synthesize("Q1?", NOTES, {}, config=RunConfig(), llm=llm)
    assert len(llm.calls) == 1
    assert "Limitations" not in llm.calls[0]["system"]  # sanity: concise prompt loaded, not report's
    assert llm.calls[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_report_style_uses_the_report_prompt_and_larger_token_budget():
    llm = RecordingLLM()
    config = RunConfig.from_overrides({"report_style": "report"})
    await synthesize("Q1?", NOTES, {}, config=config, llm=llm)
    assert len(llm.calls) == 1
    assert "report" in llm.calls[0]["system"].lower()
    assert "Limitations" in llm.calls[0]["system"]  # the report-mode section, absent from synthesis_v1.txt
    assert llm.calls[0]["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_unknown_report_style_raises_instead_of_silently_falling_back():
    llm = RecordingLLM()
    config = RunConfig.from_overrides({"report_style": "bogus"})
    with pytest.raises(ValueError, match="report_style"):
        await synthesize("Q1?", NOTES, {}, config=config, llm=llm)
    assert llm.calls == []  # never even reached the LLM call
