from __future__ import annotations

from pydantic import BaseModel, Field

from deepresearch.config import RunConfig
from deepresearch.llm.client import LLMClient, LLMUsage
from deepresearch.prompts.loader import load_prompt
from deepresearch.schemas import Report, SourceRegistryEntry, WorkerNotes


class SynthesisDraft(BaseModel):
    text: str
    cited_source_ids: list[str] = Field(default_factory=list)


# report_style -> (prompt file, max_tokens). "report" needs materially more
# headroom than "concise" (multi-section markdown vs. one paragraph) — a
# fixed 4096 silently truncates a structured report mid-section.
_STYLE_PROMPTS = {
    "concise": ("synthesis_v1.txt", 4096),
    "report": ("synthesis_report_v1.txt", 8192),
}


async def synthesize(
    question: str,
    notes: list[WorkerNotes],
    source_registry: dict[str, SourceRegistryEntry],
    *,
    config: RunConfig,
    llm: LLMClient,
) -> tuple[Report, LLMUsage]:
    if config.report_style not in _STYLE_PROMPTS:
        raise ValueError(f"unknown report_style: {config.report_style!r} (expected one of {sorted(_STYLE_PROMPTS)})")
    prompt_file, max_tokens = _STYLE_PROMPTS[config.report_style]

    notes_block = "\n\n".join(
        f"Sub-question: {n.sub_question}\n" + "\n".join(f"- {c.text} [{c.source_id}]" for c in n.claims)
        for n in notes
    )
    system = load_prompt(prompt_file)
    user_content = (
        f"Research question: {question}\n\nNotes:\n{notes_block}\n\n"
        "Cite claims inline using [source_id]."
    )
    data, usage = await llm.complete_structured(
        model=config.synthesis_model,
        system=system,
        user_content=user_content,
        response_model=SynthesisDraft,
        max_tokens=max_tokens,
    )
    citations = [source_registry[sid] for sid in data.cited_source_ids if sid in source_registry]
    report = Report(text=data.text, citations=citations)
    return report, usage
