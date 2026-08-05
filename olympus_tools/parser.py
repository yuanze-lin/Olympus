"""Parse Olympus router output into an executable plan.

The Olympus router answers a user instruction with a string of task-specific
routing tokens, e.g.::

    <image_gen>a fluffy orange cat ...</image_gen><image_edit>change the cat's
    colour to white.</image_edit><3D_gen_image>...</3D_gen_image>

This module turns that string into an ordered :class:`Plan` of :class:`Step`
objects and resolves the chain-of-action dataflow: a step that consumes an image
is wired to the most recent preceding step that produced one, falling back to the
image supplied by the user.
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

from .tokens import get_spec, TaskSpec, NONE, IMAGE, VIDEO

# <tag>payload</tag>, non-greedy; also tolerates an unclosed trailing tag.
_PAIRED = re.compile(r"<([A-Za-z0-9_]+)>(.*?)</\1>", re.DOTALL)
_UNCLOSED = re.compile(r"<([A-Za-z0-9_]+)>(?!.*?</\1>)(.*)$", re.DOTALL)

USER_INPUT = "user"


@dataclass
class Step:
    """One specialist invocation."""

    index: int
    token: str
    task: str
    backend: str
    prompt: str
    consumes: str
    produces: str
    condition: Optional[str] = None
    # Where this step's input artifact comes from: ``"user"``, ``None`` (needs
    # nothing) or the index of an earlier step whose output feeds this one.
    input_from: Optional[object] = None

    @property
    def spec(self) -> TaskSpec:
        return get_spec(self.token)


@dataclass
class Plan:
    """An ordered set of specialist invocations derived from router output."""

    steps: List[Step]
    direct_answer: str = ""   # prose the router emitted outside any routing token
    raw_output: str = ""
    instruction: str = ""
    unknown_tokens: List[str] = None

    def __post_init__(self):
        if self.unknown_tokens is None:
            self.unknown_tokens = []

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "instruction": self.instruction,
                "raw_output": self.raw_output,
                "direct_answer": self.direct_answer,
                "unknown_tokens": self.unknown_tokens,
                "steps": [asdict(s) for s in self.steps],
            },
            indent=indent,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        blob = json.loads(text)
        # Tolerate plans written before paper_model was dropped.
        steps = [Step(**{k: v for k, v in s.items() if k != "paper_model"})
                 for s in blob.get("steps", [])]
        return cls(
            steps=steps,
            direct_answer=blob.get("direct_answer", ""),
            raw_output=blob.get("raw_output", ""),
            instruction=blob.get("instruction", ""),
            unknown_tokens=blob.get("unknown_tokens", []),
        )

    def describe(self) -> str:
        if not self.steps:
            return "no routing tokens -- the router answered directly"
        lines = []
        for s in self.steps:
            src = ""
            if s.consumes != NONE:
                src = f"  <- {'user input' if s.input_from == USER_INPUT else f'step {s.input_from}'}"
            lines.append(f"  [{s.index}] <{s.token}> via {s.backend}{src}\n"
                         f"        \"{s.prompt}\"")
        return "\n".join(lines)


def _resolve_dataflow(steps: List[Step]) -> None:
    """Wire each step's input to the most recent step producing what it needs."""
    for i, step in enumerate(steps):
        if step.consumes == NONE:
            step.input_from = None
            continue
        producer = None
        for j in range(i - 1, -1, -1):
            if steps[j].produces == step.consumes:
                producer = steps[j].index
                break
        # A video task with no upstream video can still be seeded by an image
        # (Text2Video-Zero conditions on a single control frame).
        if producer is None and step.consumes == VIDEO:
            for j in range(i - 1, -1, -1):
                if steps[j].produces == IMAGE:
                    producer = steps[j].index
                    break
        step.input_from = producer if producer is not None else USER_INPUT


def parse(router_output: str, instruction: str = "") -> Plan:
    """Parse raw router output into a :class:`Plan`."""
    text = (router_output or "").strip()
    steps: List[Step] = []
    unknown: List[str] = []
    consumed_spans = []

    for m in _PAIRED.finditer(text):
        token, payload = m.group(1), m.group(2).strip()
        consumed_spans.append(m.span())
        spec = get_spec(token)
        if spec is None:
            unknown.append(token)
            continue
        steps.append(Step(
            index=len(steps),
            token=spec.token,
            task=spec.task,
            backend=spec.backend,
            prompt=payload,
            consumes=spec.consumes,
            produces=spec.produces,
            condition=spec.condition,
        ))

    # Tolerate a truncated final token (generation hit the length cap).
    tail_start = consumed_spans[-1][1] if consumed_spans else 0
    tail = text[tail_start:]
    m = _UNCLOSED.search(tail)
    if m:
        token, payload = m.group(1), m.group(2).strip()
        spec = get_spec(token)
        if spec is not None and payload:
            steps.append(Step(
                index=len(steps),
                token=spec.token,
                task=spec.task,
                backend=spec.backend,
                prompt=payload,
                consumes=spec.consumes,
                produces=spec.produces,
                condition=spec.condition,
            ))
            consumed_spans.append((tail_start + m.start(), len(text)))

    # Anything outside the tokens is the router answering directly (e.g. VQA).
    leftover, cursor = [], 0
    for start, end in consumed_spans:
        leftover.append(text[cursor:start])
        cursor = end
    leftover.append(text[cursor:])
    direct = re.sub(r"\s+", " ", " ".join(leftover)).strip()

    _resolve_dataflow(steps)
    return Plan(steps=steps, direct_answer=direct, raw_output=text,
                instruction=instruction, unknown_tokens=unknown)
