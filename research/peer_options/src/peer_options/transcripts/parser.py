"""Split an earnings-call transcript into prepared remarks vs Q&A and tag speakers.

Supported layouts (both seen in RAG/transcripts):
  * "bare":   speaker name alone on a line between blank lines; Company/Conference Call
              Participants lists in the header; usually a 'Question-and-Answer Session' line.
  * "titled": 'Name (Title):' lines, no header lists, usually no Q&A marker.
Roles: executive, analyst, operator, ir. Only `executive` speech is scored by default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    EXECUTIVE = "executive"
    ANALYST = "analyst"
    OPERATOR = "operator"
    IR = "ir"


class Section(str, Enum):
    PREPARED = "prepared"
    QA = "qa"


@dataclass(frozen=True)
class Utterance:
    index: int
    speaker: str
    role: Role
    section: Section
    text: str


@dataclass(frozen=True)
class ParsedTranscript:
    utterances: tuple[Utterance, ...]
    qa_boundary_source: str  # 'marker' | 'first_analyst' | 'none'

    def text(self, section: Section, roles: tuple[Role, ...] = (Role.EXECUTIVE,)) -> str:
        return "\n".join(
            u.text for u in self.utterances if u.section is section and u.role in roles
        )


_QA_MARKER = re.compile(
    r"^(question[- ]and[- ]answer( session)?|questions? (and|&) answers?( session)?|q ?& ?a( session)?)$",
    re.I,
)
_TITLED = re.compile(r"^(?P<name>[^:()]{2,60}?) \((?P<title>[^()]{1,80})\):\s*$")
_BARE_NAME = re.compile(r"^[A-Z][A-Za-z.'\-]*(?: [A-Za-z.'\-]+){0,4}$")
_HEADINGS = {"company participants", "conference call participants"}


def _role_from_title(name: str, title: str) -> Role:
    t, n = title.lower(), name.lower()
    if n == "operator" or t == "operator":
        return Role.OPERATOR
    if "analyst" in t:
        return Role.ANALYST
    if "investor relations" in t or re.search(r"\bir\b", t):
        return Role.IR
    return Role.EXECUTIVE


def _parse_participants(lines: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (company, conference) participant maps name -> title/firm."""
    company: dict[str, str] = {}
    conf: dict[str, str] = {}
    cur: dict[str, str] | None = None
    for raw in lines:
        s = raw.strip()
        if s.lower() == "company participants":
            cur = company
        elif s.lower() == "conference call participants":
            cur = conf
        elif cur is not None and s and " - " in s:
            n, _, t = s.partition(" - ")
            cur[n.strip()] = t.strip()
        elif cur is not None and s and s.lower() != "operator":
            cur = None
    return company, conf


def _is_blank(lines: list[str], i: int) -> bool:
    return i < 0 or i >= len(lines) or not lines[i].strip()


def parse_transcript(raw: str) -> ParsedTranscript:
    lines = raw.splitlines()
    heads: list[tuple[int, str, str, Role | None]] = []  # (line_idx, name, title, role)
    marker_line: int | None = None
    company, conf = _parse_participants(lines)
    have_lists = bool(company or conf)

    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if _QA_MARKER.match(s) and _is_blank(lines, i - 1) and _is_blank(lines, i + 1):
            marker_line = marker_line if marker_line is not None else i
            continue
        m = _TITLED.match(s)
        if m:
            name, title = m["name"].strip(), m["title"].strip()
            heads.append((i, name, title, _role_from_title(name, title)))
            continue
        if not (_is_blank(lines, i - 1) and _is_blank(lines, i + 1)):
            continue
        if s.lower() in _HEADINGS or " - " in s or len(s) > 60 or s[-1] in ".?!,:;":
            continue
        if s == "Operator" or s in company or s in conf:
            title = conf.get(s) or company.get(s) or ""
            role = Role.OPERATOR if s == "Operator" else (
                Role.ANALYST if s in conf else _role_from_title(s, title))
            heads.append((i, s, title, role))
        elif not have_lists and _BARE_NAME.match(s) and not any(c.isdigit() for c in s):
            heads.append((i, s, "", None))  # role resolved below

    utts: list[tuple[str, str, Role | None, str, int]] = []
    for k, (i, name, title, role) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body = "\n".join(
            lines[j].strip() for j in range(i + 1, end) if lines[j].strip() and j != marker_line
        )
        if body:
            utts.append((name, title, role, body, i))

    # Resolve section boundary.
    first_analyst = next((n for n, u in enumerate(utts) if u[2] is Role.ANALYST), None)
    if marker_line is not None:
        boundary = next((n for n, u in enumerate(utts) if u[4] > marker_line), len(utts))
        source = "marker"
    elif first_analyst is not None:
        boundary = first_analyst
        if boundary > 0 and utts[boundary - 1][2] is Role.OPERATOR:
            boundary -= 1  # operator introducing the first question opens Q&A
        source = "first_analyst"
    else:
        boundary, source = len(utts), "none"

    out: list[Utterance] = []
    for n, (name, title, role, body, _i) in enumerate(utts):
        if role is None:  # bare layout without participant lists: infer from position
            role = Role.OPERATOR if name == "Operator" else Role.EXECUTIVE
        out.append(Utterance(n, name, role, Section.PREPARED if n < boundary else Section.QA, body))
    return ParsedTranscript(tuple(out), source)
