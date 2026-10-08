from pathlib import Path

import pytest

from peer_options.transcripts.parser import Role, Section, parse_transcript

REPO = Path(__file__).resolve().parents[3]

# SYNTHETIC fixtures (invented text, not a real call).
BARE = """SYNTHETIC Corp (NASDAQ:SYN)
Q1 2030 Earnings Conference Call
January 1, 2030, 05:00 PM ET

Company Participants

Alice Ceo - CEO
Bob Cfo - CFO
Ian Ir - Head of IR

Conference Call Participants

Dana Analyst - Big Bank

Operator

Welcome to the call.

Alice Ceo

We had a strong quarter and expect growth to continue.

Bob Cfo

Margins improved. We may see some variability next quarter.

Question-and-Answer Session

Operator

First question from Dana Analyst.

Dana Analyst

Can you talk about demand? I am worried about weakness.

Alice Ceo

Demand is strong. We are optimistic.
"""

TITLED = """Operator (Operator):
Good day.

Ian Ir (Head of Investor Relations):
Welcome, please limit yourself to two questions.

Alice Ceo (CEO):
Great quarter, record revenue.

Operator (Operator):
We will take the first question from Dana.

Dana Analyst (Analyst):
Any concerns on demand?

Alice Ceo (CEO):
No, demand is robust.
"""


def roles(t):
    return [(u.speaker, u.role.value, u.section.value) for u in t.utterances]


def test_bare_layout_with_marker():
    t = parse_transcript(BARE)
    assert t.qa_boundary_source == "marker"
    assert roles(t) == [
        ("Operator", "operator", "prepared"),
        ("Alice Ceo", "executive", "prepared"),
        ("Bob Cfo", "executive", "prepared"),
        ("Operator", "operator", "qa"),
        ("Dana Analyst", "analyst", "qa"),
        ("Alice Ceo", "executive", "qa"),
    ]
    assert "Question-and-Answer" not in " ".join(u.text for u in t.utterances)


def test_titled_layout_infers_boundary_from_first_analyst():
    t = parse_transcript(TITLED)
    assert t.qa_boundary_source == "first_analyst"
    assert roles(t)[:4] == [
        ("Operator", "operator", "prepared"),
        ("Ian Ir", "ir", "prepared"),
        ("Alice Ceo", "executive", "prepared"),
        ("Operator", "operator", "qa"),  # operator introducing the first question opens Q&A
    ]
    assert t.utterances[-1].section is Section.QA


def test_only_executive_speech_by_default_analysts_separate():
    t = parse_transcript(BARE)
    assert "worried" not in t.text(Section.QA)
    assert "worried" in t.text(Section.QA, roles=(Role.ANALYST,))
    assert "Welcome to the call" not in t.text(Section.PREPARED)


@pytest.mark.parametrize("name,marker,first_prepared", [
    ("tesla_seeking_alpha.txt", "marker", "Elon Musk"),
    ("apple_2024Q1_seeking_alpha.txt", "first_analyst", "Tim Cook"),
])
def test_real_transcripts(name, marker, first_prepared):
    p = REPO / "RAG" / "transcripts" / name
    if not p.exists():
        pytest.skip("transcript not present")
    t = parse_transcript(p.read_text())
    assert t.qa_boundary_source == marker
    prepared_exec = [u for u in t.utterances if u.section is Section.PREPARED and u.role is Role.EXECUTIVE]
    qa_analysts = [u for u in t.utterances if u.section is Section.QA and u.role is Role.ANALYST]
    assert prepared_exec and prepared_exec[0].speaker == first_prepared
    assert len(qa_analysts) >= 5
    assert not any(u.role is Role.ANALYST for u in t.utterances if u.section is Section.PREPARED)
