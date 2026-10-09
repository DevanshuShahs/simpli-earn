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


# SYNTHETIC fixture in the layout written by config/tsmc_pdf_to_txt.py.
DASHED = """Company Participants
Jeff Su - Director of Investor Relations
C.C. Wei - Chairman and Chief Executive Officer

Conference Call Participants
Sunny Lin - UBS AG

Prepared Remarks:

Jeff Su -- Director of Investor Relations
Welcome everyone.

C.C. Wei -- Chairman and Chief Executive Officer
Demand is strong. We are optimistic.
Capacity may be tight.

Questions and Answers:

Jeff Su -- Director of Investor Relations
First question please.
Sunny Lin.

Operator
Sunny Lin, UBS.

Sunny Lin -- UBS AG, Analyst
I am worried about the weakness in demand.

C.C. Wei -- Chairman and Chief Executive Officer
We see robust demand.
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


def test_dashed_layout_from_pdf_converter():
    t = parse_transcript(DASHED)
    assert t.qa_boundary_source == "marker"
    assert roles(t) == [
        ("Jeff Su", "ir", "prepared"),
        ("C.C. Wei", "executive", "prepared"),
        ("Jeff Su", "ir", "qa"),
        ("Operator", "operator", "qa"),
        ("Sunny Lin", "analyst", "qa"),
        ("C.C. Wei", "executive", "qa"),
    ]
    assert "Questions and Answers" not in " ".join(u.text for u in t.utterances)
    assert t.text(Section.PREPARED) == "Demand is strong. We are optimistic.\nCapacity may be tight."


TSM_2Q26 = REPO / "research" / "peer_options" / "data" / "raw" / "transcripts" / "TSM_2Q26.txt"


@pytest.mark.skipif(not TSM_2Q26.exists(), reason="converted TSMC transcript not present (gitignored)")
def test_real_tsmc_2q26_transcript():
    t = parse_transcript(TSM_2Q26.read_text())
    assert t.qa_boundary_source == "marker"
    prepared = [u for u in t.utterances if u.section is Section.PREPARED]
    assert [u.speaker for u in prepared] == ["Jeff Su", "Wendell Huang", "C.C. Wei"]
    assert [u.role.value for u in prepared] == ["ir", "executive", "executive"]
    qa_analysts = {u.speaker for u in t.utterances if u.section is Section.QA and u.role is Role.ANALYST}
    assert len(qa_analysts) == 11 and "Sunny Lin" in qa_analysts
    qa_exec = {u.speaker for u in t.utterances if u.section is Section.QA and u.role is Role.EXECUTIVE}
    assert qa_exec <= {"C.C. Wei", "Wendell Huang"}
    assert 1500 < len(t.text(Section.PREPARED).split()) < 4000
    assert 2000 < len(t.text(Section.QA).split()) < 8000
    assert "LSEG" not in " ".join(u.text for u in t.utterances)
    assert "added by company" not in " ".join(u.text for u in t.utterances)


TRANSCRIPT_DIR = REPO / "research" / "peer_options" / "data" / "raw" / "transcripts"


@pytest.mark.skipif(not (TRANSCRIPT_DIR / "TSM_2Q25.txt").exists(), reason="converted transcripts not present (gitignored)")
@pytest.mark.parametrize("period", ["2Q25", "3Q25", "4Q25", "1Q26", "2Q26"])
def test_all_converted_tsmc_transcripts(period):
    text = (TRANSCRIPT_DIR / f"TSM_{period}.txt").read_text()
    t = parse_transcript(text)
    assert t.qa_boundary_source == "marker"
    by_role = lambda r: {u.speaker for u in t.utterances if u.role is r}
    assert by_role(Role.IR) == {"Jeff Su"}
    assert by_role(Role.EXECUTIVE) == {"C.C. Wei", "Wendell Huang"}
    assert len(by_role(Role.ANALYST)) >= 7
    assert all(u.section is Section.QA for u in t.utterances if u.role is Role.ANALYST)
    assert 1500 < len(t.text(Section.PREPARED).split()) < 4500
    assert 1500 < len(t.text(Section.QA).split()) < 8000
    for noise in ("LSEG", "Refinitiv", "StreetEvents", "©", "company after the call", "foreign language", "(Chinese"):
        assert noise.lower() not in text.lower(), noise
