#!/usr/bin/env python3
"""Convert a Refinitiv / LSEG StreetEvents earnings-call PDF/text (TSMC IR layout) into Seeking-Alpha-style text.

Usage:
  python3 tsmc_pdf_to_txt.py INPUT.pdf --ticker TSM --outdir research/peer_options/data/raw/transcripts \
      [--source URL] [--published-utc 2025-10-16T07:30:00Z]

Writes:
  <outdir>/TSM_3Q25.txt        (file name comes from the event header: ticker + period)
  <outdir>/manifest.csv        (appends or updates the row: file,ticker,period,call_start_utc,published_utc,source)

Output layout (what the parser should see):
  Company Participants / Conference Call Participants lists
  Prepared Remarks:
  <Speaker -- Title>        one speaker line, then that speaker's paragraphs
  ...
  Questions and Answers:
  <Speaker -- Title>
  ...
Page headers, footers, copyright, page numbers and the closing disclaimer are removed, as are passages the
company marked "(added by company after the call)" (not call speech). Letter-spaced headings
("P R E S E N T A T I O N") are supported. If --published-utc is not given it is read from the LSEG footer
("(c)2026, LSEG ... <id>-2026-07-17T05:19:36.480"), which is when LSEG generated the transcript.
Input may be .pdf (needs `pymupdf`, `pdftotext` or `pypdf`; pypdf drops spaces on LSEG PDFs) or already-extracted .txt.
"""
import argparse, csv, os, re, subprocess, sys
from datetime import datetime, timezone

MONTHS = {m: i for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER",
     "NOVEMBER", "DECEMBER"], 1)}

NOISE = [
    r"^REFINITIV STREETEVENTS", r"^EDITED TRANSCRIPT\s*$", r"^\s*\d{1,3}\s*$", r"^Call \(.*\)\s*$",
    r"^(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER) \d{1,2}, \d{4} / \d{1,2}:\d{2}\s*[AP]M,",
    r"^.*Earnings\s*$", r"^©", r"^Copyright", r"Republication or redistribution", r"^All rights reserved",
    r"^Contact Us\s*$", r"www\.refinitiv\.com", r"^THOMSON REUTERS", r"^Refinitiv\b.*(Terms|Privacy)",
    r"^LSEG STREETEVENTS", r"^(prior written )?consent of (LSEG|Refinitiv)", r"^\(Chinese, English\)\s*$",
    r"\b(LSEG|Refinitiv|REFINITIV)\b.*(reserved|trademarks|STREETEVENTS)", r"^[A-Z]+ \d{1,2}, \d{4} / \d{1,2}:\d{2}\s*[AP]M, ",
]
HEADINGS = {"CORPORATEPARTICIPANTS": "corp", "CONFERENCECALLPARTICIPANTS": "anal", "PRESENTATION": "pres",
            "QUESTIONSANDANSWERS": "qa", "DISCLAIMER": "disc"}
# a run of consecutive [bracketed] blocks followed by the marker (the marker sits after the last block only)
# Bracketed runs ending in the marker are dropped whole. Bare inline tags (incl. "corrected by company") are removed but the
# adjacent word stays, since the PDF does not delimit which word was inserted or corrected.
AFTER_CALL = re.compile(r"(?:\[[^\]]*\]\.?\s*)+\(added by company after the call\)|\((?:added|corrected) by company after the call\)", re.S)
STAGE = re.compile(r"\((?:Spoken in Foreign Language|Event Instructions)\)", re.I)
PUBLISHED = re.compile(r"All Rights Reserved\.\s*\d+-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
NOISE_RE = [re.compile(p, re.I) for p in NOISE]
# "Earnings" running-header regex above is too greedy for body text; only apply it to lines that look like headers.
HEADER_EARNINGS = re.compile(r"^\S.*\d{4} (Taiwan|[A-Z]).*Earnings\s*$")


def extract_text(path):
    if path.lower().endswith(".txt"):
        return open(path, encoding="utf-8", errors="replace").read()
    try:  # PyMuPDF keeps word spacing on the LSEG layout, where pypdf drops spaces
        import pymupdf
        with pymupdf.open(path) as doc:
            return "\n\f".join(page.get_text() for page in doc)
    except ImportError:
        pass
    try:
        return subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"], capture_output=True, text=True,
                              check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("Need `pymupdf` (preferred), `pdftotext` (poppler) or `pypdf`.")
    return "\n\f".join((p.extract_text() or "") for p in PdfReader(path).pages)


def heading(line):
    """Canonical heading key for 'PRESENTATION' or letter-spaced 'P R E S E N T A T I O N', else None."""
    s = line.strip()
    if not s or len(s) > 70 or not re.fullmatch(r"[A-Za-z ]+", s):
        return None
    return HEADINGS.get(re.sub(r"\s+", "", s).upper())


def is_noise(line):
    s = line.strip()
    if not s:
        return False
    if HEADER_EARNINGS.match(s) and "/" not in s and len(s) < 140 and re.search(r"Q[1-4] \d{4}", s):
        return True
    for r in NOISE_RE:
        if r.pattern == r"^.*Earnings\s*$":
            continue
        if r.search(s):
            return True
    return False


def parse_event(text):
    m = re.search(r"EVENT DATE/TIME:\s*([A-Z]+) (\d{1,2}), (\d{4}) / (\d{1,2}):(\d{2})\s*([AP]M)\s*(GMT|UTC)", text, re.I)
    if not m:
        sys.exit("Could not find 'EVENT DATE/TIME: MONTH D, YYYY / H:MMAM GMT' in the header.")
    mon, d, y, hh, mm, ap = m.group(1).upper(), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6).upper()
    hh = hh % 12 + (12 if ap == "PM" else 0)
    start = datetime(y, MONTHS[mon], d, hh, mm, tzinfo=timezone.utc)
    q = re.search(r"\bQ([1-4])\s+(\d{4})\b", text)
    if not q:
        sys.exit("Could not find the 'Qn YYYY' period in the header.")
    period = f"{q.group(1)}Q{q.group(2)[2:]}"
    return start, period


def parse_participants(lines):
    corp, anal = {}, {}
    mode = None
    for ln in lines:
        s = ln.strip()
        h = heading(s)
        if h == "corp": mode = corp; continue
        if h == "anal": mode = anal; continue
        if h == "pres": break
        if mode is not None and s:
            parts = [p.strip() for p in s.split(" - ")]
            if len(parts) >= 3 and len(parts[0]) < 45:
                mode[parts[0]] = (parts[1], " - ".join(parts[2:]))
    return corp, anal


def speaker_label(line, corp, anal):
    s = line.strip()
    if s == "Operator":
        return "Operator"
    parts = [p.strip() for p in s.split(" - ")]
    if len(parts) >= 3 and len(parts[0]) < 45 and len(s) < 180 and not s.endswith("."):
        name = parts[0]
        if name in corp:
            return f"{name} -- {corp[name][1]}"
        if name in anal:
            return f"{name} -- {anal[name][0]}, {anal[name][1]}"
        if name[:1].isupper() and not re.search(r"[.!?]$", name):
            title = " - ".join(parts[2:])
            return f"{name} -- {parts[1]}, {title}" if "analyst" in title.lower() else f"{name} -- {title}"
    return None


def participants_from_labels(lines, corp, anal):
    """LSEG header lists have no ' - ' separators, so build the lists from 'Name - Company - Title' speaker lines."""
    for ln in lines:
        parts = [p.strip() for p in ln.strip().split(" - ")]
        if len(parts) >= 3 and speaker_label(ln, {}, {}) not in (None, "Operator"):
            name, company, title = parts[0], parts[1], " - ".join(parts[2:])
            (anal if "analyst" in title.lower() else corp).setdefault(name, (company, title))


def convert(text):
    published = PUBLISHED.search(text)
    text = text.replace("\f", "")
    start, period = parse_event(text)
    raw = [l for l in text.replace("\r", "").split("\n")]
    corp, anal = parse_participants(raw)
    lines = [l for l in raw if not is_noise(l)]
    hi = {}
    for n, l in enumerate(lines):
        h = heading(l)
        if h and h not in hi and (h not in ("corp", "anal") or not hi.get("pres")):
            hi[h] = n
    if "pres" not in hi: sys.exit("No PRESENTATION heading found.")
    if "qa" not in hi: sys.exit("No 'QUESTIONS AND ANSWERS' heading found.")
    if not corp and not anal:
        participants_from_labels(lines, corp, anal)
    prepared_lines = lines[hi["pres"] + 1:hi["qa"]]
    qa_lines = lines[hi["qa"] + 1:hi.get("disc", len(lines))]
    removed = [0]

    def clean(p):
        removed[0] += len(AFTER_CALL.findall(p))
        return re.sub(r"\s+", " ", STAGE.sub("", AFTER_CALL.sub("", p))).strip()

    def turns(block_lines):
        out, cur = [], None
        for ln in block_lines:
            lab = speaker_label(ln, corp, anal)
            if lab:
                cur = [lab, []]; out.append(cur); continue
            if cur is None:
                if ln.strip(): cur = ["Operator", []]; out.append(cur)
                else: continue
            cur[1].append(ln)
        res = []
        for lab, ls in out:
            # join the turn first so a bracketed after-call insert spanning lines can be removed whole
            paras, buf = [], []
            for l in ls:
                if l.strip():
                    buf.append(l.strip())
                elif buf:
                    paras.append(" ".join(buf)); buf = []
            if buf: paras.append(" ".join(buf))
            paras = [c for c in (clean(p) for p in paras) if c]
            if paras: res.append((lab, paras))
        return res

    prep, qa = turns(prepared_lines), turns(qa_lines)
    o = ["Company Participants"]
    o += [f"{n} - {t}" for n, (_, t) in corp.items()]
    o += ["", "Conference Call Participants"]
    o += [f"{n} - {f}" for n, (f, _) in anal.items()]
    for head, tt in (("Prepared Remarks:", prep), ("Questions and Answers:", qa)):
        o += ["", head]
        for lab, paras in tt:
            o += ["", lab] + paras
    pub = published.group(1) + "Z" if published else ""
    return ("\n".join(o).strip() + "\n", start, period, len(prep), len(qa),
            sum(len(" ".join(p).split()) for _, p in prep), sum(len(" ".join(p).split()) for _, p in qa),
            pub, removed[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("--ticker", default="TSM")
    ap.add_argument("--outdir", default="research/peer_options/data/raw/transcripts")
    ap.add_argument("--source", default=""); ap.add_argument("--published-utc", default="")
    a = ap.parse_args()
    txt, start, period, np_, nq, wp, wq, pub, removed = convert(extract_text(a.input))
    a.published_utc = a.published_utc or pub
    os.makedirs(a.outdir, exist_ok=True)
    fname = f"{a.ticker}_{period}.txt"
    open(os.path.join(a.outdir, fname), "w", encoding="utf-8").write(txt)
    mpath = os.path.join(a.outdir, "manifest.csv")
    cols = ["file", "ticker", "period", "call_start_utc", "published_utc", "source"]
    rows = []
    if os.path.exists(mpath):
        rows = [r for r in csv.DictReader(open(mpath)) if r["file"] != fname]
    rows.append(dict(file=fname, ticker=a.ticker, period=period, call_start_utc=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     published_utc=a.published_utc, source=a.source))
    rows.sort(key=lambda r: (r["ticker"], r["call_start_utc"]))
    w = csv.DictWriter(open(mpath, "w", newline=""), fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"{fname}: prepared {np_} turns / {wp} words, Q&A {nq} turns / {wq} words, call_start_utc={start:%Y-%m-%dT%H:%MZ}, "
          f"published_utc={a.published_utc or '?'}, after-call inserts removed={removed}")
    if np_ < 3 or nq < 6 or wp < 1500 or wq < 1500:
        print("WARNING: counts look low; open the output and check the speaker lines and section split.", file=sys.stderr)


if __name__ == "__main__":
    main()
