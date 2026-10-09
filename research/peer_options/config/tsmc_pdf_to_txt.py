#!/usr/bin/env python3
"""Convert a Refinitiv StreetEvents earnings-call PDF/text (TSMC IR layout) into Seeking-Alpha-style text.

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
Page headers, footers, copyright, page numbers and the closing disclaimer are removed.
Input may be .pdf (needs `pdftotext` or `pypdf`) or already-extracted .txt.
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
    r"^\f",
]
NOISE_RE = [re.compile(p, re.I) for p in NOISE]
# "Earnings" running-header regex above is too greedy for body text; only apply it to lines that look like headers.
HEADER_EARNINGS = re.compile(r"^\S.*\d{4} (Taiwan|[A-Z]).*Earnings\s*$")


def extract_text(path):
    if path.lower().endswith(".txt"):
        return open(path, encoding="utf-8", errors="replace").read()
    try:
        return subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"], capture_output=True, text=True,
                              check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("Need `pdftotext` (poppler) or `pip install pypdf`.")
    return "\n\f".join((p.extract_text() or "") for p in PdfReader(path).pages)


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
        if re.match(r"^CORPORATE PARTICIPANTS", s, re.I): mode = corp; continue
        if re.match(r"^CONFERENCE CALL PARTICIPANTS", s, re.I): mode = anal; continue
        if re.match(r"^PRESENTATION\s*$", s, re.I): break
        if mode is not None and s:
            parts = [p.strip() for p in s.split(" - ")]
            if len(parts) >= 3:
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
            return f"{name} -- {' - '.join(parts[2:])}" if len(parts) == 3 and "analyst" not in parts[2].lower() \
                else f"{name} -- {parts[1]}, {parts[2]}"
    return None


def convert(text):
    start, period = parse_event(text)
    raw = [l for l in text.replace("\r", "").split("\n")]
    corp, anal = parse_participants(raw)
    lines = [l for l in raw if not is_noise(l)]
    full = "\n".join(lines)
    # slice body
    pm = re.search(r"^\s*PRESENTATION\s*$", full, re.M)
    qm = re.search(r"^\s*QUESTIONS AND ANSWERS\s*$", full, re.M)
    dm = re.search(r"^\s*DISCLAIMER\s*$", full, re.M)
    if not pm: sys.exit("No PRESENTATION heading found.")
    if not qm: sys.exit("No 'QUESTIONS AND ANSWERS' heading found.")
    prepared_txt = full[pm.end():qm.start()]
    qa_txt = full[qm.end():(dm.start() if dm else len(full))]

    def turns(block):
        out, cur = [], None
        for ln in block.split("\n"):
            lab = speaker_label(ln, corp, anal)
            if lab:
                cur = [lab, []]; out.append(cur); continue
            if cur is None:
                if ln.strip(): cur = ["Operator", []]; out.append(cur)
                else: continue
            cur[1].append(ln)
        res = []
        for lab, ls in out:
            paras, buf = [], []
            for l in ls:
                if l.strip():
                    buf.append(l.strip())
                elif buf:
                    paras.append(" ".join(buf)); buf = []
            if buf: paras.append(" ".join(buf))
            paras = [re.sub(r"\s+", " ", p) for p in paras]
            paras = [re.sub(r"(\w)- (\w)", r"\1-\2", p) if False else p for p in paras]
            if paras: res.append((lab, paras))
        return res

    prep, qa = turns(prepared_txt), turns(qa_txt)
    o = ["Company Participants"]
    o += [f"{n} - {t}" for n, (_, t) in corp.items()]
    o += ["", "Conference Call Participants"]
    o += [f"{n} - {f}" for n, (f, _) in anal.items()]
    for head, tt in (("Prepared Remarks:", prep), ("Questions and Answers:", qa)):
        o += ["", head]
        for lab, paras in tt:
            o += ["", lab] + paras
    return "\n".join(o).strip() + "\n", start, period, len(prep), len(qa), sum(len(" ".join(p).split()) for _, p in prep), \
        sum(len(" ".join(p).split()) for _, p in qa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("--ticker", default="TSM")
    ap.add_argument("--outdir", default="research/peer_options/data/raw/transcripts")
    ap.add_argument("--source", default=""); ap.add_argument("--published-utc", default="")
    a = ap.parse_args()
    txt, start, period, np_, nq, wp, wq = convert(extract_text(a.input))
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
    print(f"{fname}: prepared {np_} turns / {wp} words, Q&A {nq} turns / {wq} words, call_start_utc={start:%Y-%m-%dT%H:%MZ}")
    if np_ < 3 or nq < 6 or wp < 1500 or wq < 1500:
        print("WARNING: counts look low; open the output and check the speaker lines and section split.", file=sys.stderr)


if __name__ == "__main__":
    main()
