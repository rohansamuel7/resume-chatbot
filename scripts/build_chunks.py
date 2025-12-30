import pdfplumber
import json
import os
import re
from datetime import date

PDF_PATH = "data/raw/Rohan Resume MAIN.pdf"
OUT_PATH = "data/processed/chunks.jsonl"

SECTION_HEADERS = {
    "EDUCATION": "education",
    "PROFESSIONAL INTERNSHIP EXPERIENCE": "experience",
    "ACADEMIC RESEARCH EXPERIENCE": "research",
    "PERSONAL PROJECT EXPERIENCE": "project",
    "TECHNICAL SKILLS": "skills",
    "CERTIFICATIONS": "certifications",
}

def extract_text():
    text = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def normalize_lines(text):
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [re.sub(r"\s+", " ", ln) for ln in lines if ln]
    return lines

def split_sections(lines):
    sections = {}
    current = "HEADER"
    sections[current] = []

    for ln in lines:
        if ln in SECTION_HEADERS:
            current = ln
            sections[current] = []
        else:
            sections[current].append(ln)

    return sections

def write_chunk(chunk_id, chunk_type, title, content):
    return {
        "chunk_id": chunk_id,
        "type": chunk_type,
        "title": title,
        "content": content.strip(),
        "meta": {
            "source": "resume_pdf",
            "created_on": str(date.today())
        }
    }

def main():
    os.makedirs("data/processed", exist_ok=True)

    raw_text = extract_text()
    lines = normalize_lines(raw_text)
    sections = split_sections(lines)

    chunks = []

    # EDUCATION
    if "EDUCATION" in sections:
        chunks.append(write_chunk(
            "edu_001",
            "education",
            "Education – Penn State",
            "\n".join(sections["EDUCATION"])
        ))

    # EXPERIENCE
    if "PROFESSIONAL INTERNSHIP EXPERIENCE" in sections:
        chunks.append(write_chunk(
            "exp_001",
            "experience",
            "Business Intelligence Intern – H9",
            "\n".join(sections["PROFESSIONAL INTERNSHIP EXPERIENCE"])
        ))

    # RESEARCH
    if "ACADEMIC RESEARCH EXPERIENCE" in sections:
        chunks.append(write_chunk(
            "res_001",
            "research",
            "Research Assistant – Penn State",
            "\n".join(sections["ACADEMIC RESEARCH EXPERIENCE"])
        ))

    # PROJECT
    if "PERSONAL PROJECT EXPERIENCE" in sections:
        chunks.append(write_chunk(
            "proj_001",
            "project",
            "Tesla Brand Sentiment Analysis",
            "\n".join(sections["PERSONAL PROJECT EXPERIENCE"])
        ))

    # SKILLS
    if "TECHNICAL SKILLS" in sections:
        chunks.append(write_chunk(
            "skills_001",
            "skills",
            "Technical Skills",
            "\n".join(sections["TECHNICAL SKILLS"])
        ))

    # CERTIFICATIONS
    if "CERTIFICATIONS" in sections:
        chunks.append(write_chunk(
            "cert_001",
            "certifications",
            "Certifications",
            "\n".join(sections["CERTIFICATIONS"])
        ))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Created {len(chunks)} chunks at {OUT_PATH}")

if __name__ == "__main__":
    main()
