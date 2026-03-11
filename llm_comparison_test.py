"""
Code extracted from llm_comparison_test.ipynb
Cells: 14 total, 6 code
"""

import json
import re
import requests
import time

time.sleep(5)


SYSTEM_PROMPT = """You are an expert resume parser built for an Applicant Tracking System. You extract structured data from resumes with perfect accuracy. You must return ONLY valid JSON. No explanations, no markdown fences, no extra text."""

PARSE_PROMPT_TEMPLATE = """Extract ALL information from the resume below into the exact JSON structure shown. Follow every instruction carefully.

RULES:
- Use null for missing fields, empty arrays [] where no items found.
- Be thorough — capture every skill, every experience, every detail.
- The LAST word of the full name is ALWAYS the LastName. The FIRST word is ALWAYS the FirstName.
- For ExperienceInYears per role: calculate (EndYear - StartYear) + (EndMonth - StartMonth)/12, rounded to 1 decimal. Use "Present" = March 2026.
- For each skill: SkillExperienceInMonths = sum of months for each role where the skill was used. LastUsed = EndDate of the most recent role where it was used.
- Certifications are NOT skills. Spoken languages are NOT skills.
- Return ONLY valid JSON. No markdown, no explanations.

{
  "PersonalDetails": {
    "FullName": "", "FirstName": "", "MiddleName": null, "LastName": "",
    "EmailID": "", "PhoneNumber": "", "CountryCode": "", "Location": "",
    "LinkedIn": "", "GitHub": "", "Portfolio": ""
  },
  "OverallSummary": {
    "Summary": "", "CurrentJobRole": "", "RelevantJobTitles": [], "TotalExperience": "", "Domain": ""
  },
  "ListOfExperiences": [
    {
      "JobTitle": "", "CompanyName": "", "Location": "",
      "StartDate": "", "EndDate": "", "ExperienceInYears": "",
      "Summary": "", "KeyResponsibilities": []
    }
  ],
  "ListOfSkills": [
    {"SkillName": "", "SkillExperienceInMonths": 0, "LastUsed": "", "RelevantSkills": []}
  ],
  "PrimarySkills": [],
  "SecondarySkills": [],
  "ListOfEducation": [
    {"Degree": "", "Field": "", "Institution": "", "Location": "", "YearPassed": "", "GPA": ""}
  ],
  "Certifications": [
    {"CertificationName": "", "IssuerName": "", "IssuedYear": ""}
  ],
  "Projects": [
    {"ProjectName": "", "Description": "", "Technologies": [], "RoleInProject": ""}
  ],
  "Achievements": [],
  "Languages": []
}

RESUME:
---
RESUME_TEXT_HERE
---

Return ONLY the JSON object. No other text."""


# === TEST RESUMES ===
# Add your resume text here. We include 2 samples — add more for thorough testing.

TEST_RESUMES = {
    "software_engineer": """SHREYAS REDDY
Renton, WA | krishnareddyskr@outlook.com | linkedin.com/in/shreyaskreddy

EXPERIENCE

AI Engineer | Arytic, Inc — Renton, WA | Aug 2025 – Present
• Architected document processing platform with microservices design, building automated validation pipelines across 42 data fields with 99.7% accuracy.
• Designed pluggable provider architecture supporting multiple LLM backends (Groq, OpenAI, Ollama).
• Built horizontally scalable REST APIs using FastAPI with async concurrency, Docker containerization, and PostgreSQL/Redis data layer.

Software Engineer | Giving Forward | Jul 2025 – Aug 2025
• Engineered data pipelines processing financial data across 62+ organizations ($425M+ AUM).
• Built search system with TF-IDF vectorization and LLM integration, achieving sub-2 second query response.

Research Assistant | Texas A&M University — Corpus Christi, TX | Aug 2024 – May 2025
• Built hybrid analysis system integrating Neo4j graph database with multi-hop retrieval pipeline.

EDUCATION
MS in Computer Science | Texas A&M University — Corpus Christi, TX | 2025
Bachelor of Commerce (Honors), Finance | Alliance School of Business — Bengaluru, India | 2019

SKILLS
Languages: Python, Java, JavaScript, C++, SQL, HTML/CSS, Bash
Frameworks: FastAPI, Flask, React, Node.js, Django
Tools: Docker, AWS, Azure, Git, Linux, PostgreSQL, Redis, MongoDB, Neo4j
ML/AI: PyTorch, scikit-learn, Pandas, NumPy

CERTIFICATIONS
Prompt Design in Vertex AI — Google Cloud
Finetuning Large Language Models — DeepLearning.AI""",

    "project_manager": """JOHN SMITH
New York, NY | john.smith@email.com | +1 (555) 123-4567 | linkedin.com/in/johnsmith

PROFESSIONAL SUMMARY
Senior Project Manager with 12+ years of experience leading cross-functional teams in financial services and healthcare IT. PMP and CSM certified. Expertise in Agile/Scrum, Waterfall, and hybrid methodologies.

EXPERIENCE

Senior Project Manager | JPMorgan Chase — New York, NY | Mar 2019 – Present
• Led portfolio of 8 concurrent projects with combined budget of $15M across digital banking initiatives.
• Managed team of 25 engineers, designers, and analysts across 3 time zones.
• Reduced project delivery time by 30% through implementation of SAFe framework.
• Drove migration of legacy systems to cloud infrastructure (AWS), achieving 99.9% uptime.

Project Manager | Deloitte Consulting — New York, NY | Jun 2015 – Feb 2019
• Managed healthcare IT implementations for 5 hospital systems (Epic, Cerner).
• Delivered $8M EHR integration project 2 weeks ahead of schedule.
• Established PMO processes adopted across the consulting practice.

Business Analyst | Accenture — Chicago, IL | Jan 2012 – May 2015
• Gathered requirements for insurance claims processing system serving 2M+ customers.
• Created detailed BRDs, process flows, and user stories for Agile development teams.

EDUCATION
MBA | Columbia Business School — New York, NY | 2015
BS in Information Systems | University of Illinois — Urbana-Champaign, IL | 2011

CERTIFICATIONS
Project Management Professional (PMP) — PMI, 2016
Certified ScrumMaster (CSM) — Scrum Alliance, 2018
SAFe 5 Agilist — Scaled Agile, 2021

SKILLS
Project Management, Agile, Scrum, SAFe, Waterfall, JIRA, Confluence, MS Project, Stakeholder Management, Risk Management, Budgeting, Vendor Management, AWS, SQL, Tableau, Power BI"""
}

print(f"Loaded {len(TEST_RESUMES)} test resumes")


OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS = ["llama3.1:8b", "qwen2.5:7b"]

def extract_json(text):
    """Extract JSON from model response."""
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def run_model(model_name, resume_text):
    """Run a single model on a resume and return results."""
    prompt = PARSE_PROMPT_TEMPLATE.replace("RESUME_TEXT_HERE", resume_text)

    start = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 8192,
        }
    }, timeout=300)
    elapsed = round(time.time() - start, 1)

    if resp.status_code != 200:
        return {"error": resp.text, "time_sec": elapsed}

    data = resp.json()
    content = data["message"]["content"]
    parsed = extract_json(content)
    tokens = data.get("eval_count", 0)
    tokens_per_sec = round(tokens / elapsed, 1) if elapsed > 0 else 0

    return {
        "parsed": parsed,
        "json_valid": parsed is not None,
        "time_sec": elapsed,
        "tokens": tokens,
        "tokens_per_sec": tokens_per_sec,
        "raw_response": content,
    }


# Run all combinations
results = {}
for model in MODELS:
    results[model] = {}
    print(f"\n{'='*60}")
    print(f"Running: {model}")
    print(f"{'='*60}")
    for resume_name, resume_text in TEST_RESUMES.items():
        print(f"  Parsing: {resume_name}...", end=" ", flush=True)
        result = run_model(model, resume_text)
        results[model][resume_name] = result
        status = "✓ valid JSON" if result.get("json_valid") else "✗ INVALID JSON"
        print(f"{status} | {result['time_sec']}s | {result.get('tokens_per_sec', 0)} t/s")

print("\n\nAll runs complete.")

def score_result(parsed, resume_name):
    """Score parsing quality on key metrics."""
    if not parsed:
        return {"total": 0, "details": "JSON parsing failed"}

    scores = {}

    # 1. Personal details
    pd = parsed.get("PersonalDetails", {})
    name_ok = bool(pd.get("FullName") and pd.get("FirstName") and pd.get("LastName"))
    email_ok = bool(pd.get("EmailID"))
    location_ok = bool(pd.get("Location"))
    scores["personal_details"] = sum([name_ok, email_ok, location_ok]) / 3

    # 2. Experiences
    exps = parsed.get("ListOfExperiences", [])
    expected_exp = {"software_engineer": 3, "project_manager": 3}
    exp_count_ok = len(exps) == expected_exp.get(resume_name, 0)
    exp_has_titles = all(e.get("JobTitle") for e in exps) if exps else False
    exp_has_companies = all(e.get("CompanyName") for e in exps) if exps else False
    exp_has_dates = all(e.get("StartDate") for e in exps) if exps else False
    exp_has_responsibilities = all(e.get("KeyResponsibilities") for e in exps) if exps else False
    scores["experiences"] = sum([exp_count_ok, exp_has_titles, exp_has_companies, exp_has_dates, exp_has_responsibilities]) / 5

    # 3. Skills
    skills = parsed.get("ListOfSkills", [])
    primary = parsed.get("PrimarySkills", [])
    secondary = parsed.get("SecondarySkills", [])
    min_skills = {"software_engineer": 15, "project_manager": 8}
    skill_count_ok = len(skills) >= min_skills.get(resume_name, 5)
    has_primary = len(primary) > 0
    has_secondary = len(secondary) > 0
    skills_have_names = all(s.get("SkillName") for s in skills) if skills else False
    scores["skills"] = sum([skill_count_ok, has_primary, has_secondary, skills_have_names]) / 4

    # 4. Education
    edu = parsed.get("ListOfEducation", [])
    expected_edu = {"software_engineer": 2, "project_manager": 2}
    edu_count_ok = len(edu) == expected_edu.get(resume_name, 0)
    edu_has_degrees = all(e.get("Degree") for e in edu) if edu else False
    edu_has_institutions = all(e.get("Institution") for e in edu) if edu else False
    scores["education"] = sum([edu_count_ok, edu_has_degrees, edu_has_institutions]) / 3

    # 5. Certifications
    certs = parsed.get("Certifications", [])
    expected_certs = {"software_engineer": 2, "project_manager": 3}
    cert_count_ok = len(certs) == expected_certs.get(resume_name, 0)
    certs_have_names = all(c.get("CertificationName") for c in certs) if certs else False
    scores["certifications"] = sum([cert_count_ok, certs_have_names]) / 2

    # 6. No hallucination check (certs should NOT be in skills)
    cert_names_lower = {c.get("CertificationName", "").lower() for c in certs}
    skill_names_lower = {s.get("SkillName", "").lower() for s in skills}
    cert_in_skills = cert_names_lower & skill_names_lower
    scores["no_hallucination"] = 1.0 if len(cert_in_skills) == 0 else 0.0

    # 7. Summary
    summary = parsed.get("OverallSummary", {})
    has_role = bool(summary.get("CurrentJobRole"))
    has_summary = bool(summary.get("Summary"))
    has_experience = bool(summary.get("TotalExperience"))
    scores["summary"] = sum([has_role, has_summary, has_experience]) / 3

    # Weighted total
    weights = {
        "personal_details": 0.10,
        "experiences": 0.25,
        "skills": 0.25,
        "education": 0.10,
        "certifications": 0.10,
        "no_hallucination": 0.10,
        "summary": 0.10,
    }
    total = sum(scores[k] * weights[k] for k in weights)
    scores["total"] = round(total * 100, 1)

    return scores


# Score all results
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

for resume_name in TEST_RESUMES:
    print(f"\n--- Resume: {resume_name} ---\n")
    print(f"{'Metric':<25} {'Llama 3.1 8B':>15} {'Qwen 2.5 7B':>15}")
    print("-" * 57)

    llama_result = results["llama3.1:8b"][resume_name]
    qwen_result = results["qwen2.5:7b"][resume_name]

    llama_scores = score_result(llama_result.get("parsed"), resume_name)
    qwen_scores = score_result(qwen_result.get("parsed"), resume_name)

    # Print scores
    for metric in ["personal_details", "experiences", "skills", "education", "certifications", "no_hallucination", "summary"]:
        ls = llama_scores.get(metric, 0)
        qs = qwen_scores.get(metric, 0)
        ls_str = f"{ls:.0%}" if isinstance(ls, float) else str(ls)
        qs_str = f"{qs:.0%}" if isinstance(qs, float) else str(qs)
        winner = " ←" if ls > qs else (" →" if qs > ls else "")
        print(f"{metric:<25} {ls_str:>15} {qs_str:>15}{winner}")

    print("-" * 57)
    print(f"{'QUALITY SCORE':<25} {llama_scores['total']:>14}% {qwen_scores['total']:>14}%")
    print(f"{'JSON Valid':<25} {str(llama_result.get('json_valid')):>15} {str(qwen_result.get('json_valid')):>15}")
    print(f"{'Time (seconds)':<25} {llama_result['time_sec']:>15} {qwen_result['time_sec']:>15}")
    print(f"{'Tokens/sec':<25} {llama_result.get('tokens_per_sec', 0):>15} {qwen_result.get('tokens_per_sec', 0):>15}")
    print(f"{'Total tokens':<25} {llama_result.get('tokens', 0):>15} {qwen_result.get('tokens', 0):>15}")


# Overall winner
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

llama_avg = sum(score_result(results["llama3.1:8b"][r].get("parsed"), r)["total"] for r in TEST_RESUMES) / len(TEST_RESUMES)
qwen_avg = sum(score_result(results["qwen2.5:7b"][r].get("parsed"), r)["total"] for r in TEST_RESUMES) / len(TEST_RESUMES)
llama_time = sum(results["llama3.1:8b"][r]["time_sec"] for r in TEST_RESUMES) / len(TEST_RESUMES)
qwen_time = sum(results["qwen2.5:7b"][r]["time_sec"] for r in TEST_RESUMES) / len(TEST_RESUMES)
llama_json_rate = sum(1 for r in TEST_RESUMES if results["llama3.1:8b"][r].get("json_valid")) / len(TEST_RESUMES)
qwen_json_rate = sum(1 for r in TEST_RESUMES if results["qwen2.5:7b"][r].get("json_valid")) / len(TEST_RESUMES)

print(f"\n{'Metric':<25} {'Llama 3.1 8B':>15} {'Qwen 2.5 7B':>15}")
print("-" * 57)
print(f"{'Avg Quality Score':<25} {llama_avg:>14.1f}% {qwen_avg:>14.1f}%")
print(f"{'JSON Success Rate':<25} {llama_json_rate:>14.0%} {qwen_json_rate:>14.0%}")
print(f"{'Avg Time (seconds)':<25} {llama_time:>15.1f} {qwen_time:>15.1f}")

print(f"\n{'='*57}")
if qwen_avg > llama_avg + 2:
    print("WINNER: Qwen 2.5 7B")
elif llama_avg > qwen_avg + 2:
    print("WINNER: Llama 3.1 8B")
else:
    print("RESULT: Too close to call — check individual metrics above")
print(f"{'='*57}")

# Uncomment a line below to see the full parsed output for any model + resume combo

# print(json.dumps(results["llama3.1:8b"]["software_engineer"]["parsed"], indent=2))
# print(json.dumps(results["qwen2.5:7b"]["software_engineer"]["parsed"], indent=2))
# print(json.dumps(results["llama3.1:8b"]["project_manager"]["parsed"], indent=2))
# print(json.dumps(results["qwen2.5:7b"]["project_manager"]["parsed"], indent=2))

if __name__ == "__main__":
    print("Notebook code extracted successfully")