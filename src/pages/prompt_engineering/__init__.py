from openai import OpenAI
import google.generativeai as genai
import json
import os
import time
import re
import hashlib

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Optional dependency; environment variables may still come from shell/host.
    pass

SYSTEM_PROMPT = "You are a smart assistant to career advisors at the Harvard Extension School. You will reply with JSON only."

CV_TEXT_PLACEHOLDER = "<CV_TEXT>"

SYSTEM_TAILORING = """
You are a smart assistant to career advisors at the Harvard Extension School. Your take is to rewrite
resumes to be more brief and convincing according to the Resumes and Cover Letters guide.
"""

TAILORING_PROMPT = """
Consider the following CV:
<CV_TEXT>

Your task is to rewrite the given CV. Follow these guidelines:
- Be truthful and objective to the experience listed in the CV
- Be specific rather than general
- Rewrite job highlight items using STAR methodology (but do not mention STAR explicitly)
- Fix spelling and grammar errors
- Write to express not impress
- Articulate and don't be flowery
- Prefer active voice over passive voice
- Do not include a summary about the candidate

Improved CV:
"""

BASICS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface Basics {
    name: string;
    email: string;
    phone: string;
    website: string;
    address: string;
}

Write the basics section according to the Basic schema. On the response, include only the JSON.
"""

EDUCATION_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface EducationItem {
    institution: string;
    area: string;
    additionalAreas: string[];
    studyType: string;
    startDate: string;
    endDate: string;
    score: string;
    location: string;
}

interface Education {
    education: EducationItem[];
}


Write the education section according to the Education schema. On the response, include only the JSON.
"""

AWARDS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface AwardItem {
    title: string;
    date: string;
    awarder: string;
    summary: string;
}

interface Awards {
    awards: AwardItem[];
}

Write the awards section according to the Awards schema. Include only the awards section. On the response, include only the JSON.
"""

PROJECTS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface ProjectItem {
    name: string;
    description: string;
    keywords: string[];
    url: string;
}

interface Projects {
    projects: ProjectItem[];
}

Write the projects section according to the Projects schema. Include all projects, but only the ones present in the CV. On the response, include only the JSON.
"""

SKILLS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

type HardSkills = "Programming Languages" | "Tools" | "Frameworks" | "Computer Proficiency";
type SoftSkills = "Team Work" | "Communication" | "Leadership" | "Problem Solving" | "Creativity";
type OtherSkills = string;

Now consider the following TypeScript Interface for the JSON schema:

interface SkillItem {
    name: HardSkills | SoftSkills | OtherSkills;
    keywords: string[];
}

interface Skills {
    skills: SkillItem[];
}

Write the skills section according to the Skills schema. Include only up to the top 4 skill names that are present in the CV and related with the education and work experience. On the response, include only the JSON.
"""

WORK_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface WorkItem {
    company: string;
    position: string;
    startDate: string;
    endDate: string;
    location: string;
    highlights: string[];
}

interface Work {
    work: WorkItem[];
}

Write a work section for the candidate according to the Work schema. Include only the work experience and not the project experience. For each work experience, provide  a company name, position name, start and end date, and bullet point for the highlights. Follow the Harvard Extension School Resume guidelines and phrase the highlights with the STAR methodology
"""

COMBINED_RESUME_PROMPT = """
You are going to generate a complete JSON resume for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface Basics {
    name: string;
    email: string;
    phone: string;
    website: string;
    address: string;
}

interface EducationItem {
    institution: string;
    area: string;
    additionalAreas: string[];
    studyType: string;
    startDate: string;
    endDate: string;
    score: string;
    location: string;
}

interface AwardItem {
    title: string;
    date: string;
    awarder: string;
    summary: string;
}

interface ProjectItem {
    name: string;
    description: string;
    keywords: string[];
    url: string;
}

interface SkillItem {
    name: string;
    keywords: string[];
}

interface WorkItem {
    company: string;
    position: string;
    startDate: string;
    endDate: string;
    location: string;
    highlights: string[];
}

interface Resume {
    basics: Basics;
    education: EducationItem[];
    awards: AwardItem[];
    projects: ProjectItem[];
    skills: SkillItem[];
    work: WorkItem[];
}

Important output rules:
- Return JSON only (no markdown fences, no commentary)
- If a field is missing in CV, return empty string or empty list as appropriate
- Keep data truthful to CV
- Keep "skills" to up to top 4 skill categories and relevant keywords
"""

_LAST_API_CALL_TS = 0.0
MIN_API_CALL_DELAY_SECONDS = 12
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "6"))
BASE_BACKOFF_SECONDS = int(os.getenv("GEMINI_BASE_BACKOFF_SECONDS", "2"))
_LAST_RETRY_ERROR_KIND = None
_LAST_RETRY_ERROR_MESSAGE = None
_RESUME_CACHE = {}


def _extract_retry_delay_seconds(error):
    """Extract retry delay seconds from provider error details when available."""
    text = str(error)
    # Supports messages like: retry_delay { seconds: 32 }
    retry_delay_match = re.search(r"retry_delay.*?seconds:\s*(\d+)", text, re.DOTALL)
    if retry_delay_match:
        return int(retry_delay_match.group(1))
    return None


def _is_daily_quota_exceeded(error):
    text = str(error).lower()
    indicators = [
        "quota exceeded",
        "daily limit",
        "daily quota",
        "quota has been exhausted",
        "resource_exhausted",
    ]
    return any(indicator in text for indicator in indicators) and "per minute" not in text


def _is_quota_or_rate_limit_error(error):
    text = str(error).lower()
    return any(
        indicator in text
        for indicator in ["quota", "rate limit", "429", "resource_exhausted", "insufficient_quota"]
    )


def _throttle_api_calls(min_delay_seconds=MIN_API_CALL_DELAY_SECONDS):
    """Enforce a global minimum delay between API calls."""
    global _LAST_API_CALL_TS
    now = time.time()
    elapsed = now - _LAST_API_CALL_TS
    if elapsed < min_delay_seconds:
        sleep_time = min_delay_seconds - elapsed
        print(f"[INFO] Throttling API calls. Sleeping {sleep_time:.1f}s.")
        time.sleep(sleep_time)
    _LAST_API_CALL_TS = time.time()


def _extract_text_from_response(response):
    if response is None:
        return ""
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "parts") and response.parts:
        first_part = response.parts[0]
        if hasattr(first_part, "text") and first_part.text:
            return first_part.text
    return str(response)


def sanitize_json_text(raw_text):
    cleaned = (raw_text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    # Keep only the outer JSON object content if model added extra text.
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return cleaned[start_idx:end_idx + 1]
    return cleaned


def safe_parse_json(raw_text):
    cleaned = sanitize_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON response: {e}")
        return None


def generate_with_retry(model_instance, prompt):
    """
    Generate Gemini content with retry logic, exponential backoff, and global throttling.
    Returns Gemini response on success, else None.
    For compatibility with existing callers, retry/error metadata is exposed via
    module-level _LAST_RETRY_ERROR_KIND and _LAST_RETRY_ERROR_MESSAGE.
    """
    global _LAST_RETRY_ERROR_KIND, _LAST_RETRY_ERROR_MESSAGE
    _LAST_RETRY_ERROR_KIND = None
    _LAST_RETRY_ERROR_MESSAGE = None

    retries = 3
    for attempt in range(1, retries + 1):
        try:
            _throttle_api_calls()
            print(
                f"[INFO] Calling Gemini model: {getattr(model_instance, 'model_name', 'unknown')} "
                f"(attempt {attempt}/{retries})"
            )
            return model_instance.generate_content(prompt)
        except Exception as e:
            err_text = str(e)
            if _is_daily_quota_exceeded(e):
                print(f"[ERROR] Gemini daily quota exceeded: {e}")
                _LAST_RETRY_ERROR_KIND = "quota_exceeded"
                _LAST_RETRY_ERROR_MESSAGE = "API quota exceeded. Please try later or use another API key."
                # Quota exhaustion should stop retries immediately.
                return None

            retry_delay = _extract_retry_delay_seconds(e)
            backoff_delay = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            sleep_seconds = retry_delay if retry_delay is not None else backoff_delay

            if "429" in err_text or "resource_exhausted" in err_text.lower():
                print(
                    f"[WARN] Gemini rate limited (attempt {attempt}/{retries}). "
                    f"Retrying in {sleep_seconds}s."
                )
            else:
                print(
                    f"[WARN] Gemini call failed (attempt {attempt}/{retries}): {e}. "
                    f"Retrying in {sleep_seconds}s."
                )
            if attempt < retries:
                time.sleep(sleep_seconds)

    print("[ERROR] Max retries exceeded for Gemini request.")
    _LAST_RETRY_ERROR_KIND = "request_failed"
    _LAST_RETRY_ERROR_MESSAGE = "Model request failed after retries. Please try again later."
    return None


def _resume_generation_error(message, error_kind="request_failed", model=None, fallback_used=False):
    return {
        "_error": True,
        "_error_kind": error_kind,
        "_error_message": message,
        "_model_used": model,
        "_fallback_used": fallback_used,
    }


def _resume_cache_key(cv_text, model, model_type):
    payload = f"{model_type}|{model}|{cv_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_basics_from_cv_text(cv_text):
    parsed = _parse_resume_fallback(cv_text)
    contact = parsed.get("contact", {})
    return {
        "name": parsed.get("name", ""),
        "email": contact.get("email", ""),
        "phone": contact.get("phone", ""),
        "website": contact.get("linkedin") or contact.get("github") or "",
        "address": contact.get("location", ""),
    }


def _is_section_header(line):
    normalized = re.sub(r"[^A-Za-z ]", " ", line or "")
    normalized = " ".join(normalized.split()).upper()
    header_map = {
        "EDUCATION": "education",
        "EXPERIENCE": "experience",
        "WORK EXPERIENCE": "experience",
        "PROFESSIONAL EXPERIENCE": "experience",
        "SKILLS": "skills",
        "TECHNICAL SKILLS": "skills",
        "PROJECTS": "projects",
        "CERTIFICATIONS": "certifications",
        "SUMMARY": "summary",
        "PROFESSIONAL SUMMARY": "summary",
        "OBJECTIVE": "summary",
        "ACHIEVEMENTS": "achievements",
    }
    for header, key in header_map.items():
        if normalized == header or normalized.startswith(header + " "):
            return key
    return None


def _extract_contact_info(lines, text):
    top_lines = lines[:12]
    top_text = "\n".join(top_lines)
    full_text = text or ""

    email_match = re.search(r"[\w.+-]+@[\w-]+\.\w+", full_text)
    phone_match = re.search(r"(\+?\d[\d\s\-\(\)]{8,}\d)", full_text)
    linkedin_match = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+", full_text, re.IGNORECASE)
    github_match = re.search(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_]+", full_text, re.IGNORECASE)
    location_match = re.search(
        r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*,\s*[A-Z]{2,}|[A-Z][a-zA-Z]+,\s*[A-Z][a-zA-Z]+)\b",
        top_text,
    )

    return {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(1).strip() if phone_match else "",
        "linkedin": linkedin_match.group(0) if linkedin_match else "",
        "github": github_match.group(0) if github_match else "",
        "location": location_match.group(1) if location_match else "",
    }


def _group_section_lines(lines):
    section_content = {}
    current_section = None

    for line in lines[1:]:
        section = _is_section_header(line)
        if section:
            current_section = section
            section_content.setdefault(current_section, [])
            continue
        if current_section:
            section_content[current_section].append(line)

    return section_content


def _parse_resume_fallback(cv_text):
    lines = [line.strip() for line in (cv_text or "").splitlines() if line.strip()]
    text = cv_text or ""
    parsed = {
        "name": lines[0] if lines else "",
        "contact": _extract_contact_info(lines, text),
        "summary": "",
        "education": [],
        "experience": [],
        "skills": [],
        "projects": [],
        "certifications": [],
        "achievements": [],
    }

    section_content = _group_section_lines(lines)
    if section_content.get("summary"):
        parsed["summary"] = " ".join(section_content.get("summary", []))

    # Skills: split by separators and bullets.
    skill_text = " | ".join(section_content.get("skills", []))
    if skill_text:
        pieces = re.split(r"[,\|\u2022;/]", skill_text)
        parsed["skills"] = [p.strip() for p in pieces if p.strip()]

    # Education parsing.
    for line in section_content.get("education", []):
        item = {"institution": "", "degree": "", "year": "", "gpa": ""}
        year_match = re.search(r"\b(19|20)\d{2}\b(?:\s*[-–]\s*(19|20)\d{2}|(?:\s*-\s*Present))?", line)
        gpa_match = re.search(r"\bGPA[:\s]*([0-4]\.\d{1,2}|[0-9]{1,2}\.\d{1,2})\b", line, re.IGNORECASE)

        parts = [p.strip() for p in re.split(r"\s+\|\s+| - ", line) if p.strip()]
        if parts:
            item["institution"] = parts[0]
        if len(parts) > 1:
            item["degree"] = parts[1]
        if year_match:
            item["year"] = year_match.group(0)
        if gpa_match:
            item["gpa"] = gpa_match.group(1)
        if any(item.values()):
            parsed["education"].append(item)

    # Experience parsing.
    current_job = None
    date_pattern = re.compile(r"\b(19|20)\d{2}\b(?:\s*[-–]\s*(?:Present|(19|20)\d{2}))?", re.IGNORECASE)
    for line in section_content.get("experience", []):
        if re.match(r"^[-*•]\s+", line):
            if current_job is None:
                current_job = {"company": "", "role": "", "duration": "", "bullets": []}
            current_job["bullets"].append(re.sub(r"^[-*•]\s*", "", line).strip())
            continue

        if date_pattern.search(line) or " at " in line.lower() or "|" in line:
            if current_job and any(current_job.values()):
                parsed["experience"].append(current_job)
            current_job = {"company": "", "role": "", "duration": "", "bullets": []}
            current_job["duration"] = date_pattern.search(line).group(0) if date_pattern.search(line) else ""
            if " at " in line.lower():
                role, company = re.split(r"\s+at\s+", line, maxsplit=1, flags=re.IGNORECASE)
                current_job["role"] = role.strip()
                current_job["company"] = company.strip()
            else:
                parts = [p.strip() for p in re.split(r"\s+\|\s+| - ", line) if p.strip()]
                if parts:
                    current_job["company"] = parts[0]
                if len(parts) > 1:
                    current_job["role"] = parts[1]
        elif current_job:
            current_job["bullets"].append(line)

    if current_job and any(current_job.values()):
        parsed["experience"].append(current_job)

    # Projects parsing.
    for line in section_content.get("projects", []):
        project = {"name": "", "description": "", "tech_stack": []}
        if ":" in line:
            name, desc = line.split(":", 1)
            project["name"] = name.strip()
            project["description"] = desc.strip()
        else:
            parts = [p.strip() for p in re.split(r"\s+\|\s+| - ", line, maxsplit=1) if p.strip()]
            project["name"] = parts[0] if parts else ""
            project["description"] = parts[1] if len(parts) > 1 else ""

        tech_match = re.search(r"(Tech(?:nologies)?|Stack)[:\s]+(.+)$", line, re.IGNORECASE)
        if tech_match:
            project["tech_stack"] = [t.strip() for t in re.split(r"[,/|]", tech_match.group(2)) if t.strip()]
        if project["name"] or project["description"]:
            parsed["projects"].append(project)

    parsed["certifications"] = [line for line in section_content.get("certifications", []) if line]
    parsed["achievements"] = [line for line in section_content.get("achievements", []) if line]
    return parsed


def _build_basic_fallback_resume(cv_text, message, model_used):
    parsed = _parse_resume_fallback(cv_text)

    work_items = []
    for exp in parsed.get("experience", []):
        work_items.append({
            "company": exp.get("company", ""),
            "position": exp.get("role", ""),
            "startDate": exp.get("duration", ""),
            "endDate": "",
            "location": "",
            "highlights": exp.get("bullets", []),
        })

    education_items = []
    for edu in parsed.get("education", []):
        education_items.append({
            "institution": edu.get("institution", ""),
            "area": edu.get("degree", ""),
            "additionalAreas": [],
            "studyType": "",
            "startDate": "",
            "endDate": edu.get("year", ""),
            "score": edu.get("gpa", ""),
            "location": "",
        })

    project_items = []
    for project in parsed.get("projects", []):
        project_items.append({
            "name": project.get("name", ""),
            "description": project.get("description", ""),
            "keywords": project.get("tech_stack", []),
            "url": "",
        })

    skills = parsed.get("skills", [])
    skill_items = [{"name": "Skills", "keywords": skills}] if skills else []

    awards = [{"title": cert, "date": "", "awarder": "", "summary": ""} for cert in parsed.get("certifications", [])]
    awards.extend(
        {"title": achievement, "date": "", "awarder": "", "summary": ""}
        for achievement in parsed.get("achievements", [])
    )

    return {
        "basics": {
            **_extract_basics_from_cv_text(cv_text),
            "summary": parsed.get("summary", ""),
            "linkedin": parsed.get("contact", {}).get("linkedin", ""),
            "github": parsed.get("contact", {}).get("github", ""),
        },
        "education": education_items,
        "awards": awards,
        "projects": project_items,
        "skills": skill_items,
        "work": work_items,
        "_meta": {
            "fallback_used": True,
            "model_used": model_used,
            "error_kind": "quota_exceeded",
            "error_message": message,
            "source": "local_fallback",
        },
    }


def _call_gemini_with_fallback(prompt, api_key, primary_model):
    fallback_models = [os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash"), "gemini-1.5-flash-8b"]
    model_candidates = [primary_model] + [m for m in fallback_models if m and m != primary_model]

    genai.configure(api_key=api_key)
    last_error_message = "Model request failed."
    last_error_kind = "request_failed"

    for idx, model_name in enumerate(model_candidates):
        try:
            print(f"[INFO] Using Gemini model: {model_name}")
            model_instance = genai.GenerativeModel(model_name)
            response = generate_with_retry(model_instance, prompt)
            err_msg = _LAST_RETRY_ERROR_MESSAGE
            err_kind = _LAST_RETRY_ERROR_KIND
            if response is not None:
                return response, None, None, model_name, idx > 0

            last_error_message = err_msg or last_error_message
            last_error_kind = err_kind or last_error_kind
            if err_kind == "quota_exceeded":
                print(f"[WARN] Quota exceeded on model {model_name}. Skipping retries and moving to fallback model.")
            else:
                # Non-quota errors may recover by trying fallback model.
                print(f"[WARN] Gemini model {model_name} failed. Attempting fallback model if available.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize/call Gemini model {model_name}: {e}")
            last_error_message = "Model setup failed. Please verify API key and model configuration."
            last_error_kind = "request_failed"

    return None, last_error_message, last_error_kind, model_candidates[-1], len(model_candidates) > 1

def generate_json_resume(cv_text, api_key, model="gpt-4o", model_type="OpenAI"):
    """Generate a JSON resume from a CV text"""
    cache_key = _resume_cache_key(cv_text, model, model_type)
    if cache_key in _RESUME_CACHE:
        print(f"[INFO] Cache hit for {model_type} model={model}. Returning cached resume.")
        return _RESUME_CACHE[cache_key]

    if not api_key:
        provider = "OpenAI" if model_type == "OpenAI" else "Gemini"
        print(f"[ERROR] Missing API key for {provider}.")
        fallback = _build_basic_fallback_resume(
            cv_text,
            f"Missing {provider} API key. Add it in your environment or UI input.",
            model,
        )
        _RESUME_CACHE[cache_key] = fallback
        return fallback

    if model_type == "OpenAI":
        timeout = int(os.getenv("OPENAI_TIMEOUT", "60"))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        base_url = os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com")

        print(f"[INFO] Using OpenAI model: {model}")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )

    # Single combined prompt to reduce API usage and avoid quota exhaustion.
    combined_prompt = COMBINED_RESUME_PROMPT.replace(CV_TEXT_PLACEHOLDER, cv_text)

    try:
        if model_type == "OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": combined_prompt},
                ],
            )
            answer = response.choices[0].message.content
            model_used = model
            fallback_used = False
        elif model_type == "Gemini":
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {combined_prompt}\nAssistant:"
            response, err_msg, err_kind, model_used, fallback_used = _call_gemini_with_fallback(
                full_prompt,
                api_key,
                model,
            )
            if response is None:
                print(f"[ERROR] Gemini generation failed: {err_msg} (kind={err_kind})")
                fallback = _build_basic_fallback_resume(
                    cv_text,
                    err_msg or "Gemini generation failed.",
                    model_used,
                )
                _RESUME_CACHE[cache_key] = fallback
                return fallback
            answer = _extract_text_from_response(response)
        else:
            print(f"[ERROR] Unsupported model_type: {model_type}")
            fallback = _build_basic_fallback_resume(
                cv_text,
                f"Unsupported model type: {model_type}",
                model,
            )
            _RESUME_CACHE[cache_key] = fallback
            return fallback
    except Exception as e:
        if _is_quota_or_rate_limit_error(e):
            print(f"[ERROR] Quota/rate-limit error while generating resume: {e}")
        else:
            print(f"[ERROR] Failed generating resume JSON: {e}")
        fallback = _build_basic_fallback_resume(
            cv_text,
            "Failed to generate resume. Please verify your API key and quota.",
            model,
        )
        _RESUME_CACHE[cache_key] = fallback
        return fallback

    parsed = safe_parse_json(answer)
    if not isinstance(parsed, dict):
        print("[ERROR] Parsed JSON is not an object. Returning empty resume.")
        fallback = _build_basic_fallback_resume(
            cv_text,
            "The model response was not valid JSON.",
            model_used if "model_used" in locals() else model,
        )
        _RESUME_CACHE[cache_key] = fallback
        return fallback

    # Keep backward compatibility when basics comes without wrapping key.
    if "basics" not in parsed and any(
        k in parsed for k in ["name", "email", "phone", "website", "address"]
    ):
        parsed = {"basics": parsed}

    # Ensure expected top-level sections always exist.
    parsed.setdefault("basics", {})
    parsed.setdefault("education", [])
    parsed.setdefault("awards", [])
    parsed.setdefault("projects", [])
    parsed.setdefault("skills", [])
    parsed.setdefault("work", [])
    parsed["_meta"] = {
        "fallback_used": fallback_used if "fallback_used" in locals() else False,
        "model_used": model_used if "model_used" in locals() else model,
        "error_kind": None,
        "error_message": None,
        "source": "llm",
    }

    _RESUME_CACHE[cache_key] = parsed
    return parsed


def tailor_resume(cv_text, api_key, model="gpt-4o", model_type="OpenAI"):
    filled_prompt = TAILORING_PROMPT.replace("<CV_TEXT>", cv_text)
    if model_type == "OpenAI":
        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_TAILORING},
                    {"role": "user", "content": filled_prompt},
                ],
            )

            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            print(e)
            print("Failed to tailor resume.")
            return cv_text
    elif model_type == "Gemini":
        if not api_key:
            print("[ERROR] Missing Gemini API key for tailoring.")
            return cv_text
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        try:
            full_prompt = f"{SYSTEM_TAILORING}\n\nUser: {filled_prompt}\nAssistant:"
            response = generate_with_retry(model_instance, full_prompt)
            if response is None:
                err_msg = _LAST_RETRY_ERROR_MESSAGE
                err_kind = _LAST_RETRY_ERROR_KIND
                print(f"[ERROR] Gemini tailoring failed after retries: {err_msg} ({err_kind})")
                return cv_text
            answer = _extract_text_from_response(response)
            return answer
        except Exception as e:
            print(f"[ERROR] Failed to tailor resume: {e}")
            return cv_text
