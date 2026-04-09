from openai import OpenAI
import google.generativeai as genai
import json
import os
import time
import re

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
    Stops execution on daily quota exhaustion.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _throttle_api_calls()
            return model_instance.generate_content(prompt)
        except Exception as e:
            if _is_daily_quota_exceeded(e):
                print("DAILY QUOTA EXCEEDED")
                raise RuntimeError("DAILY QUOTA EXCEEDED") from e

            retry_delay = _extract_retry_delay_seconds(e)
            backoff_delay = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            sleep_seconds = retry_delay if retry_delay is not None else backoff_delay

            if "429" in str(e) or "resource_exhausted" in str(e).lower():
                print(
                    f"[WARN] Gemini rate limited (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {sleep_seconds}s."
                )
            else:
                print(
                    f"[WARN] Gemini call failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {sleep_seconds}s."
                )
            time.sleep(sleep_seconds)

    print("[ERROR] Max retries exceeded for Gemini request.")
    return None

def generate_json_resume(cv_text, api_key, model="gpt-4o", model_type="OpenAI"):
    """Generate a JSON resume from a CV text"""
    if model_type == "OpenAI":
        timeout = int(os.getenv("OPENAI_TIMEOUT", "60"))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        base_url = os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com")
        
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
    elif model_type == "Gemini":
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)

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
        elif model_type == "Gemini":
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {combined_prompt}\nAssistant:"
            response = generate_with_retry(model_instance, full_prompt)
            if response is None:
                print("[ERROR] Gemini returned no response after retries.")
                return {}
            answer = _extract_text_from_response(response)
        else:
            print(f"[ERROR] Unsupported model_type: {model_type}")
            return {}
    except Exception as e:
        if "DAILY QUOTA EXCEEDED" in str(e):
            # Bubble up to stop current execution path as requested.
            raise
        print(f"[ERROR] Failed generating resume JSON: {e}")
        return {}

    parsed = safe_parse_json(answer)
    if not isinstance(parsed, dict):
        print("[ERROR] Parsed JSON is not an object. Returning empty resume.")
        return {}

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
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        try:
            full_prompt = f"{SYSTEM_TAILORING}\n\nUser: {filled_prompt}\nAssistant:"
            response = generate_with_retry(model_instance, full_prompt)
            if response is None:
                print("[ERROR] Gemini tailoring failed after retries.")
                return cv_text
            answer = _extract_text_from_response(response)
            return answer
        except Exception as e:
            if "DAILY QUOTA EXCEEDED" in str(e):
                raise
            print(f"[ERROR] Failed to tailor resume: {e}")
            return cv_text
