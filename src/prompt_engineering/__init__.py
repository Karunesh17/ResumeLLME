from email.mime import base
from openai import OpenAI
import google.generativeai as genai
import json
import os
from stqdm import stqdm

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


import time # Ensure you import time at the top of your file
import json
import os
# ... your other imports ...

def generate_json_resume(cv_text, api_key, model="gpt-4o", model_type="OpenAI"):
    """Generate a JSON resume from a CV text"""
    sections = []
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
        model_instance = genai.GenerativeModel(model) # renamed slightly to avoid shadowing the 'model' string variable

    for prompt in stqdm(
            [
                BASICS_PROMPT,
                EDUCATION_PROMPT,
                AWARDS_PROMPT,
                PROJECTS_PROMPT,
                SKILLS_PROMPT,
                WORK_PROMPT,
            ],
            desc="Pacing API requests to avoid rate limits...",
    ):
        filled_prompt = prompt.replace(CV_TEXT_PLACEHOLDER, cv_text)
        if model_type == "OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": filled_prompt},
                ],
            )
            answer = response.choices[0].message.content
        elif model_type == "Gemini":
            full_prompt = f"{SYSTEM_TAILORING}\n\nUser: {filled_prompt}\nAssistant:"
            response = model_instance.generate_content(full_prompt)
            answer = response.parts[0].text
            answer = answer.strip("'").replace("```json\n", "").replace("\n```", "")
        
        try:
            answer = json.loads(answer)

            if prompt == BASICS_PROMPT and "basics" not in answer:
                answer = {"basics": answer}  # common mistake GPT makes

            sections.append(answer)
        except Exception as e:
            print(f"Exception occurred. {e}")
            
        # --- RATE LIMIT PREVENTION DELAY ---
        # Pauses the loop for 12 seconds before making the next API call.
        # This spreads the 6 requests over >60 seconds.
        time.sleep(12)

    final_json = {}
    for section in sections:
        final_json.update(section)

    return final_json


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
        model = genai.GenerativeModel(model)
        try:
            full_prompt = f"{SYSTEM_TAILORING}\n\nUser: {filled_prompt}\nAssistant:"
            response = model.generate_content(full_prompt)
            print(type(response.parts[0].text), response.parts[0].text)
            answer = response.parts[0].text
            return answer
        except Exception as e:
            print(e)
            print("Failed to tailor resume.")
            return cv_text
