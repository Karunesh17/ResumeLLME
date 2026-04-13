import jinja2
import os

# This is a hack to import from doc_utils
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from doc_utils import escape_for_latex

template_commands = {
    "Simple": ["pdflatex", "-interaction=nonstopmode", "resume.tex"],
    "Awesome": ["xelatex", "-interaction=nonstopmode", "resume.tex"],
    "BGJC": ["pdflatex", "-interaction=nonstopmode", "resume.tex"],
    "Deedy": ["xelatex", "-interaction=nonstopmode", "resume.tex"],
    "Modern": ["pdflatex", "-interaction=nonstopmode", "resume.tex"],
    "Plush": ["xelatex", "-interaction=nonstopmode", "resume.tex"],
    "Alta": ["xelatex", "-interaction=nonstopmode", "resume.tex"],
}


URL_FIELD_KEYS = {"website", "url", "linkedin", "github"}


def sanitize_url_for_latex(url):
    if not isinstance(url, str) or not url:
        return url

    cleaned = url.strip()
    cleaned = cleaned.replace("{-}", "-")
    cleaned = cleaned.replace("\\-", "-")
    cleaned = cleaned.replace("\\textbackslash{}", "")
    cleaned = cleaned.replace("{[}", "[").replace("{]}", "]")
    cleaned = cleaned.replace("\\%", "%")

    if cleaned.startswith("mailto:") or cleaned.startswith("tel:"):
        return cleaned
    if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
        cleaned = "https://" + cleaned
    return cleaned


def _sanitize_resume_urls(data, parent_key=None):
    if isinstance(data, dict):
        return {
            key: _sanitize_resume_urls(value, key)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_sanitize_resume_urls(item, parent_key) for item in data]
    if isinstance(data, str) and parent_key in URL_FIELD_KEYS:
        return sanitize_url_for_latex(data)
    return data


def generate_latex(template_name, json_resume, prelim_section_ordering):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    latex_jinja_env = jinja2.Environment(
        block_start_string="\BLOCK{",
        block_end_string="}",
        variable_start_string="\VAR{",
        variable_end_string="}",
        comment_start_string="\#{",
        comment_end_string="}",
        line_statement_prefix="%-",
        line_comment_prefix="%#",
        trim_blocks=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(dir_path),
    )

    escaped_json_resume = escape_for_latex(json_resume)
    escaped_json_resume = _sanitize_resume_urls(escaped_json_resume)

    return use_template(
        template_name, latex_jinja_env, escaped_json_resume, prelim_section_ordering
    )


def use_template(template_name, jinja_env, json_resume, prelim_section_ordering):
    PREFIX = f"{template_name}"
    EXTENSION = "tex.jinja"

    resume_template = jinja_env.get_template(f"{PREFIX}/resume.{EXTENSION}")
    basics_template = jinja_env.get_template(f"{PREFIX}/basics.{EXTENSION}")
    education_template = jinja_env.get_template(f"{PREFIX}/education.{EXTENSION}")
    work_template = jinja_env.get_template(f"{PREFIX}/work.{EXTENSION}")
    skills_template = jinja_env.get_template(f"{PREFIX}/skills.{EXTENSION}")
    projects_template = jinja_env.get_template(f"{PREFIX}/projects.{EXTENSION}")
    awards_template = jinja_env.get_template(f"{PREFIX}/awards.{EXTENSION}")

    sections = {}
    section_ordering = get_final_section_ordering(prelim_section_ordering)

    if "basics" in json_resume:
        firstName = json_resume["basics"]["name"].split(" ")[0]
        lastName = " ".join(json_resume["basics"]["name"].split(" ")[1:])
        sections["basics"] = basics_template.render(
            firstName=firstName, lastName=lastName, **json_resume["basics"]
        )
    if "education" in json_resume and len(json_resume["education"]) > 0:
        sections["education"] = education_template.render(
            schools=json_resume["education"], heading="Education"
        )
    if "work" in json_resume and len(json_resume["work"]) > 0:
        sections["work"] = work_template.render(
            works=json_resume["work"], heading="Work Experience"
        )

    if "skills" in json_resume and len(json_resume["skills"]) > 0:
        sections["skills"] = skills_template.render(
            skills=json_resume["skills"], heading="Skills"
        )
    if "projects" in json_resume and len(json_resume["projects"]) > 0:
        sections["projects"] = projects_template.render(
            projects=json_resume["projects"], heading="Projects"
        )

    if "awards" in json_resume and len(json_resume["awards"]) > 0:
        sections["awards"] = awards_template.render(
            awards=json_resume["awards"], heading="Awards"
        )

    resume = resume_template.render(
        sections=sections, section_ordering=section_ordering
    )
    return resume


def get_final_section_ordering(section_ordering):
    final_ordering = ["basics"]
    additional_ordering = section_ordering + [
        "education",
        "work",
        "skills",
        "projects",
        "awards",
    ]
    for section in additional_ordering:
        if section not in final_ordering:
            final_ordering.append(section)

    return final_ordering
