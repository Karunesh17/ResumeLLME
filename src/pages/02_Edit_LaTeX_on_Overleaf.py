import streamlit as st

st.set_page_config(page_title="Edit LaTeX on Overleaf", page_icon="📝")

st.title("📝 Edit Your Resume on Overleaf")

st.markdown("""
### 🚀 Continue Editing Your Resume Professionally

Once your resume is generated in LaTeX format, you can further refine and customize it using **Overleaf**, an online LaTeX editor.

👉 [Open Overleaf](https://www.overleaf.com)

---

### 📌 Step-by-Step Guide

1. Download the generated `.tex` file from this app  
2. Go to Overleaf and click **"New Project" → "Upload Project"**  
3. Upload your `.tex` file  
4. Open the file and click **"Recompile"**  
5. Edit content (text, sections, formatting) as needed  
6. Download the final PDF  

---

### ⚙️ Important Compiler Settings

Some templates require a specific compiler:

- 🟢 **XeLaTeX** → Recommended for modern fonts  
- 🔵 **pdfLaTeX** → Works for basic templates  

👉 To change:
Menu → Compiler → Select XeLaTeX → Recompile  

---

### ✨ Pro Tips

- Keep resume to **1 page (max 2)**  
- Use **action verbs** (Developed, Built, Designed)  
- Maintain consistent formatting  
- Avoid too many colors/fonts  
- Keep sections clean and ATS-friendly  

---

### ⚠️ Troubleshooting

- ❌ PDF not generating → Switch compiler to XeLaTeX  
- ❌ Font errors → Ensure font packages are included  
- ❌ Layout issues → Check margins and spacing  

---

### 💡 Why Overleaf?

- No setup required  
- Real-time preview  
- Professional LaTeX editing  
- Easy export to PDF  

---

You're now ready to create a **professional, ATS-friendly resume** 🚀
""")