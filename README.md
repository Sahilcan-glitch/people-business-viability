# People Business Viability (Streamlit)

A Streamlit app that helps assess the **business viability of people audiences**.

Visit : https://people-business-viability.streamlit.app

You can:
- 📂 Upload a **CSV** of people profiles (first name, last name, email, country, blurb, etc.)
- 📧 Paste one or many **email addresses**
- ✅ The app cleans, validates, deduplicates
- 🤖 (Optional) Run **AI Enrichment** based on your **Company/Brand** and your **Goal/Question**
  - Example: “Would these people attend my workshop?”  
  - Example: “Is this audience a fit for product X?”

---

## ✨ Features
- Upload CSV → clean + dedup → enrich with AI
- Paste raw emails → enrich directly
- Adds `pipeline_source` column (`csv_upload` / `pasted_emails`)
- Download cleaned or enriched outputs as **CSV** / **JSONL**

---

## 🚀 Local Quickstart
```bash
# clone repo
git clone https://github.com/<your-username>/people-business-viability.git
cd people-business-viability

# setup env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run app

streamlit run apppeoplefinder.py# people-business-viability
Perfect for testing market fit, event interest, or audience viability.
