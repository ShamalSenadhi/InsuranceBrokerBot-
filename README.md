# Broker Advisor Dashboard
1. pip install -r requirements.txt
2. Fill insurers.yaml (29 insurers) and/or add data/manual/<Insurer>__<category>.txt
3. python ingest.py
4. export ANTHROPIC_API_KEY=...   (optional)
5. streamlit run app.py
