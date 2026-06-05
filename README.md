Mandatory Project Requirements 
●  RocketRide Usage: Use for core data/AI pipelines and workflows; 
pipelines must be meaningfully connected to logic. 
●  Butterbase Integration: Provision and serve the backend, integrating 
database, auth, and the AI Model Gateway. 
●  XTrace Integration: Use the Memory API to allow agents to actively write 
to and read from a persistent history. 
●  Photon Integration: Deliver the agent through at least one real 
messaging platform (e.g., iMessage, WhatsApp, Slack). 


Goal of the project : 

1) Take a folder, with a deep learning model. (model_input)
2) extract the functions of the model 
3) compare it with the documentation 
4) produce a report on what is compatible and what is not
5) Propose a Novel architecture with compatible functions 

Local pipeline without RocketRide:

```bash
uv run python main.py
```

Or choose a custom model folder/report path:

```bash
uv run python agents/local_pipeline.py model_input/Transfer-Model_original --report reports/model_input_vitis_ai_compatibility.md
```

Streamlit app:

```bash
uv run streamlit run streamlit_app.py
```

The app lets you upload a model source folder from the browser, runs the local
compatibility pipeline, previews the generated report, and exposes a Markdown
download.

Butterbase persistence:

```bash
export BUTTERBASE_APP_ID=app_your_id
export BUTTERBASE_API_KEY=bb_sk_your_key
export BUTTERBASE_API_URL=https://api.butterbase.ai
```

After setting those variables, open the Streamlit sidebar and click
`Apply schema`. This creates an `analysis_jobs` table for job status, model
folder path, operator counts, result JSON, and the generated Markdown report.
If the variables are not set, the app stays in local-only mode.

Photon-facing backend:

```bash
uv run uvicorn photon_backend:app --reload --port 8000
```

For local Photon Spectrum webhook testing, expose the backend with ngrok:

```bash
ngrok http 8000
```

Then register `https://<ngrok-host>/spectrum-webhook` in Photon Spectrum. Set
`SPECTRUM_WEBHOOK_SECRET` locally if you want signature verification enabled.
Photon webhook mode delivers inbound messages to this backend; outbound replies
should be handled with the Spectrum SDK loop or another messaging sender.
Photon-triggered analyses are also persisted to Butterbase when
`BUTTERBASE_APP_ID` and `BUTTERBASE_API_KEY` are set.
