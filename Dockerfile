FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies (no hud-python needed — mocked in run_eval_agentic.py)
RUN pip install --no-cache-dir openai python-dotenv

# Copy application code
COPY env.py build_scenarios.py build_repo_mapping.py run_eval_agentic.py ./

# Data and results are volume-mounted at runtime:
#   /app/data/scenarios.json
#   /app/data/repo_mapping.json
#   /app/data/tasks_eval.json
#   /app/data/repos/          (17GB of cloned repos)
#   /app/results/             (output directory)

ENTRYPOINT ["python", "run_eval_agentic.py"]
