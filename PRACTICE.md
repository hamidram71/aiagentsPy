# Python
__pycache__/
*.pyc
.pytest_cache/
.venv/

# Secrets: your API key lives here. Never commit it.
.env

# Everything the agent writes: run folders (with uploaded company data),
# reports, charts, cleaned data, checkpoints and the usage log.
outputs/*
!outputs/.gitkeep
