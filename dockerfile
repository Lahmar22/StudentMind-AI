FROM python:3.12-slim AS streamlit

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -c "from pathlib import Path; p=Path('/tmp/requirements.txt'); data=p.read_bytes(); text=data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'); requirements=(line.split('==', 1)[0] for line in text.splitlines() if line.strip() and not line.lower().startswith('apache-airflow')); p.write_text('\n'.join(requirements) + '\n', encoding='utf-8')" \
    && pip install --no-cache-dir -r /tmp/requirements.txt streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM apache/airflow:3.4.0-python3.12 AS airflow

COPY --chown=airflow:root requirements.txt /tmp/requirements.txt

RUN python -c "from pathlib import Path; p=Path('/tmp/requirements.txt'); data=p.read_bytes(); text=data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'); requirements=(line.split('==', 1)[0] for line in text.splitlines() if line.strip()); Path('/tmp/requirements-normalized.txt').write_text('\n'.join(requirements) + '\n', encoding='utf-8')" \
    && pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt" \
    -r /tmp/requirements-normalized.txt