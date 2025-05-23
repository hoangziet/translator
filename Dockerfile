FROM python:3.12.9 
LABEL authors="vz"

WORKDIR /src 

COPY app/ /src/app/
COPY main.py /src/main.py
COPY requirements.txt /src/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

