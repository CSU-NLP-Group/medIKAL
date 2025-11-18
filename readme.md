# Code for "medIKAL: Integrating Knowledge Graphs as Assistants of LLMs for Enhanced Clinical Diagnosis on EMRs"

## Dataset

Due to copyright requirements and privacy restrictions, we only provide the URL-id of the source web page for each EMR. You can access each medical record by concatenating the URL-id with the [base URL of the source website](https://bingli.iiyi.com/).  
Please construct the full URL using the following format: `https://bingli.iiyi.com/show/{emrid}-1.html`.  
For example: [https://bingli.iiyi.com/show/65684-1.html](https://bingli.iiyi.com/show/65684-1.html)

## Env

Please make sure you have correctly installed Neo4j before running.

```sh
pip install -r requirements.txt
```

## Run

```sh
cd src/main
python main.py
```

## Updates
Stay tuned for updates on data and code!
