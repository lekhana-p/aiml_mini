# AI-Based Digital Wellbeing & Screen Addiction Detection System

A mini project for 3rd year engineering students.

## Project structure

```text
digital_wellbeing/
    app.py
    train_model.py
    dataset_generator.py
    usage_dataset.csv
    model.pkl
    database.db
    templates/
        index.html
        result.html
    static/
        style.css
```

## Features
- Input daily smartphone usage values.
- Predict addiction risk: Low / Medium / High.
- Store every prediction request in SQLite.
- Show basic analytics on dashboard.

## How to run in VS Code (macOS)

1. Open VS Code and open this folder.
2. Open terminal in VS Code.
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r digital_wellbeing/requirements.txt
   ```
5. Generate dataset:
   ```bash
   python digital_wellbeing/dataset_generator.py
   ```
6. Train and save model:
   ```bash
   python digital_wellbeing/train_model.py
   ```
7. Run Flask app:
   ```bash
   python digital_wellbeing/app.py
   ```
8. Open browser at:
   `http://127.0.0.1:5000`

## Notes
- If `database.db` is not present, it is created automatically.
- The dataset is synthetic and generated for learning purposes.
