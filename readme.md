# Personal project leverage llm power

## Used
Create python env and install dependencies:
```bash
python -m venv venv
pip install -r requirements.txt
```

Create .env file from .env_copy file:
```bash
cp .env_copy .env
```
Get free Gemini API key [here](https://aistudio.google.com/app/prompts/new_chat) and push to the .env file.


### Simple QA
Copy your own data to document forder "data", then run:
```bash
python simple_qa.py
```

