# PDD: Plant Disease Detection 

## Local setup

1. Install Python **>=3.6.5 and <3.7**
2. Create virtual environment:

```console
python -m venv .env
```

3. Activate environment
    - on windows: `.env\Scripts\activate`
    - on linux/mac: `source .env/bin/activate`

4. Install dependencies:

```console
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Inference

1. Change paths in the ***config/script_parameters.yaml***
2. Run `python script.py` in the root directory