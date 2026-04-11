# LakePrompt
LakePrompt research project for CS4964.

## Overview

LakePrompt is a prototype for answering natural-language questions over a folder of CSV files. It loads the tables, profiles columns, retrieves relevant columns semantically, finds join paths, executes queries, packages evidence for an LLM, and returns an answer with supporting evidence.

## User Interface

The main user-facing interface is `LakePrompt`. A user provides a directory of CSV files, asks a natural-language question, and receives a `LakeAnswer` containing answer text, supporting evidence rows, and cited evidence IDs.

### What the user provides

LakePrompt expects a directory containing one or more CSV files:

```text
data/
  customers.csv
  orders.csv
  products.csv
```

Each file becomes a table internally, using the filename without the `.csv` extension as the table name.

### Required setup

LakePrompt calls Anthropic during table summarization and final answer generation, so the environment must include:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

Create and activate a local virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need to refresh `requirements.txt` from the active virtual environment:

```bash
python -m pip freeze > requirements.txt
```

### Supported data sources

LakePrompt supports:

- a local directory of CSV files via `LakePrompt(...)`
- a supported remote source via `LakePrompt.from_url(...)`

`LakePrompt.from_url(...)` currently supports:

- direct `.csv` links
- direct `.zip` links containing CSV files
- GitHub repository URLs, where CSV files are discovered recursively after download

### Constructor interface

```python
from lakeprompt import LakePrompt

lp = LakePrompt(
    lake_dir="./data",
    model="claude-sonnet-4-20250514",
    cache_path="./table_summaries.json",
)
```

Constructor arguments:

- `lake_dir`: path to the directory containing CSV files
- `model`: Anthropic model name used for summarization and answer generation
- `cache_path`: optional JSON cache file for generated table summaries

### Remote source interface

```python
from lakeprompt import LakePrompt

lp = LakePrompt.from_url(
    "https://github.com/your-org/your-data-repo",
    cache_path="./table_summaries.json",
)
```

The remote source is downloaded and normalized into a local cached CSV lake before the normal LakePrompt pipeline runs.

### Query interface

Once initialized, the user interacts with the library through `query(...)`:

```python
from lakeprompt import LakePrompt

lp = LakePrompt("./data", cache_path="./table_summaries.json")
answer = lp.query("Which customers spent the most in January?")
```

This runs the full pipeline:

- load the CSV lake
- profile columns
- generate table summaries
- retrieve relevant columns
- infer join paths
- execute candidate joins
- package evidence for the prompt
- ask the LLM for the final response

### Return value

`query(...)` returns a `LakeAnswer` object:

```python
print(answer.text)
print(answer.cited_ids)
print(answer.evidence)
```

Its fields are:

- `answer.text`: the natural-language answer returned by the model
- `answer.cited_ids`: evidence IDs cited by the model response
- `answer.evidence`: a list of evidence rows used to support that answer

### Minimal end-to-end example

```python
from lakeprompt import LakePrompt

lp = LakePrompt("./data", cache_path="./table_summaries.json")
answer = lp.query("What are the top 3 products by total sales?")

print("Answer:")
print(answer.text)

print("\nCitations:")
print(answer.cited_ids)

print("\nEvidence:")
for row in answer.evidence[:3]:
    print(row.evidence_id, row.data)
```

### Typical usage flow

1. Place related CSV files in one directory.
2. Initialize `LakePrompt` with that directory.
3. Call `query(...)` with a natural-language question.
4. Read `answer.text` for the response, `answer.cited_ids` for cited evidence, and `answer.evidence` for supporting tuples.

## Acknowledgments

This project builds on several external datasets, specifications, and open-source libraries.

- Spider Join Data: evaluation planning in this repo uses the `superctj/spider-join-data` repository, which is derived from the broader Spider text-to-SQL benchmark ecosystem.
- Spider dataset: please credit the original Spider dataset authors when using Spider-derived data:
  Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. Proceedings of EMNLP 2018. DOI: `10.18653/v1/D18-1425`.
- TOON: prompt serialization work in this repo is based on Token-Oriented Object Notation (TOON), using the public `toon-format` specification and reference implementation ecosystem. Credit to the TOON project and Johann Schopplich.
- Open-source software: this project also relies on the maintainers and contributors behind `polars`, `numpy`, `hnswlib`, `sentence-transformers`, and the `anthropic` Python SDK.

References:

- Spider Join Data: https://github.com/superctj/spider-join-data
- Spider paper: https://aclanthology.org/D18-1425/
- Spider dataset repository: https://github.com/taoyds/spider
- TOON specification: https://github.com/toon-format/spec
- TOON reference implementation: https://github.com/toon-format/toon
