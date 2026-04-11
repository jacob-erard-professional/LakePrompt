# LakePrompt
LakePrompt Research project for CS4964

## AI acknoledgment
Claude Opus 4.6 was used to enhance the docstrings, write tests,

### How each AI was utilized

Claude Opus 4.6
- Enhancing docstrings

Codex
- Refactoring repository to use polars only (original workflow used spark)
- Bringing to life our idea of evaluating claude's accuracy with and without lake prompt

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

## User Interface

The library currently exposes two main user-facing entry points:

- `DataLake` for loading a directory of CSV files and querying them directly with SQL.
- `LakePrompt` for asking natural-language questions over that same CSV lake.

### 1. Load a CSV lake

Put your CSV files in one directory:

```text
data/
  customers.csv
  orders.csv
  products.csv
```

Then load them with `DataLake`:

```python
from lakeprompt import DataLake

lake = DataLake.load("./data")
print(lake)
```

Each CSV is registered as a table using its filename without the `.csv` extension.

### 2. Query the lake directly with SQL

Use `DataLake.query(...)` when you want direct structured access:

```python
from lakeprompt import DataLake

lake = DataLake.load("./data")

result = lake.query("""
SELECT c.customer_id, c.customer_name, SUM(o.total) AS revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name
ORDER BY revenue DESC
LIMIT 5
""")

print(result)
```

This returns a Polars `DataFrame`.

You can also inspect a table sample:

```python
sample = lake.get_sample("customers", n=10)
print(sample)
```

### 3. Ask natural-language questions with `LakePrompt`

`LakePrompt` is the higher-level interface. It:

- loads the CSV lake,
- profiles columns,
- generates table summaries,
- builds retrieval indexes,
- finds relevant join paths,
- executes candidate joins,
- packages the evidence,
- and asks the LLM for a final answer.

Basic usage:

```python
from lakeprompt import LakePrompt

lp = LakePrompt("./data")
answer = lp.query("Which customers spent the most in January?")

print(answer.text)
print(answer.evidence)
```

`lp.query(...)` returns a `LakeAnswer` object with:

- `answer.text`: the model's natural-language answer
- `answer.evidence`: the joined tuples used as supporting evidence

### 4. Required setup for `LakePrompt`

`LakePrompt` uses Anthropic for table summarization and final answer generation, so you must set:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

You can also choose a model and optionally cache generated table summaries:

```python
from lakeprompt import LakePrompt

lp = LakePrompt(
    "./data",
    model="claude-sonnet-4-20250514",
    cache_path="./table_summaries.json",
)
```

### 5. Minimal end-to-end example

```python
from lakeprompt import LakePrompt

lp = LakePrompt("./data", cache_path="./table_summaries.json")
answer = lp.query("What are the top 3 products by total sales?")

print("Answer:")
print(answer.text)

print("\nEvidence rows:")
for row in answer.evidence[:3]:
    print(row)
```

In short:

- use `DataLake` if you want direct SQL access to the CSV tables
- use `LakePrompt` if you want question-answering over the lake with retrieved evidence
