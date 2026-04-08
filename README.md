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
- Open-source software: this project also relies on the maintainers and contributors behind `polars`, `numpy`, `hnswlib`, `sentence-transformers`, the `openai` Python SDK, and the `anthropic` Python SDK.

References:

- Spider Join Data: https://github.com/superctj/spider-join-data
- Spider paper: https://aclanthology.org/D18-1425/
- Spider dataset repository: https://github.com/taoyds/spider
- TOON specification: https://github.com/toon-format/spec
- TOON reference implementation: https://github.com/toon-format/toon
