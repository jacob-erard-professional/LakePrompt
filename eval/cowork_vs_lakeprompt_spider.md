# Cowork vs. LakePrompt — Spider Join Comparison

- Dataset source: <https://github.com/superctj/spider-join-data/raw/main/data.zip>
- Total questions: 20
- Semantic scoring: `eval/evaluation.py::_semantic_score` with `claude-sonnet-4-6`

## Setup

This document compares the LakePrompt prototype against an unrestricted-access baseline (a Cowork agent with direct file access) on a 20-question subset of the Spider Join benchmark spanning three schemas: `department_management`, `farm`, and `student_assessment`.

The 20 questions and ground-truth answers come from the LakePrompt evaluation harness (`eval/spider_join_questions_with_ground_truth.json`, exercised via `eval/evaluation.py`). LakePrompt's answers, latency, and per-call API token usage are extracted from the harness's existing report at `LakePrompt/eval/results_with_token_usage.md`, specifically the `lakeprompt_ranked` condition — the configuration that uses the full retrieval pipeline.

The Cowork answers come from a clean-room run: a fresh Cowork session was given the 20 questions tagged by schema and read access to `cowork_workspace/spider_data/` only. It was explicitly forbidden to read the eval folder, prior outputs, or any ground-truth file. Latency was wall-clocked around its SQL execution. Approx tokens is a proxy: `ceil(chars_of_csv_read / 4) + ceil(len(answer) / 4)`. The Cowork agent did not call any LLM at answering time, its 'tokens' column is therefore measuring *data read* in token-equivalent units, not API spend.

### What this comparison can and cannot tell us

*Can* tell us: where LakePrompt's retrieval pipeline produces correct vs. incorrect answers on this 20-question subset, and what kinds of mistakes recur. 

*Cannot* tell us: whether the unrestricted-access baseline would scale. The full Spider Join archive contains roughly fifty schemas; this subset has three, and each has at most a few dozen rows. An unconstrained agent with SQL can brute-force these in microseconds. A larger lake, or one where the relevant tables can't be located by folder name, would force any agent to use a retrieval mechanism, at which point the comparison becomes meaningful in a different way.

To make that asymmetry concrete: Cowork was given the three relevant schemas pre-staged at `cowork_workspace/spider_data/`, and each question in the prompt was tagged with its target schema. LakePrompt, by contrast, ran against the full archive with untagged questions. Cowork's setup isn't really a data lake in the LakePrompt sense, it's three known small tables. The 1.000 score reflects writing SQL against known tables, not retrieving the right ones from a larger lake.

The headline number (Cowork at semantic-score 1.000 on this subset) is best read as confirmation that the questions have obtainable answers given full data access, not as a verdict on either system in production.

## Aggregate (means)

- LakePrompt API input tokens — mean over 17/20: 473.6
- LakePrompt API output tokens — mean over 17/20: 64.2
- Cowork approx tokens — mean over 20/20: 195.7
- LakePrompt latency (s) — mean over 20/20: 3.982
- Cowork latency (s) — mean over 20/20: 0.000
- LakePrompt semantic score — mean over 20/20: 0.500
- Cowork semantic score — mean over 20/20: 1.000

## How the answers are scored

Both score columns come from the project's own `_semantic_score` function in `eval/evaluation.py`. It's an LLM-as-judge: for each row, the function sends a prompt containing the question, the ground-truth answer, and the prediction to a Claude model, then parses a single number back. The rubric is fixed:

- `1.0` — the prediction conveys the same answer as the ground truth, even if it's worded or formatted differently.
- `0.5` — partially correct: contains the right answer among others, or covers some but not all of the correct values.
- `0.0` — wrong, refuses to answer, or says the evidence is insufficient.

The judge call uses `temperature=0.0` and `max_tokens=5`. Identical answers can still receive different scores across runs or across judge models on borderline cases.

### Which judge model judged what

The original LakePrompt eval that produced `results_with_token_usage.md` ran with `claude-sonnet-4-20250514`. This document's rerun uses `claude-sonnet-4-6`. The rubric, prompt template, and `temperature=0.0` are unchanged.

We re-judged the LakePrompt column instead of reusing the existing `semantic_score` rows from `results_with_token_usage.md` so that both the LakePrompt and Cowork columns are evaluated by the same judge call in the same session. Mixing a freshly judged Cowork column with historically judged LakePrompt scores would let model changes potentially influence the comparison.

### Discrepancies from the original LakePrompt report

We compared the rerun's LakePrompt column against the `lakeprompt_ranked` semantic scores recorded in `results_with_token_usage.md`. 19 of 20 rows agree exactly. The single disagreement:

| Example | Original report | This rerun | What changed |
| --- | --- | --- | --- |
| #6 (farm) | `0.0` | `0.5` | Borderline answer. LakePrompt produced years paired with raw `host_city_id` integers instead of `official_name` strings. The original judge treated that as wrong; the rerun judge treated the years as half-credit. The answer text itself didn't change. |

We're keeping the rerun value (`0.5`) on the principle that whichever judge call answers for the LakePrompt column should also answer for the Cowork column. Aggregate impact: if we pinned #6 to the historical `0.0` instead, the LakePrompt mean drops from 0.500 to 0.475.

## Per-question comparison

| Question | Ground Truth | LakePrompt answer | Cowork answer | LakePrompt API input tokens | LakePrompt API output tokens | Cowork approx tokens | LakePrompt latency (s) | Cowork latency (s) | LakePrompt semantic score | Cowork semantic score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Means (across questions)** | — | — | — | 473.6 | 64.2 | 195.7 | 3.982 | 0.000 | 0.500 | 1.000 |
| #1 (department_management) — What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'? | 1903 | 1903 | 1903 | 415 | 18 | 292 | 6.471 | 0.0002 | 1 | 1 |
| #2 (department_management) — Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'? | [{"name": "Treasury", "num_employees": 115897.0}, {"name": "Homeland Security", "num_employees": 208000.0}, {"name": "Treasury", "num_employees": 115897.0}] | Treasury with 115,897 employees and Homeland Security with 208,000 employees are the departments managed by heads whose temporary acting value is 'Yes'. | Treasury, 115897.0; Homeland Security, 208000.0 | 388 | 53 | 185 | 3.780 | 0.0000 | 1 | 1 |
| #3 (department_management) — What are the distinct ages of the heads who are acting? | [53.0, 52.0, 69.0] | The distinct ages of the heads who are acting are 53, 52, and 69. | 52.0, 53.0, 69.0 | 374 | 42 | 108 | 3.144 | 0.0000 | 1 | 1 |
| #4 (department_management) — List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born. | California | Could not find evidence in the data lake | California | [unknown] | [unknown] | 260 | 2.736 | 0.0001 | 0 | 1 |
| #5 (department_management) — Which department has more than 1 head at a time? List the id, name and the number of heads. | [{"count(*)": 2, "department_id": 2, "name": "Treasury"}] | Treasury department (id: 2) has more than 1 head at a time, with 2 heads. | id=2, name=Treasury, heads=2 | 471 | 41 | 180 | 3.710 | 0.0004 | 1 | 1 |
| #6 (farm) — Show the years and the official names of the host cities of competitions. | [{"official_name": "Grand Falls/Grand-Sault", "year": 2013}, {"official_name": "Perth-Andover", "year": 2006}, {"official_name": "Plaster Rock", "year": 2005}, {"official_name": "Drummond", "year": 2004}, {"official_name": "Aroostook", "year": 2002}, {"official_name": "Aroostook", "year": 2003}] | 2006 (host_city_id: 2), 2005 (host_city_id: 3), 2004 (host_city_id: 4), 2013 (host_city_id: 1), 2002 (host_city_id: 5), 2003 (host_city_id: 5) | 2002, Aroostook; 2003, Aroostook; 2004, Drummond; 2005, Plaster Rock; 2006, Perth-Andover; 2013, Grand Falls/Grand-Sault | 374 | 114 | 249 | 3.953 | 0.0000 | 0.5 | 1 |
| #7 (farm) — Give the years and official names of the cities of each competition. | [{"official_name": "Grand Falls/Grand-Sault", "year": 2013}, {"official_name": "Perth-Andover", "year": 2006}, {"official_name": "Plaster Rock", "year": 2005}, {"official_name": "Drummond", "year": 2004}, {"official_name": "Aroostook", "year": 2002}, {"official_name": "Aroostook", "year": 2003}] | The competitions were held in: 2006 (city ID 2), 2005 (city ID 3), 2004 (city ID 4), 2002 (city ID 5), 2013 (city ID 1), and 2003 (city ID 5). However, the official names of the cities are not provided in the evidence - only city IDs are available. | 2002, Aroostook; 2003, Aroostook; 2004, Drummond; 2005, Plaster Rock; 2006, Perth-Andover; 2013, Grand Falls/Grand-Sault | 373 | 120 | 192 | 4.442 | 0.0000 | 0.5 | 1 |
| #8 (farm) — Show the official names of the cities that have hosted more than one competition. | Aroostook | The evidence shows host_city_id values (3, 2, 4, 5) for cities that have hosted more than one competition, but the official names of these cities are not provided in the evidence rows. | Aroostook | 357 | 71 | 165 | 6.734 | 0.0000 | 0 | 1 |
| #9 (farm) — What are the official names of cities that have hosted more than one competition? | Aroostook | The evidence provided only contains host_city_id numbers (1, 2, 3, 4, 5) but does not include the official names of the cities. Cannot determine which cities hosted more than one competition or their official names from the given evidence. | Aroostook | 345 | 83 | 165 | 4.991 | 0.0000 | 0 | 1 |
| #10 (farm) — Show the status of the city that has hosted the greatest number of competitions. | Village | The city with ID 5 has hosted the greatest number of competitions (2 competitions), but the status information is not available in the provided evidence. | Village | 406 | 47 | 164 | 3.520 | 0.0000 | 0 | 1 |
| #11 (farm) — What is the status of the city that has hosted the most competitions? | Village | The question asks for the status of the city that has hosted the most competitions, but the evidence row E1 only provides the host_city_id (5) and competition count (2). No status information is available in the provided evidence. | Village | 405 | 66 | 164 | 4.041 | 0.0000 | 0 | 1 |
| #12 (farm) — Please show the themes of competitions with host cities having populations larger than 1000. | ["Carnival M is back!", "Codehunters", "MTV Asia Aid"] | Could not find evidence in the data lake | Carnival M is back!, Codehunters, MTV Asia Aid | [unknown] | [unknown] | 174 | 2.079 | 0.0000 | 0 | 1 |
| #13 (farm) — What are the themes of competitions that have corresponding host cities with more than 1000 residents? | ["Carnival M is back!", "Codehunters", "MTV Asia Aid"] | Could not find evidence in the data lake | Carnival M is back!, Codehunters, MTV Asia Aid | [unknown] | [unknown] | 174 | 2.144 | 0.0000 | 0 | 1 |
| #14 (student_assessment) — which course has most number of registered students? | statistics | Statistics has the most number of registered students with 16 students. | statistics | 526 | 30 | 249 | 4.279 | 0.0000 | 1 | 1 |
| #15 (student_assessment) — What is the name of the course with the most registered students? | statistics | Statistics | statistics | 540 | 17 | 128 | 3.722 | 0.0000 | 1 | 1 |
| #16 (student_assessment) — what are the first name and last name of all candidates? | [{"first_name": "Shannon", "last_name": "Senger"}, {"first_name": "Virginie", "last_name": "Hartmann"}, {"first_name": "Dariana", "last_name": "Bednar"}, {"first_name": "Verna", "last_name": "Grant"}, {"first_name": "Hoyt", "last_name": "Wintheiser"}, {"first_name": "Mayra", "last_name": "Hartmann"}, {"first_name": "Lizeth", "last_name": "Bartoletti"}, {"first_name": "Nova", "last_name": "Feest"}] | The candidates' names are: Robert, Alex, Jane, Cindy, Leo, Jack, Robin, and Tao. However, the data only contains single names in the candidate_details field, so it's not possible to distinguish between first names and last names from the available evidence. | Shannon Senger; Virginie Hartmann; Dariana Bednar; Verna Grant; Hoyt Wintheiser; Mayra Hartmann; Lizeth Bartoletti; Nova Feest | 329 | 97 | 309 | 4.027 | 0.0000 | 0 | 1 |
| #17 (student_assessment) — What are the first and last names of all the candidates? | [{"first_name": "Shannon", "last_name": "Senger"}, {"first_name": "Virginie", "last_name": "Hartmann"}, {"first_name": "Dariana", "last_name": "Bednar"}, {"first_name": "Verna", "last_name": "Grant"}, {"first_name": "Hoyt", "last_name": "Wintheiser"}, {"first_name": "Mayra", "last_name": "Hartmann"}, {"first_name": "Lizeth", "last_name": "Bartoletti"}, {"first_name": "Nova", "last_name": "Feest"}] | The candidates' names are: Robert, Alex, Jane, Cindy, Jack, Leo, Tao, and Robin. | Shannon Senger; Virginie Hartmann; Dariana Bednar; Verna Grant; Hoyt Wintheiser; Mayra Hartmann; Lizeth Bartoletti; Nova Feest | 329 | 64 | 309 | 3.309 | 0.0000 | 0 | 1 |
| #18 (student_assessment) — What are the ids of all students for courses and what are the names of those courses? | [{"course_name": "statistics", "student_id": 111}, {"course_name": "statistics", "student_id": 121}, {"course_name": "French", "student_id": 121}, {"course_name": "French", "student_id": 131}, {"course_name": "English", "student_id": 141}, {"course_name": "data structure", "student_id": 151}, {"course_name": "English", "student_id": 161}, {"course_name": "statistics", "student_id": 171}, {"course_name": "statistics", "student_id": 141}] | The student IDs for courses are: 161 (English), 141 (English and statistics), 151 (data structure), 171 (statistics), 111 (statistics), 121 (statistics and French), 131 (French). | student_id=111, course=statistics; student_id=121, course=statistics; student_id=121, course=French; student_id=131, course=French; student_id=141, course=English; student_id=151, course=data structure; student_id=161, course=English; student_id=171, course=statistics; student_id=141, course=statistics | 543 | 95 | 201 | 3.855 | 0.0000 | 1 | 1 |
| #19 (student_assessment) — What is detail of the student who most recently registered course? | Martin | The student who most recently registered for a course is Martin (student ID: 121). | student_id=121, student_details=Martin, course_id=303, registration_date=2008-11-14 10:35:13 | 529 | 34 | 123 | 3.513 | 0.0000 | 1 | 1 |
| #20 (student_assessment) — What details do we have on the students who registered for courses most recently? | Martin | The students who registered most recently are Martin (student ID 121) who registered on 2008-11-14, followed by Nikhil (student ID 141) and Sarah (student ID 161) who both registered on 2008-11-08 and 2008-11-07 respectively. | student_id=121, student_details=Martin, course_id=303, registration_date=2008-11-14 10:35:13 | 1347 | 99 | 123 | 5.196 | 0.0000 | 1 | 1 |

## Findings

Two recurring patterns account for almost all of LakePrompt's incorrect answers on this subset. 

### Pattern 1: foreign-key IDs surface unresolved

On most of the misses, LakePrompt's pipeline retrieves rows from one table in a join but does not pull through the human-readable column from the joined table. The answer text contains the foreign-key integer where the question asked for the related row's name, status, or other string-valued column.

- **Q6 / Q7** (host city of competitions): LakePrompt's response includes `host_city_id: 2`, `host_city_id: 3`, etc., rather than `city.official_name` values like `Aroostook` and `Plaster Rock`. The join `farm_competition.host_city_id = city.city_id` was not pushed through to surface `official_name`.
- **Q8 / Q9** (cities that hosted multiple competitions): LakePrompt counts competitions per `host_city_id` correctly but does not translate the resulting IDs back to `city.official_name`. The response says "the official names of these cities are not provided in the evidence rows" even though `city.csv` is part of the schema.
- **Q10 / Q11** (status of the busiest host city): same shape — the city ID with the highest count is identified, but `city.status` is not pulled into the evidence.
- **Q16 / Q17** (candidate first/last names): LakePrompt reads `candidate_details` directly from the `candidates` table. That column happens to contain a single name per row (`Robert`, `Alex`, `Jane`, …), unrelated to the candidates' actual first/last names. The real names live in the `people` table, joinable via `candidate_id = person_id`. The retrieval surfaced the wrong table's local field instead of joining out.

Sketch of a fix: when the planner determines the answer requires a string-valued projection from a related table, the retrieval and prompt-packaging stages need to ensure the joined table's relevant column lands in the evidence rows passed to the LLM. The current `lakeprompt_ranked` evidence shows ID columns and counts but not the resolved string values.

### Pattern 2: empty retrieval

Q4, Q12, and Q13 all end with the response `Could not find evidence in the data lake`. The corresponding Scoring Metrics rows show `evidence_count: 0`, `sources[0]: <empty>`, `evidence_groups[0]: <empty>`, and `api_input_tokens: [unknown]` — the final API call was never made because no evidence was assembled.

- **Q4** ("states where both the secretary of Treasury and Homeland Security were born") needs a self-join on `head` via `management`, filtered by two department names.
- **Q12 / Q13** ("themes of competitions in cities with population > 1000") needs a numeric filter on `city.population` on top of the same `farm_competition`–`city` join LakePrompt did execute on Q6–Q11.

In both cases the join-path discovery or query planner gave up before producing evidence, suggesting the planner does not handle self-joins or post-join numeric filters reliably.

### What the unrestricted-access baseline tells us

Cowork's clean-room run scores 1.000 across all 20 questions. As discussed in the Setup section, this is best read as a sanity check that the questions have obtainable answers given full access to the CSVs, not as a head-to-head verdict against LakePrompt. The clean re-run answers differ in formatting from the earlier run (Q2 deduplicated, Q3 reordered, Q18–Q20 use a verbose row format) — direct evidence the agent did its own work.

The latency column also tells the same story in a different register: Cowork's mean of ~0.0001 s is SQLite resolving queries over a few dozen rows in microseconds this is not a measurement of retrieval intelligence. LakePrompt's mean of ~3.98 s is dominated by the network round-trip to Anthropic for the final answer-generation call. The two columns measure structurally different things and should not be summarized as a speedup figure.

## Per-question detail

### Example 1 — department_management
Question: What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?

Ground truth: 1903

LakePrompt: 1903

Cowork: 1903

Semantic scores — LakePrompt: 1  |  Cowork: 1

### Example 2 — department_management
Question: Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?

Ground truth: [{"name": "Treasury", "num_employees": 115897.0}, {"name": "Homeland Security", "num_employees": 208000.0}, {"name": "Treasury", "num_employees": 115897.0}]

LakePrompt: Treasury with 115,897 employees and Homeland Security with 208,000 employees are the departments managed by heads whose temporary acting value is 'Yes'.

Cowork: Treasury, 115897.0; Homeland Security, 208000.0

Semantic scores — LakePrompt: 1  |  Cowork: 1

Notes: Clean-room Cowork answer drops the duplicate Treasury entry that appears in the ground truth. The earlier (potentially contaminated) run reproduced the duplicate. Direct evidence the clean agent ran its own SQL.

### Example 3 — department_management
Question: What are the distinct ages of the heads who are acting?

Ground truth: [53.0, 52.0, 69.0]

LakePrompt: The distinct ages of the heads who are acting are 53, 52, and 69.

Cowork: 52.0, 53.0, 69.0

Semantic scores — LakePrompt: 1  |  Cowork: 1

Notes: Order of returned ages differs between runs. Both are correct sets; the difference is sort direction.

### Example 4 — department_management
Question: List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.

Ground truth: California

LakePrompt: Could not find evidence in the data lake

Cowork: California

Semantic scores — LakePrompt: 0  |  Cowork: 1

Notes: LakePrompt's ranked retrieval found no evidence and returned 'Could not find evidence in the data lake'; api token counts logged as `[unknown]`. See Findings → Pattern 2.

### Example 5 — department_management
Question: Which department has more than 1 head at a time? List the id, name and the number of heads.

Ground truth: [{"count(*)": 2, "department_id": 2, "name": "Treasury"}]

LakePrompt: Treasury department (id: 2) has more than 1 head at a time, with 2 heads.

Cowork: id=2, name=Treasury, heads=2

Semantic scores — LakePrompt: 1  |  Cowork: 1

### Example 6 — farm
Question: Show the years and the official names of the host cities of competitions.

Ground truth: [{"official_name": "Grand Falls/Grand-Sault", "year": 2013}, {"official_name": "Perth-Andover", "year": 2006}, {"official_name": "Plaster Rock", "year": 2005}, {"official_name": "Drummond", "year": 2004}, {"official_name": "Aroostook", "year": 2002}, {"official_name": "Aroostook", "year": 2003}]

LakePrompt: 2006 (host_city_id: 2), 2005 (host_city_id: 3), 2004 (host_city_id: 4), 2013 (host_city_id: 1), 2002 (host_city_id: 5), 2003 (host_city_id: 5)

Cowork: 2002, Aroostook; 2003, Aroostook; 2004, Drummond; 2005, Plaster Rock; 2006, Perth-Andover; 2013, Grand Falls/Grand-Sault

Semantic scores — LakePrompt: 0.5  |  Cowork: 1

Notes: This is the one row where our re-judge disagrees with the original LakePrompt report (which scored 0.0). LakePrompt's answer pairs years with `host_city_id` integers instead of `official_name` strings — a borderline case the rubric can defensibly score either way. See 'Discrepancies from the original LakePrompt report' near the top.

### Example 7 — farm
Question: Give the years and official names of the cities of each competition.

Ground truth: [{"official_name": "Grand Falls/Grand-Sault", "year": 2013}, {"official_name": "Perth-Andover", "year": 2006}, {"official_name": "Plaster Rock", "year": 2005}, {"official_name": "Drummond", "year": 2004}, {"official_name": "Aroostook", "year": 2002}, {"official_name": "Aroostook", "year": 2003}]

LakePrompt: The competitions were held in: 2006 (city ID 2), 2005 (city ID 3), 2004 (city ID 4), 2002 (city ID 5), 2013 (city ID 1), and 2003 (city ID 5). However, the official names of the cities are not provided in the evidence - only city IDs are available.

Cowork: 2002, Aroostook; 2003, Aroostook; 2004, Drummond; 2005, Plaster Rock; 2006, Perth-Andover; 2013, Grand Falls/Grand-Sault

Semantic scores — LakePrompt: 0.5  |  Cowork: 1

### Example 8 — farm
Question: Show the official names of the cities that have hosted more than one competition.

Ground truth: Aroostook

LakePrompt: The evidence shows host_city_id values (3, 2, 4, 5) for cities that have hosted more than one competition, but the official names of these cities are not provided in the evidence rows.

Cowork: Aroostook

Semantic scores — LakePrompt: 0  |  Cowork: 1

### Example 9 — farm
Question: What are the official names of cities that have hosted more than one competition?

Ground truth: Aroostook

LakePrompt: The evidence provided only contains host_city_id numbers (1, 2, 3, 4, 5) but does not include the official names of the cities. Cannot determine which cities hosted more than one competition or their official names from the given evidence.

Cowork: Aroostook

Semantic scores — LakePrompt: 0  |  Cowork: 1

### Example 10 — farm
Question: Show the status of the city that has hosted the greatest number of competitions.

Ground truth: Village

LakePrompt: The city with ID 5 has hosted the greatest number of competitions (2 competitions), but the status information is not available in the provided evidence.

Cowork: Village

Semantic scores — LakePrompt: 0  |  Cowork: 1

### Example 11 — farm
Question: What is the status of the city that has hosted the most competitions?

Ground truth: Village

LakePrompt: The question asks for the status of the city that has hosted the most competitions, but the evidence row E1 only provides the host_city_id (5) and competition count (2). No status information is available in the provided evidence.

Cowork: Village

Semantic scores — LakePrompt: 0  |  Cowork: 1

### Example 12 — farm
Question: Please show the themes of competitions with host cities having populations larger than 1000.

Ground truth: ["Carnival M is back!", "Codehunters", "MTV Asia Aid"]

LakePrompt: Could not find evidence in the data lake

Cowork: Carnival M is back!, Codehunters, MTV Asia Aid

Semantic scores — LakePrompt: 0  |  Cowork: 1

Notes: Same retrieval miss as #4 — empty sources[0]/evidence_groups[0]; api token counts `[unknown]`. See Findings → Pattern 2.

### Example 13 — farm
Question: What are the themes of competitions that have corresponding host cities with more than 1000 residents?

Ground truth: ["Carnival M is back!", "Codehunters", "MTV Asia Aid"]

LakePrompt: Could not find evidence in the data lake

Cowork: Carnival M is back!, Codehunters, MTV Asia Aid

Semantic scores — LakePrompt: 0  |  Cowork: 1

Notes: Same retrieval miss as #12.

### Example 14 — student_assessment
Question: which course has most number of registered students?

Ground truth: statistics

LakePrompt: Statistics has the most number of registered students with 16 students.

Cowork: statistics

Semantic scores — LakePrompt: 1  |  Cowork: 1

### Example 15 — student_assessment
Question: What is the name of the course with the most registered students?

Ground truth: statistics

LakePrompt: Statistics

Cowork: statistics

Semantic scores — LakePrompt: 1  |  Cowork: 1

### Example 16 — student_assessment
Question: what are the first name and last name of all candidates?

Ground truth: [{"first_name": "Shannon", "last_name": "Senger"}, {"first_name": "Virginie", "last_name": "Hartmann"}, {"first_name": "Dariana", "last_name": "Bednar"}, {"first_name": "Verna", "last_name": "Grant"}, {"first_name": "Hoyt", "last_name": "Wintheiser"}, {"first_name": "Mayra", "last_name": "Hartmann"}, {"first_name": "Lizeth", "last_name": "Bartoletti"}, {"first_name": "Nova", "last_name": "Feest"}]

LakePrompt: The candidates' names are: Robert, Alex, Jane, Cindy, Leo, Jack, Robin, and Tao. However, the data only contains single names in the candidate_details field, so it's not possible to distinguish between first names and last names from the available evidence.

Cowork: Shannon Senger; Virginie Hartmann; Dariana Bednar; Verna Grant; Hoyt Wintheiser; Mayra Hartmann; Lizeth Bartoletti; Nova Feest

Semantic scores — LakePrompt: 0  |  Cowork: 1

### Example 17 — student_assessment
Question: What are the first and last names of all the candidates?

Ground truth: [{"first_name": "Shannon", "last_name": "Senger"}, {"first_name": "Virginie", "last_name": "Hartmann"}, {"first_name": "Dariana", "last_name": "Bednar"}, {"first_name": "Verna", "last_name": "Grant"}, {"first_name": "Hoyt", "last_name": "Wintheiser"}, {"first_name": "Mayra", "last_name": "Hartmann"}, {"first_name": "Lizeth", "last_name": "Bartoletti"}, {"first_name": "Nova", "last_name": "Feest"}]

LakePrompt: The candidates' names are: Robert, Alex, Jane, Cindy, Jack, Leo, Tao, and Robin.

Cowork: Shannon Senger; Virginie Hartmann; Dariana Bednar; Verna Grant; Hoyt Wintheiser; Mayra Hartmann; Lizeth Bartoletti; Nova Feest

Semantic scores — LakePrompt: 0  |  Cowork: 1

### Example 18 — student_assessment
Question: What are the ids of all students for courses and what are the names of those courses?

Ground truth: [{"course_name": "statistics", "student_id": 111}, {"course_name": "statistics", "student_id": 121}, {"course_name": "French", "student_id": 121}, {"course_name": "French", "student_id": 131}, {"course_name": "English", "student_id": 141}, {"course_name": "data structure", "student_id": 151}, {"course_name": "English", "student_id": 161}, {"course_name": "statistics", "student_id": 171}, {"course_name": "statistics", "student_id": 141}]

LakePrompt: The student IDs for courses are: 161 (English), 141 (English and statistics), 151 (data structure), 171 (statistics), 111 (statistics), 121 (statistics and French), 131 (French).

Cowork: student_id=111, course=statistics; student_id=121, course=statistics; student_id=121, course=French; student_id=131, course=French; student_id=141, course=English; student_id=151, course=data structure; student_id=161, course=English; student_id=171, course=statistics; student_id=141, course=statistics

Semantic scores — LakePrompt: 1  |  Cowork: 1

Notes: Cowork's clean-room answer uses a more verbose `student_id=…, course=…` format vs. the earlier run's `id: course` pairs. Both contain the same nine pairings.

### Example 19 — student_assessment
Question: What is detail of the student who most recently registered course?

Ground truth: Martin

LakePrompt: The student who most recently registered for a course is Martin (student ID: 121).

Cowork: student_id=121, student_details=Martin, course_id=303, registration_date=2008-11-14 10:35:13

Semantic scores — LakePrompt: 1  |  Cowork: 1

Notes: Cowork's clean-room answer reports the full row (student_id, student_details, course_id, registration_date) rather than just the name 'Martin'. The judge accepts both.

### Example 20 — student_assessment
Question: What details do we have on the students who registered for courses most recently?

Ground truth: Martin

LakePrompt: The students who registered most recently are Martin (student ID 121) who registered on 2008-11-14, followed by Nikhil (student ID 141) and Sarah (student ID 161) who both registered on 2008-11-08 and 2008-11-07 respectively.

Cowork: student_id=121, student_details=Martin, course_id=303, registration_date=2008-11-14 10:35:13

Semantic scores — LakePrompt: 1  |  Cowork: 1

Notes: Same as #19 — clean-room answer is fuller; same row is correct.
