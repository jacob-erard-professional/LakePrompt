# LakePrompt Evaluation Report

## Run Metadata
- Generated at: `2026-04-27T05:13:13.544784+00:00`
- Dataset root: `data`
- Metadata CSV: `.`
- Claude model: `claude-sonnet-4-20250514`
## Metric Glossary

- `exact_match_rate`: Fraction of examples where the answer text exactly matches the ground truth after lowercasing and trimming.
- `mean_ground_truth_coverage`: Average fraction of ground-truth tokens/atoms found in the answer.
- `mean_faithfulness`: Average fraction of valid evidence IDs cited or mentioned in the answer.
- `mean_semantic_score`: Average Claude-judged semantic correctness (0, 0.5, or 1).
- `mean_latency_seconds`: Average wall-clock runtime for that baseline per example.
- `mean_context_tokens`: Average whitespace-token count of the evidence/context portion only.
- `mean_prompt_tokens`: Average whitespace-token count of the full prompt sent to Claude.
- `mean_api_input_tokens`: Average actual Anthropic API input tokens used for the baseline's answer call.
- `mean_api_output_tokens`: Average actual Anthropic API output tokens used for the baseline's answer call.
- `mean_join_count`: Average number of distinct join paths used by the evidence for that baseline.
- `error_rate`: Fraction of examples where that baseline recorded an execution or API error.
- `exact_match`: Whether one example's answer exactly matched the ground truth after lowercasing and trimming.
- `ground_truth_coverage`: Fraction of the ground truth contained in the answer for one example.
- `faithfulness`: Fraction of valid evidence IDs cited or explicitly mentioned for one example.
- `semantic_score`: Claude-judged semantic correctness for one example (0, 0.5, or 1).
- `latency_seconds`: Runtime for one example.
- `evidence_count`: Number of evidence rows returned for one example.
- `join_count`: Number of distinct join paths represented in those evidence rows.
- `context_tokens`: Approximate whitespace-token count of the evidence/context only for one example.
- `prompt_tokens`: Approximate whitespace-token count of the full prompt for one example.
- `api_input_tokens`: Actual Anthropic API input tokens used for that answer call when available.
- `api_output_tokens`: Actual Anthropic API output tokens used for that answer call when available.
- `cited_ids`: Evidence IDs cited by the answer when present.
- `error`: Execution or API error captured for that example, if any.

## Aggregate Metrics

### no_context

| Metric | Value |
| --- | --- |
| `n` | `20` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.0026` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.0` |
| `mean_latency_seconds` | `3.3558` |
| `mean_context_tokens` | `0.0` |
| `mean_prompt_tokens` | `22.9` |
| `mean_api_input_tokens` | `49.0` |
| `mean_api_output_tokens` | `126.6` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### single_table

| Metric | Value |
| --- | --- |
| `n` | `20` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.0351` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.0` |
| `mean_latency_seconds` | `2.9493` |
| `mean_context_tokens` | `51.4` |
| `mean_prompt_tokens` | `80.2` |
| `mean_api_input_tokens` | `270.8` |
| `mean_api_output_tokens` | `89.8` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### naive_multitable

| Metric | Value |
| --- | --- |
| `n` | `20` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.547` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.525` |
| `mean_latency_seconds` | `3.9909` |
| `mean_context_tokens` | `191.8` |
| `mean_prompt_tokens` | `223.7` |
| `mean_api_input_tokens` | `931.0` |
| `mean_api_output_tokens` | `189.6` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### schema_baseline

| Metric | Value |
| --- | --- |
| `n` | `20` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.4577` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.4` |
| `mean_latency_seconds` | `5.2313` |
| `mean_context_tokens` | `234.4` |
| `mean_prompt_tokens` | `272.4` |
| `mean_api_input_tokens` | `1078.0` |
| `mean_api_output_tokens` | `203.3` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### join_no_ranking

| Metric | Value |
| --- | --- |
| `n` | `20` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.4687` |
| `mean_faithfulness` | `0.3125` |
| `mean_semantic_score` | `0.15` |
| `mean_latency_seconds` | `6.8807` |
| `mean_context_tokens` | `20.9` |
| `mean_prompt_tokens` | `55.2` |
| `mean_api_input_tokens` | `168.9` |
| `mean_api_output_tokens` | `103.3` |
| `mean_join_count` | `0.9` |
| `error_rate` | `0.0` |

### lakeprompt_ranked

| Metric | Value |
| --- | --- |
| `n` | `20` |
| `exact_match_rate` | `0.1` |
| `mean_ground_truth_coverage` | `0.4062` |
| `mean_faithfulness` | `0.7775` |
| `mean_semantic_score` | `0.475` |
| `mean_latency_seconds` | `3.9823` |
| `mean_context_tokens` | `20.9` |
| `mean_prompt_tokens` | `55.2` |
| `mean_api_input_tokens` | `473.6` |
| `mean_api_output_tokens` | `64.2` |
| `mean_join_count` | `0.9` |
| `error_rate` | `0.0` |

## Examples

## Example 1

- Schema: `department_management`
- Question: What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?
- Ground Truth: 1903

**Expected SQL Query**

```sql
SELECT DISTINCT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id  =  T2.department_id JOIN head AS T3 ON T2.head_id  =  T3.head_id WHERE T3.born_state  =  'Alabama'
```

**LakePrompt Generated Query**

```sql
SELECT DISTINCT("department"."creation") AS "agg_0__distinct__department__creation" FROM "department" JOIN "management" ON "department"."department_id" = "management"."department_id" JOIN "head" ON "management"."head_id" = "head"."head_id" WHERE "head"."born_state" = 'Alabama'
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?
```

**Response**

```text
I don't have access to any database or dataset that contains information about departments, secretaries, their birth states, or department creation years. To answer this question, I would need:

1. A database or dataset with department information including creation years
2. Information about department managers/secretaries including their birth states
3. The ability to query or filter this data to find secretaries born in Alabama and the departments they manage

Without access to this specific data, I cannot provide the distinct creation years you're looking for. If you have access to such a database, you would typically need to write a query that:
- Joins department and secretary/manager tables
- Filters for secretaries born in Alabama
- Selects the distinct creation years of departments they manage

Could you provide the relevant dataset or clarify what data source you'd like me to work with?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.832` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `26` |
| `api_input_tokens` | `54` |
| `api_output_tokens` | `183` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Question:
What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown contains information about departments (budget, creation year, name, employees, etc.), but it does not include any information about secretaries or their birth states. To answer the question about departments managed by secretaries born in Alabama, I would need additional data that links departments to their secretaries and includes information about where those secretaries were born.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `7.756` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `38` |
| `prompt_tokens` | `70` |
| `api_input_tokens` | `224` |
| `api_output_tokens` | `85` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?
```

**Response**

```text
Looking at the provided data, I need to find departments managed by heads born in Alabama and then identify their creation years.

From the head table, I can see that heads with IDs 1 and 3 were born in Alabama:
- Head ID 1: Tiger Woods, born in Alabama
- Head ID 3: K. J. Choi, born in Alabama

However, when I check the management table to see which departments these Alabama-born heads manage, I find:
- The management table only contains entries for head IDs 4, 5, and 6
- Head IDs 1 and 3 (the Alabama-born heads) do not appear in the management table

Since there are no management records showing that any Alabama-born heads manage any departments in the provided data, I cannot determine any creation years for departments managed by secretaries born in Alabama.

The answer is that there are no departments shown to be managed by heads born in Alabama based on the provided rows.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.958` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `88` |
| `prompt_tokens` | `123` |
| `api_input_tokens` | `420` |
| `api_output_tokens` | `216` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: department
Columns: department_id, name, creation, ranking, budget_in_billions, num_employees
Sample rows:
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
Columns: head_id, name, born_state, age
Sample rows:
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
Columns: department_id, head_id, temporary_acting
Sample rows:
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?
```

**Response**

```text
I cannot answer this question confidently from the information provided.

While I can see that there are heads born in Alabama (Tiger Woods and K. J. Choi in the sample data), I cannot determine which departments they manage because:

1. The sample rows from the `management` table don't include any entries for head_id 1 or 3 (the Alabama-born heads from the sample data)
2. The sample data may not be complete - there could be other heads born in Alabama not shown in the samples
3. I would need to join all three tables (head → management → department) to find the creation years of departments managed by Alabama-born heads, but the sample data doesn't provide sufficient overlap to make this connection

To answer this question, I would need more complete data showing the management relationships for heads born in Alabama.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `5.084` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `110` |
| `prompt_tokens` | `151` |
| `api_input_tokens` | `494` |
| `api_output_tokens` | `182` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"distinct_creation": 1903}

Question:
What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence (E1) only contains a single distinct creation year of 1903, but it lacks the necessary information to determine:
1. Which departments are managed by secretaries
2. Which secretaries were born in Alabama
3. The connection between the creation year and departments managed by Alabama-born secretaries

To properly answer this question, I would need evidence showing the relationship between secretaries' birth states, the departments they manage, and those departments' creation years.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `9.318` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `3` |
| `prompt_tokens` | `40` |
| `api_input_tokens` | `92` |
| `api_output_tokens` | `113` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[3]: department,management,head
      join_keys[2]:
        [0]: ["department","department_id","management","department_id"]
        [1]: ["management","head_id","head","head_id"]
      path_score: 0.7600000000000001
      sql: "SELECT DISTINCT(\"department\".\"creation\") AS \"agg_0__distinct__department__creation\" FROM \"department\" JOIN \"management\" ON \"department\".\"department_id\" = \"management\".\"department_id\" JOIN \"head\" ON \"management\".\"head_id\" = \"head\".\"head_id\" WHERE \"head\".\"born_state\" = 'Alabama'"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,distinct_creation}:
        E1,1903
  schemas[1]:
    [0]:
      id: C1
      field_map:
        distinct_creation: agg_0__distinct__department__creation
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"1903","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
1903
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `True` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `6.471` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `3` |
| `prompt_tokens` | `40` |
| `api_input_tokens` | `415` |
| `api_output_tokens` | `18` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 2

- Schema: `department_management`
- Question: Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
- Ground Truth: [{"name": "Treasury", "num_employees": 115897.0}, {"name": "Homeland Security", "num_employees": 208000.0}, {"name": "Treasury", "num_employees": 115897.0}]

**Expected SQL Query**

```sql
SELECT T1.name ,  T1.num_employees FROM department AS T1 JOIN management AS T2 ON T1.department_id  =  T2.department_id WHERE T2.temporary_acting  =  'Yes'
```

**LakePrompt Generated Query**

```sql
SELECT "department"."name" AS "name", "department"."num_employees" AS "num_employees" FROM "management" JOIN "department" ON "management"."department_id" = "department"."department_id" WHERE "management"."temporary_acting" = 'Yes'
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
```

**Response**

```text
I'd be happy to help you with this query, but I don't have access to the specific database or dataset you're referring to. To answer your question about departments managed by heads with temporary acting value 'Yes', I would need:

1. **Database schema information** - What tables contain this data and how they're structured
2. **Table names** - Which tables contain department, employee, and manager information
3. **Column names** - The specific field names for:
   - Department names
   - Employee counts or employee records
   - Manager/head information
   - The "temporary acting" field

**General approach for this type of query:**
If you have tables like `departments`, `employees`, and `managers` (or similar), the SQL query would typically look something like:

```sql
SELECT d.department_name, COUNT(e.employee_id) as number_of_employees
FROM departments d
JOIN managers m ON d.manager_id = m.manager_id
LEFT JOIN employees e ON d.department_id = e.department_id
WHERE m.temporary_acting = 'Yes'
GROUP BY d.department_name, d.department_id;
```

Could you please provide:
- The database schema or table structure
- Sample data or table names
- The specific field names used in your database

This will allow me to give you an accurate, specific answer for your dataset.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.495` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `28` |
| `api_input_tokens` | `56` |
| `api_output_tokens` | `310` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Question:
Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown only contains information about departments (name, budget, creation year, department_id, number of employees, and ranking), but it does not include any information about department heads or their temporary acting status. To answer this question, I would need additional data that includes information about department heads and whether they have a temporary acting value of 'Yes'.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.086` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `38` |
| `prompt_tokens` | `72` |
| `api_input_tokens` | `226` |
| `api_output_tokens` | `86` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
```

**Response**

```text
Looking at the provided data, I need to find departments managed by heads with temporary_acting = 'Yes' and show their names and employee counts.

From the management table, I can see that departments with IDs 2 and 15 are managed by heads with temporary_acting = 'Yes':
- Department ID 2 (managed by head_id 5 and head_id 6)
- Department ID 15 (managed by head_id 4)

However, when I look at the department table, I only have information for departments with IDs 1, 2, and 3. Department ID 15 is not present in the provided department data.

For department ID 2 (which does exist in the department table):
- Name: Treasury
- Number of employees: 115,897

**Answer:** Based on the available data, only one department can be identified:
- Treasury: 115,897 employees

Note: Department ID 15 is referenced in the management table as having temporary acting management, but no corresponding department information is provided in the department table rows.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.286` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `6.097` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `88` |
| `prompt_tokens` | `125` |
| `api_input_tokens` | `422` |
| `api_output_tokens` | `242` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: department
Columns: department_id, name, creation, ranking, budget_in_billions, num_employees
Sample rows:
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
Columns: head_id, name, born_state, age
Sample rows:
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
Columns: department_id, head_id, temporary_acting
Sample rows:
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
```

**Response**

```text
Based on the schema and sample data provided, I can construct the query logic, but I cannot provide a complete answer because the sample data is insufficient.

To answer this question, I would need to:
1. Join the `management` table with the `department` table on `department_id`
2. Filter for records where `temporary_acting = 'Yes'`
3. Select the department `name` and `num_employees`

From the sample data, I can see that departments with IDs 2 and 15 are managed by heads with temporary_acting = 'Yes'. However:

- Department ID 2 (Treasury) has the required information: name = "Treasury", num_employees = "115897.0"
- Department ID 15 is referenced in the management table but has no corresponding row in the department table samples provided

Therefore, I cannot answer this question confidently with the given sample data alone, as it appears incomplete - specifically missing department information for department_id = 15 and potentially other departments referenced in the management table.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.429` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `5.504` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `110` |
| `prompt_tokens` | `153` |
| `api_input_tokens` | `496` |
| `api_output_tokens` | `231` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E2: {"name": "Homeland Security", "num_employees": 208000.0}
E1: {"name": "Treasury", "num_employees": 115897.0}

Question:
Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence only contains department names and employee counts for "Homeland Security" and "Treasury" departments. However, the question asks for departments managed by heads whose temporary acting value is 'Yes', which requires information about department heads and their acting status. This information is not present in the given evidence rows.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.429` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `5.942` |
| `evidence_count` | `2` |
| `join_count` | `1` |
| `context_tokens` | `11` |
| `prompt_tokens` | `50` |
| `api_input_tokens` | `125` |
| `api_output_tokens` | `77` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Show the name and number of employees for the departments managed by heads whose temporary acting value is 'Yes'?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[2]: management,department
      join_keys[1]:
        [0]: ["management","department_id","department","department_id"]
      path_score: 0.81
      sql: "SELECT \"department\".\"name\" AS \"name\", \"department\".\"num_employees\" AS \"num_employees\" FROM \"management\" JOIN \"department\" ON \"management\".\"department_id\" = \"department\".\"department_id\" WHERE \"management\".\"temporary_acting\" = 'Yes'"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[2]{id,name,num_employees}:
        E1,Treasury,115897.0
        E2,"Homeland Security",208000.0
  schemas[1]:
    [0]:
      id: C1
      field_map:
        name: name
        num_employees: num_employees
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"Treasury with 115,897 employees and Homeland Security with 208,000 employees are the departments managed by heads whose temporary acting value is 'Yes'.","cited_ids":["E1","E2"]}
```

**Parsed Answer Used For Scoring**

```text
Treasury with 115,897 employees and Homeland Security with 208,000 employees are the departments managed by heads whose temporary acting value is 'Yes'.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.429` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.78` |
| `evidence_count` | `2` |
| `join_count` | `1` |
| `context_tokens` | `11` |
| `prompt_tokens` | `50` |
| `api_input_tokens` | `388` |
| `api_output_tokens` | `53` |
| `cited_ids` | `E1, E2` |
| `error` | `[none]` |

## Example 3

- Schema: `department_management`
- Question: What are the distinct ages of the heads who are acting?
- Ground Truth: [53.0, 52.0, 69.0]

**Expected SQL Query**

```sql
SELECT DISTINCT T1.age FROM management AS T2 JOIN head AS T1 ON T1.head_id  =  T2.head_id WHERE T2.temporary_acting  =  'Yes'
```

**LakePrompt Generated Query**

```sql
SELECT DISTINCT("head"."age") AS "agg_0__distinct__head__age" FROM "head" JOIN "management" ON "head"."head_id" = "management"."head_id" WHERE "management"."temporary_acting" = 'Yes'
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What are the distinct ages of the heads who are acting?
```

**Response**

```text
I don't have access to any data about heads who are acting or their ages. To answer this question, I would need:

1. A dataset or database containing information about people in head/leadership positions
2. Information about which of these heads are currently acting (presumably in interim or temporary roles)
3. Age data for those individuals

Could you please provide the relevant data or clarify what specific dataset you'd like me to analyze?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.798` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `20` |
| `api_input_tokens` | `45` |
| `api_output_tokens` | `94` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Question:
What are the distinct ages of the heads who are acting?
```

**Response**

```text
I cannot answer this question using the provided rows. The table contains information about departments (name, budget, creation year, employees, etc.) but does not include any information about heads of departments, their ages, or whether they are acting heads. To answer your question, I would need data that includes information about department heads and their status.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.426` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `38` |
| `prompt_tokens` | `64` |
| `api_input_tokens` | `215` |
| `api_output_tokens` | `71` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
What are the distinct ages of the heads who are acting?
```

**Response**

```text
Looking at the provided data, I need to find the ages of heads who are acting (temporary_acting = "Yes").

From the management table, the heads who are acting have head_ids: 5, 4, and 6.

However, when I check the head table, I only see data for head_ids 1, 2, and 3 (Tiger Woods, Sergio García, and K. J. Choi respectively).

The head table does not contain information for head_ids 4, 5, and 6, which are the ones marked as acting in the management table.

Therefore, I cannot determine the ages of the acting heads based on the provided rows, as the necessary head records are not included in the data.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.839` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `88` |
| `prompt_tokens` | `117` |
| `api_input_tokens` | `411` |
| `api_output_tokens` | `164` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: department
Columns: department_id, name, creation, ranking, budget_in_billions, num_employees
Sample rows:
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
Columns: head_id, name, born_state, age
Sample rows:
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
Columns: department_id, head_id, temporary_acting
Sample rows:
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
What are the distinct ages of the heads who are acting?
```

**Response**

```text
Looking at the schema and sample data, I need to join the `head` and `management` tables to find heads who are acting (where `temporary_acting = "Yes"`), then get their distinct ages.

From the sample data provided:
- The `management` table shows heads with `temporary_acting = "Yes"` have `head_id` values of 4, 5, and 6
- However, the `head` table sample only shows `head_id` values 1, 2, and 3 with ages 67, 68, and 69 respectively

Since there's no overlap between the `head_id` values in the sample rows from these two tables, I cannot determine the ages of the acting heads from the provided sample data alone.

I cannot answer this question confidently from the information provided, as the sample data doesn't include the head records for the acting department heads shown in the management table.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.936` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `110` |
| `prompt_tokens` | `145` |
| `api_input_tokens` | `485` |
| `api_output_tokens` | `204` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E2: {"distinct_age": 52.0}
E1: {"distinct_age": 53.0}
E3: {"distinct_age": 69.0}

Question:
What are the distinct ages of the heads who are acting?
```

**Response**

```text
The evidence is insufficient to answer the question about the distinct ages of heads who are acting.

The provided evidence rows only contain distinct age values (52.0, 53.0, and 69.0), but there is no information indicating:
1. Whether these individuals are "heads" 
2. Whether they are "acting"
3. Any context connecting the ages to the specific roles mentioned in the question

To properly answer this question, I would need evidence that explicitly identifies which ages correspond to heads who are in acting positions.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.78` |
| `evidence_count` | `3` |
| `join_count` | `1` |
| `context_tokens` | `9` |
| `prompt_tokens` | `40` |
| `api_input_tokens` | `112` |
| `api_output_tokens` | `115` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What are the distinct ages of the heads who are acting?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[2]: head,management
      join_keys[1]:
        [0]: ["head","head_id","management","head_id"]
      path_score: 0.81
      sql: "SELECT DISTINCT(\"head\".\"age\") AS \"agg_0__distinct__head__age\" FROM \"head\" JOIN \"management\" ON \"head\".\"head_id\" = \"management\".\"head_id\" WHERE \"management\".\"temporary_acting\" = 'Yes'"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[3]{id,distinct_age}:
        E1,53.0
        E2,52.0
        E3,69.0
  schemas[1]:
    [0]:
      id: C1
      field_map:
        distinct_age: agg_0__distinct__head__age
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The distinct ages of the heads who are acting are 53, 52, and 69.","cited_ids":["E1","E2","E3"]}
```

**Parsed Answer Used For Scoring**

```text
The distinct ages of the heads who are acting are 53, 52, and 69.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.144` |
| `evidence_count` | `3` |
| `join_count` | `1` |
| `context_tokens` | `9` |
| `prompt_tokens` | `40` |
| `api_input_tokens` | `374` |
| `api_output_tokens` | `42` |
| `cited_ids` | `E1, E2, E3` |
| `error` | `[none]` |

## Example 4

- Schema: `department_management`
- Question: List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.
- Ground Truth: California

**Expected SQL Query**

```sql
SELECT T3.born_state FROM department AS T1 JOIN management AS T2 ON T1.department_id  =  T2.department_id JOIN head AS T3 ON T2.head_id  =  T3.head_id WHERE T1.name  =  'Treasury' INTERSECT SELECT T3.born_state FROM department AS T1 JOIN management AS T2 ON T1.department_id  =  T2.department_id JOIN head AS T3 ON T2.head_id  =  T3.head_id WHERE T1.name  =  'Homeland Security'
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.
```

**Response**

```text
I don't have access to a current database of information about where the current secretaries of the Treasury and Homeland Security departments were born. To answer this question accurately, I would need:

1. The names of the current Secretary of the Treasury and Secretary of Homeland Security
2. Their birth states/locations

This information changes with different presidential administrations, and without access to current data, I cannot provide a reliable answer about which states both current secretaries were born in.

If you could provide the names of the specific secretaries you're asking about, or if you have access to this data, I'd be happy to help analyze it further.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.42` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `27` |
| `api_input_tokens` | `59` |
| `api_output_tokens` | `137` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Question:
List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.
```

**Response**

```text
I cannot answer this question using only the provided rows. The table contains information about departments (name, budget, creation date, employees, etc.), but it does not include any information about:

1. Secretaries of the departments
2. Where the secretaries were born
3. The Homeland Security department (which is not present in the provided rows)

To answer this question, I would need additional data that includes information about department secretaries and their birth locations.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.639` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `38` |
| `prompt_tokens` | `71` |
| `api_input_tokens` | `229` |
| `api_output_tokens` | `100` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.
```

**Response**

```text
I cannot answer this question with the provided data because:

1. The management table shows department_id "2" (Treasury) is managed by head_ids "5" and "6", but the head table only contains information for head_ids "1", "2", and "3".

2. There is no "Homeland Security" department in the provided department table rows, and no corresponding management records for such a department.

3. Without the complete head information for Treasury's managers and without any data about Homeland Security department or its managers, I cannot determine the birth states of either department's secretaries.

To answer this question, I would need:
- Complete head table data including head_ids "5" and "6"
- Department table data for Homeland Security
- Management table data showing who manages Homeland Security
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.726` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `88` |
| `prompt_tokens` | `124` |
| `api_input_tokens` | `425` |
| `api_output_tokens` | `178` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: department
Columns: department_id, name, creation, ranking, budget_in_billions, num_employees
Sample rows:
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
Columns: head_id, name, born_state, age
Sample rows:
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
Columns: department_id, head_id, temporary_acting
Sample rows:
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.
```

**Response**

```text
I cannot answer this question confidently from the information provided.

Here's why:

1. **Missing department**: While I can see the Treasury department (department_id = 2) in the sample data, there is no "Homeland Security" department in the provided sample rows from the department table.

2. **Incomplete management data**: The management table sample only shows department_id 2 (Treasury) and 15, but doesn't include complete information about who heads which department.

3. **Missing head information**: The sample head data shows people with names like "Tiger Woods" and "Sergio García" which appear to be golfer names rather than actual department secretaries, suggesting this might be sample/test data rather than real government data.

4. **Incomplete linkage**: Even for Treasury, I would need to join the management and head tables to find who leads Treasury and where they were born, but the sample data doesn't provide a complete picture of these relationships.

To answer this question properly, I would need:
- Complete department data including Homeland Security
- Complete management relationships showing current (non-temporary) heads
- Complete head information for the relevant secretaries
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `9.837` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `110` |
| `prompt_tokens` | `152` |
| `api_input_tokens` | `499` |
| `api_output_tokens` | `252` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
[no evidence returned]

Question:
List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born.
```

**Response**

```text
The evidence is insufficient to answer this question. No evidence rows were provided to determine the birth states of the Secretary of Treasury or the Secretary of Homeland Security.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.4` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `41` |
| `api_input_tokens` | `90` |
| `api_output_tokens` | `35` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born."
  sources[0]:
  evidence_groups[0]:
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
Could not find evidence in the data lake
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.736` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `41` |
| `api_input_tokens` | `[unknown]` |
| `api_output_tokens` | `[unknown]` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

## Example 5

- Schema: `department_management`
- Question: Which department has more than 1 head at a time? List the id, name and the number of heads.
- Ground Truth: [{"count(*)": 2, "department_id": 2, "name": "Treasury"}]

**Expected SQL Query**

```sql
SELECT T1.department_id ,  T1.name ,  count(*) FROM management AS T2 JOIN department AS T1 ON T1.department_id  =  T2.department_id GROUP BY T1.department_id HAVING count(*)  >  1
```

**LakePrompt Generated Query**

```sql
SELECT "department"."department_id" AS "department__department_id", "department"."name" AS "name", COUNT("management"."head_id") AS "agg_2__count__management__head_id" FROM "management" JOIN "department" ON "management"."department_id" = "department"."department_id" GROUP BY "department"."department_id", "department"."name"
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Which department has more than 1 head at a time? List the id, name and the number of heads.
```

**Response**

```text
I don't have access to any data about departments, heads, or organizational structures. To answer your question about which department has more than 1 head at a time, along with their IDs, names, and number of heads, I would need:

1. A dataset or database containing department information
2. Information about department heads and their assignments
3. Timestamps or date ranges to determine what constitutes "at a time"

Could you please provide the relevant data or specify which dataset you'd like me to analyze?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.712` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `28` |
| `api_input_tokens` | `57` |
| `api_output_tokens` | `112` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Question:
Which department has more than 1 head at a time? List the id, name and the number of heads.
```

**Response**

```text
Based on the provided rows from the department table, I cannot answer this question. The table contains information about budget, creation year, department ID, name, number of employees, and ranking, but there is no column or data about the number of heads each department has. To answer which department has more than 1 head at a time, I would need additional data that includes information about department leadership structure or number of heads.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.561` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `38` |
| `prompt_tokens` | `72` |
| `api_input_tokens` | `227` |
| `api_output_tokens` | `89` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: department
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
Which department has more than 1 head at a time? List the id, name and the number of heads.
```

**Response**

```text
Looking at the provided data, I need to check which departments have multiple heads by examining the management table.

From the management table, I can see:
- Department ID "2" appears twice (with head_id "5" and head_id "6")
- Department ID "15" appears once (with head_id "4")

So department ID "2" has 2 heads at a time.

From the department table, department ID "2" is the "Treasury" department.

**Answer:**
- Department ID: 2
- Department Name: Treasury  
- Number of heads: 2
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.059` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `88` |
| `prompt_tokens` | `125` |
| `api_input_tokens` | `423` |
| `api_output_tokens` | `135` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: department
Columns: department_id, name, creation, ranking, budget_in_billions, num_employees
Sample rows:
{"budget_in_billions": "9.96", "creation": "1789", "department_id": "1", "name": "State", "num_employees": "30266.0", "ranking": "1"}
{"budget_in_billions": "11.1", "creation": "1789", "department_id": "2", "name": "Treasury", "num_employees": "115897.0", "ranking": "2"}
{"budget_in_billions": "439.3", "creation": "1947", "department_id": "3", "name": "Defense", "num_employees": "3000000.0", "ranking": "3"}

Table: head
Columns: head_id, name, born_state, age
Sample rows:
{"age": "67.0", "born_state": "Alabama", "head_id": "1", "name": "Tiger Woods"}
{"age": "68.0", "born_state": "California", "head_id": "2", "name": "Sergio Garc\u00eda"}
{"age": "69.0", "born_state": "Alabama", "head_id": "3", "name": "K. J. Choi"}

Table: management
Columns: department_id, head_id, temporary_acting
Sample rows:
{"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
{"department_id": "15", "head_id": "4", "temporary_acting": "Yes"}
{"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

Question:
Which department has more than 1 head at a time? List the id, name and the number of heads.
```

**Response**

```text
Looking at the schema and sample data, I can see that the `management` table links departments to their heads, and multiple entries for the same department_id would indicate multiple heads.

From the sample data in the `management` table, I can see that department_id "2" appears twice:
- {"department_id": "2", "head_id": "5", "temporary_acting": "Yes"}
- {"department_id": "2", "head_id": "6", "temporary_acting": "Yes"}

From the `department` table sample, department_id "2" corresponds to the "Treasury" department.

Based on this sample data, the answer would be:

**Department ID: 2**
**Name: Treasury** 
**Number of heads: 2**

However, I should note that this analysis is based only on the sample rows provided. To give a complete and confident answer about all departments that have more than 1 head, I would need to see the full dataset rather than just these sample rows, as there may be other departments with multiple heads that aren't represented in the samples shown.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `5.333` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `110` |
| `prompt_tokens` | `153` |
| `api_input_tokens` | `497` |
| `api_output_tokens` | `246` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E3: {"count_head_id": 1, "department_id": 7, "name": "Commerce"}
E2: {"count_head_id": 1, "department_id": 11, "name": "Transportation"}
E1: {"count_head_id": 2, "department_id": 2, "name": "Treasury"}
E4: {"count_head_id": 1, "department_id": 15, "name": "Homeland Security"}

Question:
Which department has more than 1 head at a time? List the id, name and the number of heads.
```

**Response**

```text
Based on the provided evidence, the department with more than 1 head at a time is:

**Department ID: 2, Name: Treasury, Number of heads: 2**

This is determined from evidence row E1, which shows the Treasury department (department_id: 2) has a count_head_id of 2, indicating it has 2 heads at a time.

All other departments in the evidence (Commerce, Transportation, and Homeland Security) have a count_head_id of 1, meaning they each have only 1 head at a time.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.250` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `7.232` |
| `evidence_count` | `4` |
| `join_count` | `1` |
| `context_tokens` | `29` |
| `prompt_tokens` | `68` |
| `api_input_tokens` | `196` |
| `api_output_tokens` | `128` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Which department has more than 1 head at a time? List the id, name and the number of heads."
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[2]: management,department
      join_keys[1]:
        [0]: ["management","department_id","department","department_id"]
      path_score: 0.81
      sql: "SELECT \"department\".\"department_id\" AS \"department__department_id\", \"department\".\"name\" AS \"name\", COUNT(\"management\".\"head_id\") AS \"agg_2__count__management__head_id\" FROM \"management\" JOIN \"department\" ON \"management\".\"department_id\" = \"department\".\"department_id\" GROUP BY \"department\".\"department_id\", \"department\".\"name\""
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[4]{id,count_head_id,department_id,name}:
        E1,2,2,Treasury
        E2,1,11,Transportation
        E3,1,7,Commerce
        E4,1,15,"Homeland Security"
  schemas[1]:
    [0]:
      id: C1
      field_map:
        count_head_id: agg_2__count__management__head_id
        department_id: department__department_id
        name: name
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"Treasury department (id: 2) has more than 1 head at a time, with 2 heads.","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
Treasury department (id: 2) has more than 1 head at a time, with 2 heads.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.250` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.71` |
| `evidence_count` | `4` |
| `join_count` | `1` |
| `context_tokens` | `29` |
| `prompt_tokens` | `68` |
| `api_input_tokens` | `471` |
| `api_output_tokens` | `41` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 6

- Schema: `farm`
- Question: Show the years and the official names of the host cities of competitions.
- Ground Truth: [{"official_name": "Grand Falls/Grand-Sault", "year": 2013}, {"official_name": "Perth-Andover", "year": 2006}, {"official_name": "Plaster Rock", "year": 2005}, {"official_name": "Drummond", "year": 2004}, {"official_name": "Aroostook", "year": 2002}, {"official_name": "Aroostook", "year": 2003}]

**Expected SQL Query**

```sql
SELECT T2.Year ,  T1.Official_Name FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID
```

**LakePrompt Generated Query**

```sql
SELECT "farm_competition"."year" AS "year", "farm_competition"."host_city_id" AS "host_city_id" FROM "farm_competition"
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Show the years and the official names of the host cities of competitions.
```

**Response**

```text
I don't have access to any specific dataset or table containing information about competitions, their years, and host cities. To answer your question about the years and official names of host cities for competitions, I would need:

1. Access to a database or dataset with competition information
2. Clarification about which specific competitions you're interested in (Olympics, World Cup, etc.)
3. The relevant data tables or files to query

Could you please provide the dataset or specify which competitions you'd like information about? Once I have access to the actual data, I'll be happy to show you the years and official names of the host cities.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.178` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `22` |
| `api_input_tokens` | `47` |
| `api_output_tokens` | `134` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
Show the years and the official names of the host cities of competitions.
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown contains information about cities (including their area, census ranking, population, and status), but it does not contain any information about:

1. Years of competitions
2. Which cities hosted competitions
3. Any competition-related data

To answer the question about years and official names of host cities of competitions, I would need additional data that links these cities to specific competitions and their corresponding years.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.326` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `74` |
| `api_input_tokens` | `252` |
| `api_output_tokens` | `97` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Show the years and the official names of the host cities of competitions.
```

**Response**

```text
Based on the provided data, I can show the years and official names of host cities for competitions by combining information from the farm_competition and city tables:

**Competition Years and Host Cities:**

- **2013**: Grand Falls/Grand-Sault (Theme: "Carnival M is back!")
- **2006**: Perth-Andover (Theme: "Codehunters") 
- **2005**: Plaster Rock (Theme: "MTV Asia Aid")

This information comes from matching the host_city_id in the farm_competition table with the city_id in the city table to get the official city names.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.600` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `2.871` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `207` |
| `api_input_tokens` | `792` |
| `api_output_tokens` | `140` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Show the years and the official names of the host cities of competitions.
```

**Response**

```text
Based on the schema and sample data provided, I can answer this question by joining the `farm_competition` and `city` tables.

Here's the information for the competitions shown in the sample data:

| Year | Official Name of Host City |
|------|---------------------------|
| 2013 | Grand Falls/Grand-Sault |
| 2006 | Perth-Andover |
| 2005 | Plaster Rock |

This result comes from joining the `farm_competition` table (which contains the year and host_city_id) with the `city` table (which contains the official_name for each city_id) on the host_city_id = city_id relationship.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.600` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `3.127` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `249` |
| `api_input_tokens` | `913` |
| `api_output_tokens` | `159` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E5: {"host_city_id": 5, "year": 2002}
E3: {"host_city_id": 4, "year": 2004}
E2: {"host_city_id": 3, "year": 2005}
E4: {"host_city_id": 1, "year": 2013}
E1: {"host_city_id": 2, "year": 2006}
E6: {"host_city_id": 5, "year": 2003}

Question:
Show the years and the official names of the host cities of competitions.
```

**Response**

```text
The evidence is insufficient to answer this question completely.

The provided evidence rows contain years and host_city_id values, but they do not include the official names of the host cities. The evidence shows:

- Host city ID 1: 2013
- Host city ID 2: 2006
- Host city ID 3: 2005
- Host city ID 4: 2004
- Host city ID 5: 2002, 2003

However, without additional evidence that maps the host_city_id values to their corresponding official city names, I cannot provide the official names of the host cities as requested in the question.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.400` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `7.032` |
| `evidence_count` | `6` |
| `join_count` | `1` |
| `context_tokens` | `30` |
| `prompt_tokens` | `63` |
| `api_input_tokens` | `198` |
| `api_output_tokens` | `142` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Show the years and the official names of the host cities of competitions."
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: farm_competition
      join_keys[0]:
      path_score: 0.9083333333333333
      sql: "SELECT \"farm_competition\".\"year\" AS \"year\", \"farm_competition\".\"host_city_id\" AS \"host_city_id\" FROM \"farm_competition\""
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[6]{id,year,host_city_id}:
        E1,2006,2
        E2,2005,3
        E3,2004,4
        E4,2013,1
        E5,2002,5
        E6,2003,5
  schemas[1]:
    [0]:
      id: C1
      field_map:
        year: year
        host_city_id: host_city_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"2006 (host_city_id: 2), 2005 (host_city_id: 3), 2004 (host_city_id: 4), 2013 (host_city_id: 1), 2002 (host_city_id: 5), 2003 (host_city_id: 5)","cited_ids":["E1","E2","E3","E4","E5","E6"]}
```

**Parsed Answer Used For Scoring**

```text
2006 (host_city_id: 2), 2005 (host_city_id: 3), 2004 (host_city_id: 4), 2013 (host_city_id: 1), 2002 (host_city_id: 5), 2003 (host_city_id: 5)
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.400` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.953` |
| `evidence_count` | `6` |
| `join_count` | `1` |
| `context_tokens` | `30` |
| `prompt_tokens` | `63` |
| `api_input_tokens` | `374` |
| `api_output_tokens` | `114` |
| `cited_ids` | `E1, E2, E3, E4, E5, E6` |
| `error` | `[none]` |

## Example 7

- Schema: `farm`
- Question: Give the years and official names of the cities of each competition.
- Ground Truth: [{"official_name": "Grand Falls/Grand-Sault", "year": 2013}, {"official_name": "Perth-Andover", "year": 2006}, {"official_name": "Plaster Rock", "year": 2005}, {"official_name": "Drummond", "year": 2004}, {"official_name": "Aroostook", "year": 2002}, {"official_name": "Aroostook", "year": 2003}]

**Expected SQL Query**

```sql
SELECT T2.Year ,  T1.Official_Name FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID
```

**LakePrompt Generated Query**

```sql
SELECT "farm_competition"."year" AS "year", "farm_competition"."host_city_id" AS "host_city_id" FROM "farm_competition"
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Give the years and official names of the cities of each competition.
```

**Response**

```text
I don't see any data about competitions or cities provided in your question. To answer your question about the years and official names of cities for each competition, I would need you to specify:

1. Which competitions you're referring to (e.g., Olympics, World Cup, Commonwealth Games, etc.)
2. Or provide the relevant data about these competitions

Could you please clarify which competitions you'd like information about, or share the data you'd like me to analyze?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.0` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `21` |
| `api_input_tokens` | `46` |
| `api_output_tokens` | `101` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
Give the years and official names of the cities of each competition.
```

**Response**

```text
I cannot answer this question using only the provided rows. The table contains information about cities including their official names, population, area, census ranking, and status, but there is no information about:

1. Years
2. Competitions

The data provided only shows details about three municipalities (Grand Falls/Grand-Sault, Perth-Andover, and Plaster Rock) but contains no temporal information or competition-related data that would be needed to answer your question.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.400` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.682` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `73` |
| `api_input_tokens` | `251` |
| `api_output_tokens` | `102` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Give the years and official names of the cities of each competition.
```

**Response**

```text
Based on the provided data, here are the years and official names of the cities for each competition:

**Competition 1 (2013):**
- Year: 2013
- Host City: Grand Falls/Grand-Sault

**Competition 2 (2006):**
- Year: 2006
- Host City: Perth-Andover

**Competition 3 (2005):**
- Year: 2005
- Host City: Plaster Rock

I was able to match each competition to its host city by linking the competition_id from the farm_competition table to the host_city_id, then finding the corresponding official_name in the city table.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.600` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `2.669` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `206` |
| `api_input_tokens` | `791` |
| `api_output_tokens` | `153` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Give the years and official names of the cities of each competition.
```

**Response**

```text
Based on the schema and sample data provided, I can answer this question by joining the `farm_competition` and `city` tables.

Here are the years and official names of the cities for each competition:

**Competition 1 (2013):** Grand Falls/Grand-Sault
**Competition 2 (2006):** Perth-Andover  
**Competition 3 (2005):** Plaster Rock

The information comes from joining the `farm_competition` table (which contains the competition year and host_city_id) with the `city` table (which contains the official_name for each city_id).
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.600` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `2.721` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `248` |
| `api_input_tokens` | `912` |
| `api_output_tokens` | `142` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E4: {"host_city_id": 5, "year": 2002}
E3: {"host_city_id": 4, "year": 2004}
E6: {"host_city_id": 5, "year": 2003}
E2: {"host_city_id": 3, "year": 2005}
E5: {"host_city_id": 1, "year": 2013}
E1: {"host_city_id": 2, "year": 2006}

Question:
Give the years and official names of the cities of each competition.
```

**Response**

```text
The evidence is insufficient to answer this question. 

While the provided evidence rows show the host_city_id and year for each competition, they do not contain the official names of the cities. The evidence only provides:

- City ID 1 hosted in 2013
- City ID 2 hosted in 2006  
- City ID 3 hosted in 2005
- City ID 4 hosted in 2004
- City ID 5 hosted in 2002 and 2003

To answer the question about the official names of the cities, I would need additional evidence that maps the city IDs to their actual city names.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.400` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `7.338` |
| `evidence_count` | `6` |
| `join_count` | `1` |
| `context_tokens` | `30` |
| `prompt_tokens` | `62` |
| `api_input_tokens` | `197` |
| `api_output_tokens` | `144` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Give the years and official names of the cities of each competition."
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: farm_competition
      join_keys[0]:
      path_score: 0.9083333333333333
      sql: "SELECT \"farm_competition\".\"year\" AS \"year\", \"farm_competition\".\"host_city_id\" AS \"host_city_id\" FROM \"farm_competition\""
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[6]{id,year,host_city_id}:
        E1,2006,2
        E2,2005,3
        E3,2004,4
        E4,2002,5
        E5,2013,1
        E6,2003,5
  schemas[1]:
    [0]:
      id: C1
      field_map:
        year: year
        host_city_id: host_city_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The competitions were held in: 2006 (city ID 2), 2005 (city ID 3), 2004 (city ID 4), 2002 (city ID 5), 2013 (city ID 1), and 2003 (city ID 5). However, the official names of the cities are not provided in the evidence - only city IDs are available.","cited_ids":["E1","E2","E3","E4","E5","E6"]}
```

**Parsed Answer Used For Scoring**

```text
The competitions were held in: 2006 (city ID 2), 2005 (city ID 3), 2004 (city ID 4), 2002 (city ID 5), 2013 (city ID 1), and 2003 (city ID 5). However, the official names of the cities are not provided in the evidence - only city IDs are available.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.400` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `4.442` |
| `evidence_count` | `6` |
| `join_count` | `1` |
| `context_tokens` | `30` |
| `prompt_tokens` | `62` |
| `api_input_tokens` | `373` |
| `api_output_tokens` | `120` |
| `cited_ids` | `E1, E2, E3, E4, E5, E6` |
| `error` | `[none]` |

## Example 8

- Schema: `farm`
- Question: Show the official names of the cities that have hosted more than one competition.
- Ground Truth: Aroostook

**Expected SQL Query**

```sql
SELECT T1.Official_Name FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID GROUP BY T2.Host_city_ID HAVING COUNT(*)  >  1
```

**LakePrompt Generated Query**

```sql
SELECT "farm_competition"."host_city_id" AS "host_city_id" FROM "farm_competition" GROUP BY "farm_competition"."host_city_id" HAVING "farm_competition"."host_city_id" > 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Show the official names of the cities that have hosted more than one competition.
```

**Response**

```text
I don't have access to any data about cities, competitions, or their hosting history. To answer your question about which cities have hosted more than one competition, I would need:

1. A dataset containing information about competitions and their host cities
2. The official names of those cities
3. The ability to count how many competitions each city has hosted

Could you please provide the relevant data or specify which particular type of competitions you're asking about (e.g., Olympics, World Cup, etc.)? With that information, I'd be happy to help identify cities that have hosted multiple competitions.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.767` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `23` |
| `api_input_tokens` | `48` |
| `api_output_tokens` | `125` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
Show the official names of the cities that have hosted more than one competition.
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown contains information about cities including their area, census ranking, population, and status, but there is no information about competitions or which cities have hosted competitions. To answer your question, I would need data that includes competition hosting information for these cities.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.275` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `75` |
| `api_input_tokens` | `253` |
| `api_output_tokens` | `65` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Show the official names of the cities that have hosted more than one competition.
```

**Response**

```text
Looking at the provided data, I need to examine the farm_competition table to see which cities have hosted competitions and count how many each city has hosted.

From the farm_competition table:
- Competition 1: hosted by city_id "1" (Grand Falls/Grand-Sault)
- Competition 2: hosted by city_id "2" (Perth-Andover) 
- Competition 3: hosted by city_id "3" (Plaster Rock)

Each city (city_id 1, 2, and 3) has hosted exactly one competition. No city has hosted more than one competition based on the provided data.

Therefore, there are no cities that have hosted more than one competition.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.212` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `208` |
| `api_input_tokens` | `793` |
| `api_output_tokens` | `160` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Show the official names of the cities that have hosted more than one competition.
```

**Response**

```text
I cannot answer this question confidently from the provided information alone.

While I can see the schema and sample data, the sample rows only show 3 competitions (competition_id 1, 2, and 3), each hosted by a different city (host_city_id 1, 2, and 3 respectively). To determine which cities have hosted more than one competition, I would need to see more complete data from the farm_competition table that might include additional competitions where the same host_city_id appears multiple times.

The sample data provided doesn't contain any cities that have hosted more than one competition, but this could simply be due to the limited sample size rather than reflecting the complete dataset.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.954` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `250` |
| `api_input_tokens` | `914` |
| `api_output_tokens` | `152` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E3: {"host_city_id": 4}
E2: {"host_city_id": 2}
E4: {"host_city_id": 5}
E1: {"host_city_id": 3}

Question:
Show the official names of the cities that have hosted more than one competition.
```

**Response**

```text
The evidence is insufficient to answer this question.

The provided evidence only shows host_city_id values (2, 3, 4, and 5) but does not include:
1. The official names of the cities
2. Information about how many competitions each city has hosted
3. Any way to determine which cities have hosted more than one competition

To answer this question, I would need evidence that includes city names and competition counts or frequency data for each city.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `9.833` |
| `evidence_count` | `4` |
| `join_count` | `1` |
| `context_tokens` | `12` |
| `prompt_tokens` | `46` |
| `api_input_tokens` | `129` |
| `api_output_tokens` | `102` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Show the official names of the cities that have hosted more than one competition."
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: farm_competition
      join_keys[0]:
      path_score: 0.9083333333333333
      sql: "SELECT \"farm_competition\".\"host_city_id\" AS \"host_city_id\" FROM \"farm_competition\" GROUP BY \"farm_competition\".\"host_city_id\" HAVING \"farm_competition\".\"host_city_id\" > 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[4]{id,host_city_id}:
        E1,3
        E2,2
        E3,4
        E4,5
  schemas[1]:
    [0]:
      id: C1
      field_map:
        host_city_id: host_city_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The evidence shows host_city_id values (3, 2, 4, 5) for cities that have hosted more than one competition, but the official names of these cities are not provided in the evidence rows.","cited_ids":["E1","E2","E3","E4"]}
```

**Parsed Answer Used For Scoring**

```text
The evidence shows host_city_id values (3, 2, 4, 5) for cities that have hosted more than one competition, but the official names of these cities are not provided in the evidence rows.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.734` |
| `evidence_count` | `4` |
| `join_count` | `1` |
| `context_tokens` | `12` |
| `prompt_tokens` | `46` |
| `api_input_tokens` | `357` |
| `api_output_tokens` | `71` |
| `cited_ids` | `E1, E2, E3, E4` |
| `error` | `[none]` |

## Example 9

- Schema: `farm`
- Question: What are the official names of cities that have hosted more than one competition?
- Ground Truth: Aroostook

**Expected SQL Query**

```sql
SELECT T1.Official_Name FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID GROUP BY T2.Host_city_ID HAVING COUNT(*)  >  1
```

**LakePrompt Generated Query**

```sql
SELECT "farm_competition"."host_city_id" AS "host_city_id" FROM "farm_competition" GROUP BY "farm_competition"."host_city_id"
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What are the official names of cities that have hosted more than one competition?
```

**Response**

```text
I don't have access to any data about cities, competitions, or hosting records in our conversation. To answer your question about which cities have hosted more than one competition and what their official names are, I would need:

1. Information about what type of competitions you're referring to (Olympics, World Cup, etc.)
2. Data showing which cities have hosted these competitions
3. The frequency of hosting for each city
4. The official names of those cities

Could you please provide the relevant data or specify what competitions and dataset you'd like me to analyze?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.238` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `23` |
| `api_input_tokens` | `48` |
| `api_output_tokens` | `119` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
What are the official names of cities that have hosted more than one competition?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table contains information about cities including their official names, population, area, census ranking, and status, but there is no information about competitions or which cities have hosted competitions. To answer your question, I would need additional data that shows which competitions were hosted by which cities.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.171` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `75` |
| `api_input_tokens` | `253` |
| `api_output_tokens` | `69` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
What are the official names of cities that have hosted more than one competition?
```

**Response**

```text
Looking at the provided data, I need to examine the farm_competition table to see which cities have hosted competitions and count how many each city has hosted.

From the farm_competition table, I can see:
- Competition 1: hosted by city_id "1" 
- Competition 2: hosted by city_id "2"
- Competition 3: hosted by city_id "3"

Each city (city_id 1, 2, and 3) has hosted exactly one competition. There are no cities that have hosted more than one competition based on the provided data.

Therefore, there are no cities that have hosted more than one competition in the given dataset.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.411` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `208` |
| `api_input_tokens` | `793` |
| `api_output_tokens` | `148` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
What are the official names of cities that have hosted more than one competition?
```

**Response**

```text
I cannot answer this question confidently from the information provided.

While I can see the schema and sample data, the sample rows only show 3 competitions (competition_ids 1, 2, and 3), each hosted by a different city (host_city_ids 1, 2, and 3 respectively). To determine which cities have hosted more than one competition, I would need to see more complete data from the farm_competition table that might include additional competitions hosted by the same cities.

The sample data doesn't contain any examples of cities hosting multiple competitions, so I cannot identify which cities, if any, have hosted more than one competition.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.532` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `250` |
| `api_input_tokens` | `914` |
| `api_output_tokens` | `141` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E2: {"host_city_id": 2}
E5: {"host_city_id": 5}
E3: {"host_city_id": 1}
E1: {"host_city_id": 3}
E4: {"host_city_id": 4}

Question:
What are the official names of cities that have hosted more than one competition?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence rows only show host_city_id values (1, 2, 3, 4, and 5), but they do not contain:
1. The official names of the cities
2. Information about how many competitions each city has hosted
3. Any way to determine which cities have hosted more than one competition

To answer this question, I would need evidence that includes city names and competition counts or multiple entries per city showing repeated hosting.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `8.07` |
| `evidence_count` | `5` |
| `join_count` | `1` |
| `context_tokens` | `15` |
| `prompt_tokens` | `49` |
| `api_input_tokens` | `143` |
| `api_output_tokens` | `110` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What are the official names of cities that have hosted more than one competition?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: farm_competition
      join_keys[0]:
      path_score: 0.9083333333333333
      sql: "SELECT \"farm_competition\".\"host_city_id\" AS \"host_city_id\" FROM \"farm_competition\" GROUP BY \"farm_competition\".\"host_city_id\""
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[5]{id,host_city_id}:
        E1,3
        E2,2
        E3,1
        E4,4
        E5,5
  schemas[1]:
    [0]:
      id: C1
      field_map:
        host_city_id: host_city_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The evidence provided only contains host_city_id numbers (1, 2, 3, 4, 5) but does not include the official names of the cities. Cannot determine which cities hosted more than one competition or their official names from the given evidence.","cited_ids":["E1","E2","E3","E4","E5"]}
```

**Parsed Answer Used For Scoring**

```text
The evidence provided only contains host_city_id numbers (1, 2, 3, 4, 5) but does not include the official names of the cities. Cannot determine which cities hosted more than one competition or their official names from the given evidence.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.991` |
| `evidence_count` | `5` |
| `join_count` | `1` |
| `context_tokens` | `15` |
| `prompt_tokens` | `49` |
| `api_input_tokens` | `345` |
| `api_output_tokens` | `83` |
| `cited_ids` | `E1, E2, E3, E4, E5` |
| `error` | `[none]` |

## Example 10

- Schema: `farm`
- Question: Show the status of the city that has hosted the greatest number of competitions.
- Ground Truth: Village

**Expected SQL Query**

```sql
SELECT T1.Status FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID GROUP BY T2.Host_city_ID ORDER BY COUNT(*) DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT "farm_competition"."host_city_id" AS "host_city_id", COUNT("farm_competition"."competition_id") AS "agg_0__count__farm_competition__competition_id" FROM "farm_competition" GROUP BY "farm_competition"."host_city_id" ORDER BY "agg_0__count__farm_competition__competition_id" DESC LIMIT 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Show the status of the city that has hosted the greatest number of competitions.
```

**Response**

```text
I don't have access to any data about cities, competitions, or their hosting records. To answer your question about which city has hosted the greatest number of competitions and what its status is, I would need:

1. A dataset containing information about competitions and their host cities
2. Data about the status of those cities (which could mean various things like population, economic status, administrative status, etc.)

Could you please provide the relevant data or clarify what specific dataset you'd like me to analyze?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.907` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `23` |
| `api_input_tokens` | `48` |
| `api_output_tokens` | `106` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
Show the status of the city that has hosted the greatest number of competitions.
```

**Response**

```text
I cannot answer this question using only the provided rows. The table contains information about cities including their area, census ranking, population, and status, but there is no data about competitions or the number of competitions hosted by each city. To answer this question, I would need additional data that includes information about competitions hosted by these cities.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `1.984` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `75` |
| `api_input_tokens` | `253` |
| `api_output_tokens` | `69` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Show the status of the city that has hosted the greatest number of competitions.
```

**Response**

```text
Looking at the provided data, I need to:

1. Count how many competitions each city has hosted
2. Find the city with the most competitions
3. Show that city's status

From the farm_competition table, I can see the host_city_id for each competition:
- Competition 1: hosted by city_id "1"
- Competition 2: hosted by city_id "2" 
- Competition 3: hosted by city_id "3"

Each city has hosted exactly 1 competition, so there's a tie between all three cities.

Since all cities are tied with 1 competition each, I cannot determine which city has hosted the "greatest number" of competitions - they all hosted the same number. The data is insufficient to answer this question as posed, because there is no single city that has hosted more competitions than the others.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.741` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `208` |
| `api_input_tokens` | `793` |
| `api_output_tokens` | `188` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Show the status of the city that has hosted the greatest number of competitions.
```

**Response**

```text
I cannot answer this question confidently from the provided information alone.

While I can see the schema and sample data, the sample rows only show 3 competitions (competition_id 1, 2, and 3), each hosted by a different city (host_city_id 1, 2, and 3 respectively). From this limited sample, each city appears to have hosted exactly one competition, making them tied.

To properly answer which city has hosted the greatest number of competitions, I would need to see the complete dataset from the farm_competition table, not just these 3 sample rows. The sample data is insufficient to determine if any city has hosted more competitions than others.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.037` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `250` |
| `api_input_tokens` | `914` |
| `api_output_tokens` | `148` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"count_competition_id": 2, "host_city_id": 5}

Question:
Show the status of the city that has hosted the greatest number of competitions.
```

**Response**

```text
The evidence is insufficient to answer this question.

The provided evidence (E1) only shows that there are 2 competitions for host_city_id 5, but it doesn't include:
1. Information about other cities and their competition counts to determine which city hosted the greatest number
2. The status information for any city
3. Any city identification or status data

To answer this question, I would need evidence showing competition counts for all cities, identification of which city has the maximum count, and the status information for that city.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.787` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `39` |
| `api_input_tokens` | `97` |
| `api_output_tokens` | `115` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Show the status of the city that has hosted the greatest number of competitions."
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: farm_competition
      join_keys[0]:
      path_score: 0.9083333333333333
      sql: "SELECT \"farm_competition\".\"host_city_id\" AS \"host_city_id\", COUNT(\"farm_competition\".\"competition_id\") AS \"agg_0__count__farm_competition__competition_id\" FROM \"farm_competition\" GROUP BY \"farm_competition\".\"host_city_id\" ORDER BY \"agg_0__count__farm_competition__competition_id\" DESC LIMIT 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,count_competition_id,host_city_id}:
        E1,2,5
  schemas[1]:
    [0]:
      id: C1
      field_map:
        count_competition_id: agg_0__count__farm_competition__competition_id
        host_city_id: host_city_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The city with ID 5 has hosted the greatest number of competitions (2 competitions), but the status information is not available in the provided evidence.","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
The city with ID 5 has hosted the greatest number of competitions (2 competitions), but the status information is not available in the provided evidence.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.52` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `39` |
| `api_input_tokens` | `406` |
| `api_output_tokens` | `47` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 11

- Schema: `farm`
- Question: What is the status of the city that has hosted the most competitions?
- Ground Truth: Village

**Expected SQL Query**

```sql
SELECT T1.Status FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID GROUP BY T2.Host_city_ID ORDER BY COUNT(*) DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT "farm_competition"."host_city_id" AS "host_city_id", COUNT("farm_competition"."competition_id") AS "agg_0__count__farm_competition__competition_id" FROM "farm_competition" GROUP BY "farm_competition"."host_city_id" ORDER BY "agg_0__count__farm_competition__competition_id" DESC LIMIT 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What is the status of the city that has hosted the most competitions?
```

**Response**

```text
I don't have access to any data about cities, competitions, or their hosting records in our conversation. To answer your question about the status of the city that has hosted the most competitions, I would need:

1. Data showing which cities have hosted competitions
2. The number of competitions each city has hosted
3. Information about what "status" refers to (economic status, political status, population status, etc.)

Could you please provide the relevant data or clarify what specific competitions and type of status you're asking about?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.271` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `22` |
| `api_input_tokens` | `47` |
| `api_output_tokens` | `112` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
What is the status of the city that has hosted the most competitions?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table contains information about cities including their area, census ranking, population, and status, but there is no data about competitions or which cities have hosted competitions. To answer your question, I would need additional data showing which cities have hosted competitions and how many.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.195` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `74` |
| `api_input_tokens` | `252` |
| `api_output_tokens` | `66` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
What is the status of the city that has hosted the most competitions?
```

**Response**

```text
Looking at the provided data, I need to find which city has hosted the most competitions and then determine its status.

From the farm_competition table, I can see the host cities for each competition:
- Competition 1: hosted by city_id "1" 
- Competition 2: hosted by city_id "2"
- Competition 3: hosted by city_id "3"

Each city has hosted exactly one competition, so there is a three-way tie for "most competitions hosted" with each city hosting 1 competition.

From the city table, the statuses are:
- City 1 (Grand Falls/Grand-Sault): "Town"
- City 2 (Perth-Andover): "Village" 
- City 3 (Plaster Rock): "Village"

Since there is no single city that has hosted more competitions than the others (they all hosted exactly 1 competition each), I cannot definitively answer which city has hosted the "most" competitions, as there is a tie between all three cities.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `4.437` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `207` |
| `api_input_tokens` | `792` |
| `api_output_tokens` | `232` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
What is the status of the city that has hosted the most competitions?
```

**Response**

```text
I cannot answer this question confidently from the information provided.

While I can see the schema and sample data, the sample rows only show 3 competitions (competition_ids 1, 2, and 3), each hosted by a different city (host_city_ids 1, 2, and 3 respectively). This means each city in the sample has hosted exactly one competition, so there's no clear "most competitions" winner.

To properly answer this question, I would need:
1. The complete dataset of all competitions and their host cities, not just the sample rows
2. Or at least enough data to determine which city has actually hosted the most competitions

From the sample data alone, I cannot determine which city has hosted the most competitions overall.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.278` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `249` |
| `api_input_tokens` | `913` |
| `api_output_tokens` | `164` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"count_competition_id": 2, "host_city_id": 5}

Question:
What is the status of the city that has hosted the most competitions?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence (E1) only shows that there are 2 competitions associated with host city ID 5, but it doesn't indicate whether this city has hosted the most competitions compared to other cities, nor does it provide any information about the status of any city.

To answer this question, I would need:
1. Data showing competition counts for all host cities to determine which has hosted the most
2. Information about the status of that city
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.934` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `38` |
| `api_input_tokens` | `96` |
| `api_output_tokens` | `107` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What is the status of the city that has hosted the most competitions?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: farm_competition
      join_keys[0]:
      path_score: 0.9083333333333333
      sql: "SELECT \"farm_competition\".\"host_city_id\" AS \"host_city_id\", COUNT(\"farm_competition\".\"competition_id\") AS \"agg_0__count__farm_competition__competition_id\" FROM \"farm_competition\" GROUP BY \"farm_competition\".\"host_city_id\" ORDER BY \"agg_0__count__farm_competition__competition_id\" DESC LIMIT 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,count_competition_id,host_city_id}:
        E1,2,5
  schemas[1]:
    [0]:
      id: C1
      field_map:
        count_competition_id: agg_0__count__farm_competition__competition_id
        host_city_id: host_city_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The question asks for the status of the city that has hosted the most competitions, but the evidence row E1 only provides the host_city_id (5) and competition count (2). No status information is available in the provided evidence.","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
The question asks for the status of the city that has hosted the most competitions, but the evidence row E1 only provides the host_city_id (5) and competition count (2). No status information is available in the provided evidence.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.041` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `38` |
| `api_input_tokens` | `405` |
| `api_output_tokens` | `66` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 12

- Schema: `farm`
- Question: Please show the themes of competitions with host cities having populations larger than 1000.
- Ground Truth: ["Carnival M is back!", "Codehunters", "MTV Asia Aid"]

**Expected SQL Query**

```sql
SELECT T2.Theme FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID WHERE T1.Population  >  1000
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Please show the themes of competitions with host cities having populations larger than 1000.
```

**Response**

```text
I don't have access to any data about competitions, host cities, or their populations in our current conversation. To answer your question about themes of competitions with host cities having populations larger than 1000, I would need:

1. A dataset containing information about competitions
2. Data about host cities and their populations
3. Information about the themes of these competitions

Could you please provide the relevant data or specify which dataset you'd like me to analyze? Once you share that information, I'll be happy to help identify the themes of competitions hosted in cities with populations over 1000.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.741` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `23` |
| `api_input_tokens` | `50` |
| `api_output_tokens` | `123` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
Please show the themes of competitions with host cities having populations larger than 1000.
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown contains information about cities (including their populations), but there is no information about competitions or their themes. To answer your question, I would need data from a table that contains information about competitions and their themes, along with their host cities.

The provided data only shows three cities with their populations (all of which are larger than 1000), but lacks any competition-related information.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.125` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.076` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `75` |
| `api_input_tokens` | `255` |
| `api_output_tokens` | `94` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Please show the themes of competitions with host cities having populations larger than 1000.
```

**Response**

```text
Looking at the data provided, I need to:
1. Find cities with populations larger than 1000
2. Find competitions hosted by those cities
3. Show the themes of those competitions

From the city table, cities with population > 1000:
- Grand Falls/Grand-Sault (city_id: 1): population 5706.0
- Perth-Andover (city_id: 2): population 1778.0  
- Plaster Rock (city_id: 3): population 1135.0

From the farm_competition table, competitions hosted by these cities:
- Competition 1: hosted by city_id 1, theme "Carnival M is back!"
- Competition 2: hosted by city_id 2, theme "Codehunters"
- Competition 3: hosted by city_id 3, theme "MTV Asia Aid"

**Answer:** The themes of competitions with host cities having populations larger than 1000 are:
- "Carnival M is back!"
- "Codehunters" 
- "MTV Asia Aid"
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.851` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `208` |
| `api_input_tokens` | `795` |
| `api_output_tokens` | `248` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
Please show the themes of competitions with host cities having populations larger than 1000.
```

**Response**

```text
Looking at the schema and sample data, I can answer this question by joining the `farm_competition` and `city` tables.

From the sample data:

**Competition 1**: 
- Theme: "Carnival M is back!"
- Host city: Grand Falls/Grand-Sault (population: 5,706)

**Competition 2**: 
- Theme: "Codehunters" 
- Host city: Perth-Andover (population: 1,778)

**Competition 3**: 
- Theme: "MTV Asia Aid"
- Host city: Plaster Rock (population: 1,135)

Since all three host cities have populations larger than 1,000, the themes of competitions with host cities having populations larger than 1,000 are:

1. "Carnival M is back!"
2. "Codehunters"
3. "MTV Asia Aid"
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `9.326` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `250` |
| `api_input_tokens` | `916` |
| `api_output_tokens` | `206` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
[no evidence returned]

Question:
Please show the themes of competitions with host cities having populations larger than 1000.
```

**Response**

```text
The evidence is insufficient to answer this question. No evidence rows were provided to determine the themes of competitions with host cities having populations larger than 1000.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.125` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.481` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `37` |
| `api_input_tokens` | `81` |
| `api_output_tokens` | `34` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Please show the themes of competitions with host cities having populations larger than 1000."
  sources[0]:
  evidence_groups[0]:
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
Could not find evidence in the data lake
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.079` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `37` |
| `api_input_tokens` | `[unknown]` |
| `api_output_tokens` | `[unknown]` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

## Example 13

- Schema: `farm`
- Question: What are the themes of competitions that have corresponding host cities with more than 1000 residents?
- Ground Truth: ["Carnival M is back!", "Codehunters", "MTV Asia Aid"]

**Expected SQL Query**

```sql
SELECT T2.Theme FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID WHERE T1.Population  >  1000
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What are the themes of competitions that have corresponding host cities with more than 1000 residents?
```

**Response**

```text
I don't have access to any data about competitions, their themes, host cities, or population information in our conversation. To answer your question about which competition themes correspond to host cities with more than 1000 residents, I would need:

1. A dataset or information about competitions and their themes
2. Data about which cities hosted these competitions
3. Population data for those host cities

Could you please provide the relevant data or specify which competitions and cities you're referring to? Once I have that information, I'll be happy to help identify the themes of competitions hosted in cities with populations over 1000.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.609` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `25` |
| `api_input_tokens` | `53` |
| `api_output_tokens` | `129` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Question:
What are the themes of competitions that have corresponding host cities with more than 1000 residents?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown contains information about cities (including their population, area, and status), but there is no information about competitions or their themes. To answer your question about competition themes and their host cities, I would need data from a different table or additional columns that contain information about competitions and their themes.

The provided data only shows three cities with their population data:
- Grand Falls/Grand-Sault: 5,706 residents
- Perth-Andover: 1,778 residents  
- Plaster Rock: 1,135 residents

All three have more than 1,000 residents, but there's no competition or theme information available in these rows.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.125` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.762` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `46` |
| `prompt_tokens` | `77` |
| `api_input_tokens` | `258` |
| `api_output_tokens` | `158` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: city
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
What are the themes of competitions that have corresponding host cities with more than 1000 residents?
```

**Response**

```text
Looking at the data provided, I need to:

1. Identify cities with more than 1000 residents
2. Find competitions hosted by those cities
3. Get the themes of those competitions

From the city table, cities with more than 1000 residents:
- Grand Falls/Grand-Sault (city_id: 1): 5706 residents
- Perth-Andover (city_id: 2): 1778 residents  
- Plaster Rock (city_id: 3): 1135 residents

All three cities have more than 1000 residents.

From the farm_competition table, the competitions and their themes hosted by these cities are:

1. **"Carnival M is back!"** - hosted by Grand Falls/Grand-Sault (competition_id: 1, year: 2013)
2. **"Codehunters"** - hosted by Perth-Andover (competition_id: 2, year: 2006)
3. **"MTV Asia Aid"** - hosted by Plaster Rock (competition_id: 3, year: 2005)

Therefore, the themes of competitions that have corresponding host cities with more than 1000 residents are:
- Carnival M is back!
- Codehunters
- MTV Asia Aid
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `4.906` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `176` |
| `prompt_tokens` | `210` |
| `api_input_tokens` | `798` |
| `api_output_tokens` | `300` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: city
Columns: city_id, official_name, status, area_km_2, population, census_ranking
Sample rows:
{"area_km_2": "18.06", "census_ranking": "636 of 5008", "city_id": "1", "official_name": "Grand Falls/Grand-Sault", "population": "5706.0", "status": "Town"}
{"area_km_2": "8.89", "census_ranking": "1442 of 5,008", "city_id": "2", "official_name": "Perth-Andover", "population": "1778.0", "status": "Village"}
{"area_km_2": "3.09", "census_ranking": "1936 of 5,008", "city_id": "3", "official_name": "Plaster Rock", "population": "1135.0", "status": "Village"}

Table: competition_record
Columns: competition_id, farm_id, rank
Sample rows:
{"competition_id": "1", "farm_id": "8", "rank": "1"}
{"competition_id": "1", "farm_id": "2", "rank": "2"}
{"competition_id": "1", "farm_id": "3", "rank": "3"}

Table: farm
Columns: farm_id, year, total_horses, working_horses, total_cattle, oxen, bulls, cows, pigs, sheep_and_goats
Sample rows:
{"bulls": "31.6", "cows": "3852.1", "farm_id": "1", "oxen": "805.5", "pigs": "4412.4", "sheep_and_goats": "7956.3", "total_cattle": "8374.5", "total_horses": "5056.5", "working_horses": "3900.1", "year": "1927"}
{"bulls": "32.8", "cows": "3987.0", "farm_id": "2", "oxen": "895.3", "pigs": "6962.9", "sheep_and_goats": "8112.2", "total_cattle": "8604.8", "total_horses": "5486.9", "working_horses": "4090.5", "year": "1928"}
{"bulls": "26.9", "cows": "3873.0", "farm_id": "3", "oxen": "593.7", "pigs": "4161.2", "sheep_and_goats": "7030.8", "total_cattle": "7611.0", "total_horses": "5607.5", "working_horses": "4198.8", "year": "1929"}

Table: farm_competition
Columns: competition_id, year, theme, host_city_id, hosts
Sample rows:
{"competition_id": "1", "host_city_id": "1", "hosts": "Miley Cyrus Jared Leto and Karen Mok", "theme": "Carnival M is back!", "year": "2013"}
{"competition_id": "2", "host_city_id": "2", "hosts": "Leehom Wang and Kelly Rowland", "theme": "Codehunters", "year": "2006"}
{"competition_id": "3", "host_city_id": "3", "hosts": "Alicia Keys", "theme": "MTV Asia Aid", "year": "2005"}

Question:
What are the themes of competitions that have corresponding host cities with more than 1000 residents?
```

**Response**

```text
Looking at the schema and sample data, I can answer this question by joining the farm_competition and city tables.

From the sample data:

**Cities with more than 1000 residents:**
- Grand Falls/Grand-Sault (city_id: 1) - 5,706 residents
- Perth-Andover (city_id: 2) - 1,778 residents  
- Plaster Rock (city_id: 3) - 1,135 residents

**Competition themes for these host cities:**
- Competition 1 (hosted in Grand Falls/Grand-Sault): "Carnival M is back!"
- Competition 2 (hosted in Perth-Andover): "Codehunters"
- Competition 3 (hosted in Plaster Rock): "MTV Asia Aid"

So the themes of competitions hosted in cities with more than 1000 residents are:
1. "Carnival M is back!"
2. "Codehunters" 
3. "MTV Asia Aid"

All three cities in the sample data have populations above 1000, so all competition themes in the sample qualify.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `4.157` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `212` |
| `prompt_tokens` | `252` |
| `api_input_tokens` | `919` |
| `api_output_tokens` | `260` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
[no evidence returned]

Question:
What are the themes of competitions that have corresponding host cities with more than 1000 residents?
```

**Response**

```text
The evidence is insufficient to answer this question. No evidence rows were provided to determine the themes of competitions or information about host cities and their population sizes.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.125` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.49` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `39` |
| `api_input_tokens` | `84` |
| `api_output_tokens` | `33` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What are the themes of competitions that have corresponding host cities with more than 1000 residents?"
  sources[0]:
  evidence_groups[0]:
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
Could not find evidence in the data lake
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.144` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `39` |
| `api_input_tokens` | `[unknown]` |
| `api_output_tokens` | `[unknown]` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

## Example 14

- Schema: `student_assessment`
- Question: which course has most number of registered students?
- Ground Truth: statistics

**Expected SQL Query**

```sql
SELECT T1.course_name FROM courses AS T1 JOIN student_course_registrations AS T2 ON T1.course_id = T2.course_Id GROUP BY T1.course_id ORDER BY count(*) DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT "courses"."course_name" AS "course_name", COUNT("student_course_registrations"."student_id") AS "agg_1__count__student_course_registrations__student_id" FROM "courses" JOIN "student_course_registrations" ON "courses"."course_id" = "student_course_registrations"."course_id" JOIN "student_course_attendance" ON "student_course_registrations"."course_id" = "student_course_attendance"."course_id" GROUP BY "courses"."course_name" ORDER BY "agg_1__count__student_course_registrations__student_id" DESC LIMIT 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
which course has most number of registered students?
```

**Response**

```text
I don't have access to any data about courses or student registrations. To answer which course has the most registered students, I would need:

- A dataset or table showing courses and their enrollment numbers
- Information about the specific institution or system you're asking about
- The time period you're interested in (current semester, academic year, etc.)

Could you please provide the relevant data or specify which educational institution or course catalog you'd like me to analyze?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.163` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `17` |
| `api_input_tokens` | `42` |
| `api_output_tokens` | `98` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Question:
which course has most number of registered students?
```

**Response**

```text
I cannot answer the question "which course has most number of registered students?" using the provided data. The table shown contains only address information (address_id, city, country, address lines, state/province, and zip codes). There is no information about courses or student registrations in these rows.

To answer this question, I would need data from a table that contains course information and student registration counts.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.549` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `67` |
| `prompt_tokens` | `90` |
| `api_input_tokens` | `320` |
| `api_output_tokens` | `86` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
which course has most number of registered students?
```

**Response**

```text
Looking at the student_course_registrations table, I can count the registrations for each course:

**Course 301 (statistics):**
- Student 111: registered on 2008-11-04 10:35:13
- Student 121: registered on 2008-10-04 10:35:13
- Total: 2 registered students

**Course 303 (French):**
- Student 121: registered on 2008-11-14 10:35:13
- Total: 1 registered student

**Course 302 (English):**
- No registrations in the provided data

Therefore, **Course 301 (statistics)** has the most number of registered students with 2 registrations.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `2.697` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `284` |
| `prompt_tokens` | `310` |
| `api_input_tokens` | `1449` |
| `api_output_tokens` | `170` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: addresses
Columns: address_id, line_1, line_2, city, zip_postcode, state_province_county, country
Sample rows:
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
Columns: candidate_id, qualification, assessment_date, asessment_outcome_code
Sample rows:
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
Columns: candidate_id, candidate_details
Sample rows:
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
Columns: course_id, course_name, course_description, other_details
Sample rows:
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
Columns: person_id, first_name, middle_name, last_name, cell_mobile_number, email_address, login_name, password
Sample rows:
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
Columns: person_address_id, person_id, address_id, date_from, date_to
Sample rows:
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
Columns: student_id, course_id, date_of_attendance
Sample rows:
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
Columns: student_id, course_id, registration_date
Sample rows:
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
Columns: student_id, student_details
Sample rows:
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
which course has most number of registered students?
```

**Response**

```text
Based on the schema and sample data provided, I can see that course registrations are tracked in the `student_course_registrations` table, which contains `student_id`, `course_id`, and `registration_date` columns.

From the sample rows in `student_course_registrations`:
- Course 301: 2 registered students (student_id 111 and 121)
- Course 303: 1 registered student (student_id 121)

Based on this limited sample data, **course 301 (statistics)** has the most registered students with 2 registrations.

However, I should note that this conclusion is based only on the sample rows provided. To get the definitive answer for the complete dataset, you would need to run a query that counts all registrations per course across the entire `student_course_registrations` table, not just these sample rows.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `4.492` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `349` |
| `prompt_tokens` | `381` |
| `api_input_tokens` | `1678` |
| `api_output_tokens` | `198` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"count_student_id": 16, "course_name": "statistics"}

Question:
which course has most number of registered students?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence (E1) only shows that the "statistics" course has 16 registered students. To determine which course has the most registered students, I would need enrollment data for all courses to make a comparison.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.525` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `33` |
| `api_input_tokens` | `89` |
| `api_output_tokens` | `59` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "which course has most number of registered students?"
  sources[1]:
    [0]:
      id: S1
      path_id: P2
      tables[3]: courses,student_course_registrations,student_course_attendance
      join_keys[2]:
        [0]: ["courses","course_id","student_course_registrations","course_id"]
        [1]: ["student_course_registrations","course_id","student_course_attendance","course_id"]
      path_score: 0.7833333333333334
      sql: "SELECT \"courses\".\"course_name\" AS \"course_name\", COUNT(\"student_course_registrations\".\"student_id\") AS \"agg_1__count__student_course_registrations__student_id\" FROM \"courses\" JOIN \"student_course_registrations\" ON \"courses\".\"course_id\" = \"student_course_registrations\".\"course_id\" JOIN \"student_course_attendance\" ON \"student_course_registrations\".\"course_id\" = \"student_course_attendance\".\"course_id\" GROUP BY \"courses\".\"course_name\" ORDER BY \"agg_1__count__student_course_registrations__student_id\" DESC LIMIT 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,course_name,count_student_id}:
        E1,statistics,16
  schemas[1]:
    [0]:
      id: C1
      field_map:
        course_name: course_name
        count_student_id: agg_1__count__student_course_registrations__student_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"Statistics has the most number of registered students with 16 students.","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
Statistics has the most number of registered students with 16 students.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `4.279` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `33` |
| `api_input_tokens` | `526` |
| `api_output_tokens` | `30` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 15

- Schema: `student_assessment`
- Question: What is the name of the course with the most registered students?
- Ground Truth: statistics

**Expected SQL Query**

```sql
SELECT T1.course_name FROM courses AS T1 JOIN student_course_registrations AS T2 ON T1.course_id = T2.course_Id GROUP BY T1.course_id ORDER BY count(*) DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT "courses"."course_name" AS "course_name", COUNT("student_course_registrations"."student_id") AS "agg_1__count__student_course_registrations__student_id" FROM "courses" JOIN "student_course_registrations" ON "courses"."course_id" = "student_course_registrations"."course_id" JOIN "student_course_attendance" ON "student_course_registrations"."course_id" = "student_course_attendance"."course_id" GROUP BY "courses"."course_id", "courses"."course_name" ORDER BY "agg_1__count__student_course_registrations__student_id" DESC LIMIT 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What is the name of the course with the most registered students?
```

**Response**

```text
I don't have access to any data about courses or student registrations. To answer your question about which course has the most registered students, I would need:

- A dataset or database containing course information
- Registration numbers or enrollment data for each course
- Course names or identifiers

Could you please provide the relevant data about courses and their enrollment numbers? Once you share that information, I'll be happy to help identify which course has the highest number of registered students.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.656` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `21` |
| `api_input_tokens` | `46` |
| `api_output_tokens` | `99` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Question:
What is the name of the course with the most registered students?
```

**Response**

```text
I cannot answer the question about which course has the most registered students using the provided data. The table shown contains only address information (address_id, city, country, address lines, state/province, and zip codes), but no information about courses or student registrations. To answer this question, I would need data from a table that contains course information and student registration counts.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.613` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `67` |
| `prompt_tokens` | `94` |
| `api_input_tokens` | `324` |
| `api_output_tokens` | `79` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What is the name of the course with the most registered students?
```

**Response**

```text
Looking at the student_course_registrations table, I can count the registrations for each course:

- Course 301: 2 registrations (student_id "111" and "121")
- Course 303: 1 registration (student_id "121")

Course 301 has the most registered students with 2 registrations.

From the courses table, course 301 has:
- course_name: "statistics"
- course_description: "statistics"

Therefore, the course with the most registered students is **statistics**.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.569` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `284` |
| `prompt_tokens` | `314` |
| `api_input_tokens` | `1453` |
| `api_output_tokens` | `124` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: addresses
Columns: address_id, line_1, line_2, city, zip_postcode, state_province_county, country
Sample rows:
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
Columns: candidate_id, qualification, assessment_date, asessment_outcome_code
Sample rows:
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
Columns: candidate_id, candidate_details
Sample rows:
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
Columns: course_id, course_name, course_description, other_details
Sample rows:
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
Columns: person_id, first_name, middle_name, last_name, cell_mobile_number, email_address, login_name, password
Sample rows:
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
Columns: person_address_id, person_id, address_id, date_from, date_to
Sample rows:
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
Columns: student_id, course_id, date_of_attendance
Sample rows:
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
Columns: student_id, course_id, registration_date
Sample rows:
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
Columns: student_id, student_details
Sample rows:
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What is the name of the course with the most registered students?
```

**Response**

```text
Looking at the schema and sample data, I can see that:

1. The `student_course_registrations` table contains course registration data with columns: student_id, course_id, registration_date
2. The `courses` table contains course information with columns: course_id, course_name, course_description, other_details

From the sample data in `student_course_registrations`:
- Course 301: 2 registrations (students 111 and 121)
- Course 303: 1 registration (student 121)

From the sample data in `courses`:
- Course 301 is named "statistics"
- Course 303 is named "French"

Based on this limited sample data, course 301 ("statistics") has the most registered students with 2 registrations.

However, I should note that this conclusion is based only on the sample rows provided. To give a definitive answer about which course has the most registered students across the entire database, I would need to see all the data or have access to run a query that counts registrations for all courses.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `4.758` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `349` |
| `prompt_tokens` | `385` |
| `api_input_tokens` | `1682` |
| `api_output_tokens` | `243` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"count_student_id": 16, "course_name": "statistics"}

Question:
What is the name of the course with the most registered students?
```

**Response**

```text
Based on the provided evidence, I cannot definitively answer what course has the most registered students.

The evidence only shows one course - "statistics" with 16 registered students. To determine which course has the MOST students, I would need enrollment data for all courses to make a comparison. With only information about one course, it's impossible to know if this is the highest enrollment or not.

The evidence is insufficient to answer the question.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.919` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `37` |
| `api_input_tokens` | `93` |
| `api_output_tokens` | `94` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What is the name of the course with the most registered students?"
  sources[1]:
    [0]:
      id: S1
      path_id: P2
      tables[3]: courses,student_course_registrations,student_course_attendance
      join_keys[2]:
        [0]: ["courses","course_id","student_course_registrations","course_id"]
        [1]: ["student_course_registrations","course_id","student_course_attendance","course_id"]
      path_score: 0.7833333333333334
      sql: "SELECT \"courses\".\"course_name\" AS \"course_name\", COUNT(\"student_course_registrations\".\"student_id\") AS \"agg_1__count__student_course_registrations__student_id\" FROM \"courses\" JOIN \"student_course_registrations\" ON \"courses\".\"course_id\" = \"student_course_registrations\".\"course_id\" JOIN \"student_course_attendance\" ON \"student_course_registrations\".\"course_id\" = \"student_course_attendance\".\"course_id\" GROUP BY \"courses\".\"course_id\", \"courses\".\"course_name\" ORDER BY \"agg_1__count__student_course_registrations__student_id\" DESC LIMIT 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,course_name,count_student_id}:
        E1,statistics,16
  schemas[1]:
    [0]:
      id: C1
      field_map:
        course_name: course_name
        count_student_id: agg_1__count__student_course_registrations__student_id
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"Statistics","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
Statistics
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `True` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.722` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `37` |
| `api_input_tokens` | `540` |
| `api_output_tokens` | `17` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 16

- Schema: `student_assessment`
- Question: what are the first name and last name of all candidates?
- Ground Truth: [{"first_name": "Shannon", "last_name": "Senger"}, {"first_name": "Virginie", "last_name": "Hartmann"}, {"first_name": "Dariana", "last_name": "Bednar"}, {"first_name": "Verna", "last_name": "Grant"}, {"first_name": "Hoyt", "last_name": "Wintheiser"}, {"first_name": "Mayra", "last_name": "Hartmann"}, {"first_name": "Lizeth", "last_name": "Bartoletti"}, {"first_name": "Nova", "last_name": "Feest"}]

**Expected SQL Query**

```sql
SELECT T2.first_name ,  T2.last_name FROM candidates AS T1 JOIN people AS T2 ON T1.candidate_id = T2.person_id
```

**LakePrompt Generated Query**

```sql
SELECT "candidates"."candidate_details" AS "candidate_details" FROM "candidates"
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
what are the first name and last name of all candidates?
```

**Response**

```text
I don't have access to any data about candidates in our conversation. To answer your question about the first and last names of candidates, I would need you to provide:

1. A dataset, table, or list containing candidate information
2. Context about what type of candidates you're referring to (job candidates, political candidates, etc.)
3. The specific source or database you'd like me to query

Could you please share the relevant data or specify where I should look for this candidate information?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.696` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `20` |
| `api_input_tokens` | `45` |
| `api_output_tokens` | `105` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Question:
what are the first name and last name of all candidates?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown is an "addresses" table that contains address information (address_id, city, country, line_1, line_2, state_province_county, zip_postcode), but it does not contain any information about candidates or their first and last names.

To answer the question about candidates' first and last names, I would need access to a different table that contains candidate information with name fields.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.33` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `67` |
| `prompt_tokens` | `93` |
| `api_input_tokens` | `323` |
| `api_output_tokens` | `100` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
what are the first name and last name of all candidates?
```

**Response**

```text
Based on the provided data, I can answer this question by looking at the candidates table and the people table.

From the candidates table, I can see the candidate IDs:
- Candidate ID 111: Jane
- Candidate ID 121: Robert  
- Candidate ID 131: Alex

From the people table, I can get the first and last names for these same person IDs:
- Person ID 111: Shannon Senger
- Person ID 121: Virginie Hartmann
- Person ID 131: Dariana Bednar

The first name and last name of all candidates are:
1. Shannon Senger
2. Virginie Hartmann  
3. Dariana Bednar
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.438` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `3.291` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `284` |
| `prompt_tokens` | `313` |
| `api_input_tokens` | `1452` |
| `api_output_tokens` | `163` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: addresses
Columns: address_id, line_1, line_2, city, zip_postcode, state_province_county, country
Sample rows:
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
Columns: candidate_id, qualification, assessment_date, asessment_outcome_code
Sample rows:
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
Columns: candidate_id, candidate_details
Sample rows:
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
Columns: course_id, course_name, course_description, other_details
Sample rows:
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
Columns: person_id, first_name, middle_name, last_name, cell_mobile_number, email_address, login_name, password
Sample rows:
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
Columns: person_address_id, person_id, address_id, date_from, date_to
Sample rows:
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
Columns: student_id, course_id, date_of_attendance
Sample rows:
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
Columns: student_id, course_id, registration_date
Sample rows:
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
Columns: student_id, student_details
Sample rows:
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
what are the first name and last name of all candidates?
```

**Response**

```text
Based on the schema provided, I cannot answer this question confidently. 

The `candidates` table only contains `candidate_id` and `candidate_details` columns, where `candidate_details` appears to store just a single name (like "Jane", "Robert", "Alex") rather than separate first and last names.

The `people` table does have `first_name` and `last_name` columns, but there's no clear relationship shown between the `candidates` and `people` tables in the schema. While the sample data shows matching IDs (111, 121, 131) between candidates and people, without knowing the explicit relationship or foreign key constraints, I cannot definitively say these represent the same individuals.

To answer this question properly, I would need either:
1. A clear relationship/foreign key between the candidates and people tables, or
2. The candidates table to have separate first_name and last_name columns
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `5.383` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `349` |
| `prompt_tokens` | `384` |
| `api_input_tokens` | `1681` |
| `api_output_tokens` | `202` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E2: {"candidate_details": "Alex"}
E7: {"candidate_details": "Robin"}
E3: {"candidate_details": "Jane"}
E8: {"candidate_details": "Tao"}
E5: {"candidate_details": "Leo"}
E6: {"candidate_details": "Jack"}
E1: {"candidate_details": "Robert"}
E4: {"candidate_details": "Cindy"}

Question:
what are the first name and last name of all candidates?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence rows only contain first names for each candidate (Alex, Robin, Jane, Tao, Leo, Jack, Robert, and Cindy), but do not include any last names. To answer the question about both first and last names of all candidates, I would need evidence that contains the complete names with both first and last name information.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.567` |
| `evidence_count` | `8` |
| `join_count` | `1` |
| `context_tokens` | `24` |
| `prompt_tokens` | `55` |
| `api_input_tokens` | `170` |
| `api_output_tokens` | `86` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "what are the first name and last name of all candidates?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: candidates
      join_keys[0]:
      path_score: 0.90625
      sql: "SELECT \"candidates\".\"candidate_details\" AS \"candidate_details\" FROM \"candidates\""
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[8]{id,candidate_details}:
        E1,Robert
        E2,Alex
        E3,Jane
        E4,Cindy
        E5,Leo
        E6,Jack
        E7,Robin
        E8,Tao
  schemas[1]:
    [0]:
      id: C1
      field_map:
        candidate_details: candidate_details
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The candidates' names are: Robert, Alex, Jane, Cindy, Leo, Jack, Robin, and Tao. However, the data only contains single names in the candidate_details field, so it's not possible to distinguish between first names and last names from the available evidence.","cited_ids":["E1","E2","E3","E4","E5","E6","E7","E8"]}
```

**Parsed Answer Used For Scoring**

```text
The candidates' names are: Robert, Alex, Jane, Cindy, Leo, Jack, Robin, and Tao. However, the data only contains single names in the candidate_details field, so it's not possible to distinguish between first names and last names from the available evidence.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.027` |
| `evidence_count` | `8` |
| `join_count` | `1` |
| `context_tokens` | `24` |
| `prompt_tokens` | `55` |
| `api_input_tokens` | `329` |
| `api_output_tokens` | `97` |
| `cited_ids` | `E1, E2, E3, E4, E5, E6, E7, E8` |
| `error` | `[none]` |

## Example 17

- Schema: `student_assessment`
- Question: What are the first and last names of all the candidates?
- Ground Truth: [{"first_name": "Shannon", "last_name": "Senger"}, {"first_name": "Virginie", "last_name": "Hartmann"}, {"first_name": "Dariana", "last_name": "Bednar"}, {"first_name": "Verna", "last_name": "Grant"}, {"first_name": "Hoyt", "last_name": "Wintheiser"}, {"first_name": "Mayra", "last_name": "Hartmann"}, {"first_name": "Lizeth", "last_name": "Bartoletti"}, {"first_name": "Nova", "last_name": "Feest"}]

**Expected SQL Query**

```sql
SELECT T2.first_name ,  T2.last_name FROM candidates AS T1 JOIN people AS T2 ON T1.candidate_id = T2.person_id
```

**LakePrompt Generated Query**

```sql
SELECT "candidates"."candidate_details" AS "candidate_details" FROM "candidates"
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What are the first and last names of all the candidates?
```

**Response**

```text
I don't have access to any data about candidates in your question. To answer what the first and last names of candidates are, I would need you to provide:

- A dataset, table, or list containing candidate information
- Context about what type of candidates you're asking about (political candidates, job candidates, etc.)
- The specific source or document you'd like me to analyze

Could you please share the relevant data or specify which candidates you're referring to?
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.257` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `20` |
| `api_input_tokens` | `45` |
| `api_output_tokens` | `99` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Question:
What are the first and last names of all the candidates?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown is an "addresses" table that contains address information (address_id, city, country, address lines, state, zip code), but it does not contain any information about candidates or their first and last names. To answer your question about candidates' names, I would need access to a different table that contains candidate information.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.401` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `67` |
| `prompt_tokens` | `93` |
| `api_input_tokens` | `323` |
| `api_output_tokens` | `82` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What are the first and last names of all the candidates?
```

**Response**

```text
Based on the provided data, I can see information about candidates, but I need to clarify something about the data structure.

From the `candidates` table, I can see there are 3 candidates:
- Candidate ID 111: Jane
- Candidate ID 121: Robert  
- Candidate ID 131: Alex

However, the `candidate_details` field in the candidates table appears to only contain what looks like first names or single name entries.

From the `people` table, I can see more complete name information for people with the same IDs:
- Person ID 111: Shannon Elissa Senger
- Person ID 121: Virginie Jasmin Hartmann
- Person ID 131: Dariana Hayley Bednar

Since the candidate IDs match the person IDs, and assuming these refer to the same individuals, the first and last names of all candidates would be:

1. **Shannon Senger** (first name: Shannon, last name: Senger)
2. **Virginie Hartmann** (first name: Virginie, last name: Hartmann)  
3. **Dariana Bednar** (first name: Dariana, last name: Bednar)

However, if you're looking strictly for information only from the `candidates` table, then I can only provide the single name entries: Jane, Robert, and Alex, but these don't appear to be complete first and last name pairs.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.438` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `6.358` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `284` |
| `prompt_tokens` | `313` |
| `api_input_tokens` | `1452` |
| `api_output_tokens` | `327` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: addresses
Columns: address_id, line_1, line_2, city, zip_postcode, state_province_county, country
Sample rows:
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
Columns: candidate_id, qualification, assessment_date, asessment_outcome_code
Sample rows:
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
Columns: candidate_id, candidate_details
Sample rows:
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
Columns: course_id, course_name, course_description, other_details
Sample rows:
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
Columns: person_id, first_name, middle_name, last_name, cell_mobile_number, email_address, login_name, password
Sample rows:
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
Columns: person_address_id, person_id, address_id, date_from, date_to
Sample rows:
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
Columns: student_id, course_id, date_of_attendance
Sample rows:
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
Columns: student_id, course_id, registration_date
Sample rows:
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
Columns: student_id, student_details
Sample rows:
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What are the first and last names of all the candidates?
```

**Response**

```text
Based on the schema provided, I cannot answer this question confidently. 

The `candidates` table only contains `candidate_id` and `candidate_details` columns, where `candidate_details` appears to store just a single name (like "Jane", "Robert", "Alex") rather than separate first and last names.

While there is a `people` table that has `first_name` and `last_name` columns, there's no clear relationship established between the `candidates` and `people` tables in the schema provided. The `candidate_id` values (111, 121, 131) happen to match some `person_id` values, but without an explicit foreign key relationship or junction table, I cannot assume these are the same entities.

To properly answer this question, I would need either:
1. The `candidates` table to have separate `first_name` and `last_name` columns, or
2. A clear relationship defined between the `candidates` and `people` tables
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `5.202` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `349` |
| `prompt_tokens` | `384` |
| `api_input_tokens` | `1681` |
| `api_output_tokens` | `217` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E4: {"candidate_details": "Cindy"}
E2: {"candidate_details": "Alex"}
E8: {"candidate_details": "Robin"}
E3: {"candidate_details": "Jane"}
E6: {"candidate_details": "Leo"}
E7: {"candidate_details": "Tao"}
E5: {"candidate_details": "Jack"}
E1: {"candidate_details": "Robert"}

Question:
What are the first and last names of all the candidates?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence rows only contain single names for each candidate (Cindy, Alex, Robin, Jane, Leo, Tao, Jack, Robert), but they do not specify which of these are first names or last names, nor do they provide complete first and last name combinations for any of the candidates.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.239` |
| `evidence_count` | `8` |
| `join_count` | `1` |
| `context_tokens` | `24` |
| `prompt_tokens` | `55` |
| `api_input_tokens` | `170` |
| `api_output_tokens` | `77` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What are the first and last names of all the candidates?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: candidates
      join_keys[0]:
      path_score: 0.90625
      sql: "SELECT \"candidates\".\"candidate_details\" AS \"candidate_details\" FROM \"candidates\""
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[8]{id,candidate_details}:
        E1,Robert
        E2,Alex
        E3,Jane
        E4,Cindy
        E5,Jack
        E6,Leo
        E7,Tao
        E8,Robin
  schemas[1]:
    [0]:
      id: C1
      field_map:
        candidate_details: candidate_details
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The candidates' names are: Robert, Alex, Jane, Cindy, Jack, Leo, Tao, and Robin.","cited_ids":["E1","E2","E3","E4","E5","E6","E7","E8"]}
```

**Parsed Answer Used For Scoring**

```text
The candidates' names are: Robert, Alex, Jane, Cindy, Jack, Leo, Tao, and Robin.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.309` |
| `evidence_count` | `8` |
| `join_count` | `1` |
| `context_tokens` | `24` |
| `prompt_tokens` | `55` |
| `api_input_tokens` | `329` |
| `api_output_tokens` | `64` |
| `cited_ids` | `E1, E2, E3, E4, E5, E6, E7, E8` |
| `error` | `[none]` |

## Example 18

- Schema: `student_assessment`
- Question: What are the ids of all students for courses and what are the names of those courses?
- Ground Truth: [{"course_name": "statistics", "student_id": 111}, {"course_name": "statistics", "student_id": 121}, {"course_name": "French", "student_id": 121}, {"course_name": "French", "student_id": 131}, {"course_name": "English", "student_id": 141}, {"course_name": "data structure", "student_id": 151}, {"course_name": "English", "student_id": 161}, {"course_name": "statistics", "student_id": 171}, {"course_name": "statistics", "student_id": 141}]

**Expected SQL Query**

```sql
SELECT T1.student_id ,  T2.course_name FROM student_course_registrations AS T1 JOIN courses AS T2 ON T1.course_id = T2.course_id
```

**LakePrompt Generated Query**

```sql
SELECT "student_course_registrations"."student_id" AS "student_course_registrations__student_id", "courses"."course_name" AS "course_name" FROM "courses" JOIN "student_course_registrations" ON "courses"."course_id" = "student_course_registrations"."course_id" JOIN "student_course_attendance" ON "student_course_registrations"."course_id" = "student_course_attendance"."course_id"
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What are the ids of all students for courses and what are the names of those courses?
```

**Response**

```text
I don't have access to any student or course data to answer your question. To provide the student IDs and corresponding course names, I would need access to a database or dataset that contains:

1. Student enrollment records with student IDs
2. Course information with course names
3. The relationship between students and their enrolled courses

If you have a specific database, spreadsheet, or dataset you'd like me to analyze, please share that information and I'll be happy to help extract the student IDs and course names for you.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.053` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.684` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `26` |
| `api_input_tokens` | `51` |
| `api_output_tokens` | `113` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Question:
What are the ids of all students for courses and what are the names of those courses?
```

**Response**

```text
I cannot answer this question using only the provided rows. The table shown is an "addresses" table that contains address information (address_id, city, country, line_1, line_2, state_province_county, zip_postcode), but it does not contain any information about students, courses, or their relationships. 

To answer your question about student IDs and course names, I would need data from tables that contain student information and course information, which are not provided in these rows.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.053` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.669` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `67` |
| `prompt_tokens` | `99` |
| `api_input_tokens` | `329` |
| `api_output_tokens` | `107` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What are the ids of all students for courses and what are the names of those courses?
```

**Response**

```text
Based on the provided data, I can identify the student IDs and course names from the student_course_registrations table and courses table:

**Student IDs and Course Names:**

1. **Student ID: 111** - Course: **statistics** (course_id: 301)
2. **Student ID: 121** - Course: **statistics** (course_id: 301)
3. **Student ID: 121** - Course: **French** (course_id: 303)

So the complete list is:
- Student 111 is registered for statistics
- Student 121 is registered for statistics and French

Note: Student 131 appears in the students table but has no course registrations in the provided data.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.579` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `3.761` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `284` |
| `prompt_tokens` | `319` |
| `api_input_tokens` | `1458` |
| `api_output_tokens` | `165` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: addresses
Columns: address_id, line_1, line_2, city, zip_postcode, state_province_county, country
Sample rows:
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
Columns: candidate_id, qualification, assessment_date, asessment_outcome_code
Sample rows:
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
Columns: candidate_id, candidate_details
Sample rows:
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
Columns: course_id, course_name, course_description, other_details
Sample rows:
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
Columns: person_id, first_name, middle_name, last_name, cell_mobile_number, email_address, login_name, password
Sample rows:
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
Columns: person_address_id, person_id, address_id, date_from, date_to
Sample rows:
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
Columns: student_id, course_id, date_of_attendance
Sample rows:
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
Columns: student_id, course_id, registration_date
Sample rows:
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
Columns: student_id, student_details
Sample rows:
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What are the ids of all students for courses and what are the names of those courses?
```

**Response**

```text
Based on the schema and sample data provided, I can answer this question by looking at the student course registrations and courses tables.

From the `student_course_registrations` table, I can see the student IDs and course IDs:
- Student 111 is registered for course 301
- Student 121 is registered for course 301 
- Student 121 is registered for course 303

From the `courses` table, I can see the course names:
- Course 301: "statistics"
- Course 302: "English" 
- Course 303: "French"

So the student IDs for courses and their corresponding course names are:

- Student ID 111 → Course 301 (statistics)
- Student ID 121 → Course 301 (statistics)
- Student ID 121 → Course 303 (French)

Note: I'm using the `student_course_registrations` table rather than `student_course_attendance` since the question asks about students "for courses" which suggests enrollment/registration rather than just attendance.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.526` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `4.511` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `349` |
| `prompt_tokens` | `390` |
| `api_input_tokens` | `1687` |
| `api_output_tokens` | `239` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E9: {"course_name": "French", "student_id": 131}
E2: {"course_name": "English", "student_id": 141}
E8: {"course_name": "French", "student_id": 121}
E6: {"course_name": "statistics", "student_id": 121}
E3: {"course_name": "data structure", "student_id": 151}
E5: {"course_name": "statistics", "student_id": 111}
E4: {"course_name": "statistics", "student_id": 171}
E7: {"course_name": "statistics", "student_id": 141}
E1: {"course_name": "English", "student_id": 161}

Question:
What are the ids of all students for courses and what are the names of those courses?
```

**Response**

```text
Based on the provided evidence rows, here are the student IDs and their corresponding course names:

**Student ID 111:** statistics
**Student ID 121:** French, statistics
**Student ID 131:** French
**Student ID 141:** English, statistics
**Student ID 151:** data structure
**Student ID 161:** English
**Student ID 171:** statistics

The courses represented in the data are:
- English
- French
- statistics
- data structure

Each evidence row shows one student-course enrollment, and I've organized them by student ID to show all courses each student is enrolled in.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.895` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `7.71` |
| `evidence_count` | `9` |
| `join_count` | `1` |
| `context_tokens` | `46` |
| `prompt_tokens` | `83` |
| `api_input_tokens` | `257` |
| `api_output_tokens` | `134` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What are the ids of all students for courses and what are the names of those courses?"
  sources[1]:
    [0]:
      id: S1
      path_id: P2
      tables[3]: courses,student_course_registrations,student_course_attendance
      join_keys[2]:
        [0]: ["courses","course_id","student_course_registrations","course_id"]
        [1]: ["student_course_registrations","course_id","student_course_attendance","course_id"]
      path_score: 0.7833333333333334
      sql: "SELECT \"student_course_registrations\".\"student_id\" AS \"student_course_registrations__student_id\", \"courses\".\"course_name\" AS \"course_name\" FROM \"courses\" JOIN \"student_course_registrations\" ON \"courses\".\"course_id\" = \"student_course_registrations\".\"course_id\" JOIN \"student_course_attendance\" ON \"student_course_registrations\".\"course_id\" = \"student_course_attendance\".\"course_id\""
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[9]{id,student_id,course_name}:
        E1,161,English
        E2,141,English
        E3,151,"data structure"
        E4,171,statistics
        E5,111,statistics
        E6,121,statistics
        E7,141,statistics
        E8,121,French
        E9,131,French
  schemas[1]:
    [0]:
      id: C1
      field_map:
        student_id: student_course_registrations__student_id
        course_name: course_name
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The student IDs for courses are: 161 (English), 141 (English and statistics), 151 (data structure), 171 (statistics), 111 (statistics), 121 (statistics and French), 131 (French).","cited_ids":["E1","E2","E3","E4","E5","E6","E7","E8","E9"]}
```

**Parsed Answer Used For Scoring**

```text
The student IDs for courses are: 161 (English), 141 (English and statistics), 151 (data structure), 171 (statistics), 111 (statistics), 121 (statistics and French), 131 (French).
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.895` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.855` |
| `evidence_count` | `9` |
| `join_count` | `1` |
| `context_tokens` | `46` |
| `prompt_tokens` | `83` |
| `api_input_tokens` | `543` |
| `api_output_tokens` | `95` |
| `cited_ids` | `E1, E2, E3, E4, E5, E6, E7, E8, E9` |
| `error` | `[none]` |

## Example 19

- Schema: `student_assessment`
- Question: What is detail of the student who most recently registered course?
- Ground Truth: Martin

**Expected SQL Query**

```sql
SELECT T2.student_details FROM student_course_registrations AS T1 JOIN students AS T2 ON T1.student_id = T2.student_id ORDER BY T1.registration_date DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT "students"."student_id" AS "students__student_id", "students"."student_details" AS "student_details" FROM "courses" JOIN "student_course_attendance" ON "courses"."course_id" = "student_course_attendance"."course_id" JOIN "student_course_registrations" ON "student_course_attendance"."course_id" = "student_course_registrations"."course_id" JOIN "students" ON "student_course_registrations"."student_id" = "students"."student_id" ORDER BY "student_course_registrations"."registration_date" DESC LIMIT 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What is detail of the student who most recently registered course?
```

**Response**

```text
I don't have access to any student registration data or course enrollment information. To answer your question about the student who most recently registered for a course, I would need:

1. A database or dataset containing student registration records
2. Information about registration dates/timestamps
3. Student details (name, ID, contact information, etc.)
4. Course information

If you have this data available, please share it and I'll be happy to help identify the most recent registration and provide the student's details.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.167` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `20` |
| `api_input_tokens` | `45` |
| `api_output_tokens` | `107` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Question:
What is detail of the student who most recently registered course?
```

**Response**

```text
I cannot answer the question about which student most recently registered for a course using only the provided rows. The table shown contains only address information (address_id, city, country, address lines, state, and zip code), but does not include any data about students, course registrations, or registration dates that would be needed to determine which student most recently registered for a course.

To answer this question, I would need access to tables containing student information and course registration data with timestamps.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.9` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `67` |
| `prompt_tokens` | `93` |
| `api_input_tokens` | `323` |
| `api_output_tokens` | `101` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What is detail of the student who most recently registered course?
```

**Response**

```text
Looking at the student_course_registrations table to find the most recent registration:

- Student 111 registered for course 301 on "2008-11-04 10:35:13"
- Student 121 registered for course 301 on "2008-10-04 10:35:13" 
- Student 121 registered for course 303 on "2008-11-14 10:35:13"

The most recent registration date is "2008-11-14 10:35:13" by student 121.

Looking up student 121 in the students table:
Student 121 has student_details: "Martin"

Therefore, the detail of the student who most recently registered for a course is "Martin".
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `2.898` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `284` |
| `prompt_tokens` | `313` |
| `api_input_tokens` | `1452` |
| `api_output_tokens` | `172` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: addresses
Columns: address_id, line_1, line_2, city, zip_postcode, state_province_county, country
Sample rows:
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
Columns: candidate_id, qualification, assessment_date, asessment_outcome_code
Sample rows:
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
Columns: candidate_id, candidate_details
Sample rows:
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
Columns: course_id, course_name, course_description, other_details
Sample rows:
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
Columns: person_id, first_name, middle_name, last_name, cell_mobile_number, email_address, login_name, password
Sample rows:
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
Columns: person_address_id, person_id, address_id, date_from, date_to
Sample rows:
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
Columns: student_id, course_id, date_of_attendance
Sample rows:
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
Columns: student_id, course_id, registration_date
Sample rows:
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
Columns: student_id, student_details
Sample rows:
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What is detail of the student who most recently registered course?
```

**Response**

```text
Looking at the schema and sample data, I can see that the `student_course_registrations` table contains registration dates, and the `students` table contains student details.

From the sample data in `student_course_registrations`, the most recent registration date is "2008-11-14 10:35:13" for student_id "121".

Looking up student_id "121" in the `students` table, the student details are "Martin".

Therefore, the detail of the student who most recently registered for a course is "Martin".

However, I should note that this answer is based only on the sample rows provided. To be completely confident, I would need to see all the data in the `student_course_registrations` table to ensure there isn't a more recent registration date not shown in the samples.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.754` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `349` |
| `prompt_tokens` | `384` |
| `api_input_tokens` | `1681` |
| `api_output_tokens` | `180` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"student_details": "Martin", "student_id": 121}

Question:
What is detail of the student who most recently registered course?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence (E1) only contains details for one student named Martin with ID 121, but there is no information about:
- Course registration dates or timing
- Whether Martin registered for any courses
- Any other students who may have registered for courses
- Which student registered most recently

To answer this question, I would need evidence showing multiple students' course registration records with timestamps or dates to determine who registered most recently.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.843` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `36` |
| `api_input_tokens` | `90` |
| `api_output_tokens` | `102` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What is detail of the student who most recently registered course?"
  sources[1]:
    [0]:
      id: S1
      path_id: P4
      tables[4]: courses,student_course_attendance,student_course_registrations,students
      join_keys[3]:
        [0]: ["courses","course_id","student_course_attendance","course_id"]
        [1]: ["student_course_attendance","course_id","student_course_registrations","course_id"]
        [2]: ["student_course_registrations","student_id","students","student_id"]
      path_score: 0.7333333333333334
      sql: "SELECT \"students\".\"student_id\" AS \"students__student_id\", \"students\".\"student_details\" AS \"student_details\" FROM \"courses\" JOIN \"student_course_attendance\" ON \"courses\".\"course_id\" = \"student_course_attendance\".\"course_id\" JOIN \"student_course_registrations\" ON \"student_course_attendance\".\"course_id\" = \"student_course_registrations\".\"course_id\" JOIN \"students\" ON \"student_course_registrations\".\"student_id\" = \"students\".\"student_id\" ORDER BY \"student_course_registrations\".\"registration_date\" DESC LIMIT 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,student_id,student_details}:
        E1,121,Martin
  schemas[1]:
    [0]:
      id: C1
      field_map:
        student_id: students__student_id
        student_details: student_details
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The student who most recently registered for a course is Martin (student ID: 121).","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
The student who most recently registered for a course is Martin (student ID: 121).
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.513` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `36` |
| `api_input_tokens` | `529` |
| `api_output_tokens` | `34` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 20

- Schema: `student_assessment`
- Question: What details do we have on the students who registered for courses most recently?
- Ground Truth: Martin

**Expected SQL Query**

```sql
SELECT T2.student_details FROM student_course_registrations AS T1 JOIN students AS T2 ON T1.student_id = T2.student_id ORDER BY T1.registration_date DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT "students"."student_id" AS "students__student_id", "students"."student_details" AS "student_details", "student_course_registrations"."registration_date" AS "registration_date" FROM "courses" JOIN "student_course_attendance" ON "courses"."course_id" = "student_course_attendance"."course_id" JOIN "student_course_registrations" ON "student_course_attendance"."course_id" = "student_course_registrations"."course_id" JOIN "students" ON "student_course_registrations"."student_id" = "students"."student_id" ORDER BY "registration_date" DESC
```

```sql
SELECT "students"."student_id" AS "students__student_id", "students"."student_details" AS "student_details", "student_course_registrations"."registration_date" AS "registration_date" FROM "courses" JOIN "student_course_registrations" ON "courses"."course_id" = "student_course_registrations"."course_id" JOIN "student_course_attendance" ON "student_course_registrations"."course_id" = "student_course_attendance"."course_id" JOIN "students" ON "student_course_attendance"."student_id" = "students"."student_id" ORDER BY "registration_date" DESC
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What details do we have on the students who registered for courses most recently?
```

**Response**

```text
I don't have access to any student registration data or course enrollment information. To answer your question about the most recently registered students, I would need access to a database or dataset containing:

- Student registration records
- Registration timestamps/dates
- Student details (names, IDs, contact information, etc.)
- Course information

If you have such data available, please share it and I'll be happy to help analyze which students registered most recently and provide their details. Alternatively, if you're looking to query an existing database, I could help you formulate the appropriate SQL query or analysis approach.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.525` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `23` |
| `api_input_tokens` | `48` |
| `api_output_tokens` | `126` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Question:
What details do we have on the students who registered for courses most recently?
```

**Response**

```text
I cannot answer the question about students who registered for courses most recently using only the provided data. The table shown contains address information (address_id, city, country, address lines, state, and zip code), but there is no information about students, course registrations, or registration dates in these rows.

To answer your question, I would need data from tables that contain student information and course registration details with timestamps or dates.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.585` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `67` |
| `prompt_tokens` | `96` |
| `api_input_tokens` | `326` |
| `api_output_tokens` | `90` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: addresses
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What details do we have on the students who registered for courses most recently?
```

**Response**

```text
Looking at the student_course_registrations table to find the most recent registration dates:

- Student 111: registered on 2008-11-04 10:35:13
- Student 121: registered on 2008-10-04 10:35:13 and 2008-11-14 10:35:13
- Student 131: no registrations in the provided data

The most recent registration date is **2008-11-14 10:35:13**, which belongs to student 121.

From the students table, the details for student 121 are:
- **Student ID**: 121
- **Student Details**: Martin

This student registered for course 303 (French) on that most recent date.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `3.467` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `284` |
| `prompt_tokens` | `316` |
| `api_input_tokens` | `1455` |
| `api_output_tokens` | `167` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: addresses
Columns: address_id, line_1, line_2, city, zip_postcode, state_province_county, country
Sample rows:
{"address_id": "5", "city": "Linnealand", "country": "USA", "line_1": "0900 Roderick Oval\nNew Albina, WA 19200-7914", "line_2": "Suite 096", "state_province_county": "Montana", "zip_postcode": "862"}
{"address_id": "9", "city": "South Minnie", "country": "USA", "line_1": "966 Dach Ports Apt. 322\nLake Harmonyhaven, VA 65235", "line_2": "Apt. 163", "state_province_county": "Texas", "zip_postcode": "716"}
{"address_id": "29", "city": "North Trystanborough", "country": "USA", "line_1": "28550 Broderick Underpass Suite 667\nZakaryhaven, WY 22945-1534", "line_2": "Apt. 419", "state_province_county": "Vermont", "zip_postcode": "112"}

Table: candidate_assessments
Columns: candidate_id, qualification, assessment_date, asessment_outcome_code
Sample rows:
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-07 11:44:34", "candidate_id": "111", "qualification": "A"}
{"asessment_outcome_code": "Pass", "assessment_date": "2010-04-17 11:44:34", "candidate_id": "121", "qualification": "B"}
{"asessment_outcome_code": "Fail", "assessment_date": "2010-04-05 11:44:34", "candidate_id": "131", "qualification": "D"}

Table: candidates
Columns: candidate_id, candidate_details
Sample rows:
{"candidate_details": "Jane", "candidate_id": "111"}
{"candidate_details": "Robert", "candidate_id": "121"}
{"candidate_details": "Alex", "candidate_id": "131"}

Table: courses
Columns: course_id, course_name, course_description, other_details
Sample rows:
{"course_description": "statistics", "course_id": "301", "course_name": "statistics", "other_details": ""}
{"course_description": "English", "course_id": "302", "course_name": "English", "other_details": ""}
{"course_description": "French", "course_id": "303", "course_name": "French", "other_details": ""}

Table: people
Columns: person_id, first_name, middle_name, last_name, cell_mobile_number, email_address, login_name, password
Sample rows:
{"cell_mobile_number": "01955267735", "email_address": "javier.trantow@example.net", "first_name": "Shannon", "last_name": "Senger", "login_name": "pgub", "middle_name": "Elissa", "password": "5e4ff49a61b3544da3ad7dc7e2cf28847564c64c", "person_id": "111"}
{"cell_mobile_number": "(508)319-2970x043", "email_address": "boyer.lonie@example.com", "first_name": "Virginie", "last_name": "Hartmann", "login_name": "bkkv", "middle_name": "Jasmin", "password": "b063331ea8116befaa7b84c59c6a22200f5f8caa", "person_id": "121"}
{"cell_mobile_number": "(262)347-9364x516", "email_address": "leila14@example.net", "first_name": "Dariana", "last_name": "Bednar", "login_name": "zops", "middle_name": "Hayley", "password": "b20b6a9f24aadeda70d54e410c3219f61fb063fb", "person_id": "131"}

Table: people_addresses
Columns: person_address_id, person_id, address_id, date_from, date_to
Sample rows:
{"address_id": "9", "date_from": "2012-09-26 13:21:00", "date_to": "2018-03-21 09:46:30", "person_address_id": "122", "person_id": "111"}
{"address_id": "5", "date_from": "2008-07-31 02:17:25", "date_to": "2018-03-09 02:11:12", "person_address_id": "257", "person_id": "121"}
{"address_id": "88", "date_from": "2008-05-26 20:43:41", "date_to": "2018-03-11 20:26:41", "person_address_id": "269", "person_id": "131"}

Table: student_course_attendance
Columns: student_id, course_id, date_of_attendance
Sample rows:
{"course_id": "301", "date_of_attendance": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "date_of_attendance": "2012-04-09 11:44:34", "student_id": "121"}
{"course_id": "303", "date_of_attendance": "2014-04-09 11:44:34", "student_id": "121"}

Table: student_course_registrations
Columns: student_id, course_id, registration_date
Sample rows:
{"course_id": "301", "registration_date": "2008-11-04 10:35:13", "student_id": "111"}
{"course_id": "301", "registration_date": "2008-10-04 10:35:13", "student_id": "121"}
{"course_id": "303", "registration_date": "2008-11-14 10:35:13", "student_id": "121"}

Table: students
Columns: student_id, student_details
Sample rows:
{"student_details": "Marry", "student_id": "111"}
{"student_details": "Martin", "student_id": "121"}
{"student_details": "Barry", "student_id": "131"}

Question:
What details do we have on the students who registered for courses most recently?
```

**Response**

```text
Looking at the schema and sample data, I can see that student registration information is in the `student_course_registrations` table, which contains `student_id`, `course_id`, and `registration_date`. Student details are in the `students` table.

From the sample data in `student_course_registrations`, the most recent registration date is "2008-11-14 10:35:13" for student_id "121" registering for course_id "303".

However, I cannot answer this question confidently because:

1. **Limited sample data**: The sample rows only show 3 registration records, which may not represent the full dataset or the actual most recent registrations.

2. **Insufficient student details**: The `students` table only contains `student_details` (which appears to be just names like "Martin" for student 121), but doesn't provide comprehensive information about students.

3. **Missing connections**: To get complete student information, I would likely need to join with the `people` table (which has more detailed information like names, contact details, etc.), but the sample data doesn't clearly show how students relate to people records.

To properly answer this question, I would need the complete dataset to identify the truly most recent registrations, and a clearer understanding of how the student, people, and other related tables connect to provide comprehensive student details.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `8.7` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `349` |
| `prompt_tokens` | `387` |
| `api_input_tokens` | `1684` |
| `api_output_tokens` | `300` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E7: {"registration_date": "2008-10-04 10:35:13", "student_details": "Martin", "student_id": 121}
E4: {"registration_date": "2008-11-07 10:35:13", "student_details": "Nikhil", "student_id": 141}
E10: {"registration_date": "2008-11-08 10:35:13", "student_details": "Martin", "student_id": 121}
E13: {"registration_date": "2008-10-04 10:35:13", "student_details": "Nikhil", "student_id": 141}
E6: {"registration_date": "2008-11-07 10:35:13", "student_details": "Sarah", "student_id": 161}
E14: {"registration_date": "2008-11-07 10:35:13", "student_details": "John", "student_id": 151}
E19: {"registration_date": "2008-11-05 10:35:13", "student_details": "Barry", "student_id": 131}
E15: {"registration_date": "2008-11-04 10:35:13", "student_details": "Joe", "student_id": 171}
E18: {"registration_date": "2008-11-08 10:35:13", "student_details": "Joe", "student_id": 171}
E9: {"registration_date": "2008-11-05 10:35:13", "student_details": "Martin", "student_id": 121}
E1: {"registration_date": "2008-11-14 10:35:13", "student_details": "Martin", "student_id": 121}
E8: {"registration_date": "2008-11-06 10:35:13", "student_details": "Nikhil", "student_id": 141}
E11: {"registration_date": "2008-11-08 10:35:13", "student_details": "Nikhil", "student_id": 141}
E5: {"registration_date": "2008-11-04 10:35:13", "student_details": "Nikhil", "student_id": 141}
E12: {"registration_date": "2008-11-06 10:35:13", "student_details": "Sarah", "student_id": 161}
E2: {"registration_date": "2008-11-07 10:35:13", "student_details": "Martin", "student_id": 121}
E16: {"registration_date": "2008-11-07 10:35:13", "student_details": "Joe", "student_id": 171}
E17: {"registration_date": "2008-10-04 10:35:13", "student_details": "Joe", "student_id": 171}
E3: {"registration_date": "2008-11-04 10:35:13", "student_details": "Martin", "student_id": 121}
E20: {"registration_date": "2008-11-04 10:35:13", "student_details": "Marry", "student_id": 111}

Question:
What details do we have on the students who registered for courses most recently?
```

**Response**

```text
Looking at the evidence provided, I need to identify the most recent registration date and then find the students who registered on that date.

Examining all the registration dates in the evidence:
- 2008-10-04: E7, E13, E17
- 2008-11-04: E15, E5, E3, E20
- 2008-11-05: E19, E9
- 2008-11-06: E8, E12
- 2008-11-07: E4, E6, E14, E2, E16
- 2008-11-08: E10, E18, E11
- 2008-11-14: E1

The most recent registration date is **2008-11-14**.

From the evidence, only one student registered on this date:
- **E1**: Martin (student_id: 121) registered on 2008-11-14 10:35:13

Therefore, based on the provided evidence, Martin (student ID 121) is the student who registered for courses most recently, with a registration date of November 14, 2008.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `10.174` |
| `evidence_count` | `20` |
| `join_count` | `2` |
| `context_tokens` | `160` |
| `prompt_tokens` | `194` |
| `api_input_tokens` | `869` |
| `api_output_tokens` | `260` |
| `cited_ids` | `E7, E4, E10, E13, E6, E14, E19, E15, E18, E9, E1, E8, E11, E5, E12, E2, E16, E17, E3, E20` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What details do we have on the students who registered for courses most recently?"
  sources[2]:
    [0]:
      id: S1
      path_id: P4
      tables[4]: courses,student_course_attendance,student_course_registrations,students
      join_keys[3]:
        [0]: ["courses","course_id","student_course_attendance","course_id"]
        [1]: ["student_course_attendance","course_id","student_course_registrations","course_id"]
        [2]: ["student_course_registrations","student_id","students","student_id"]
      path_score: 0.7333333333333334
      sql: "SELECT \"students\".\"student_id\" AS \"students__student_id\", \"students\".\"student_details\" AS \"student_details\", \"student_course_registrations\".\"registration_date\" AS \"registration_date\" FROM \"courses\" JOIN \"student_course_attendance\" ON \"courses\".\"course_id\" = \"student_course_attendance\".\"course_id\" JOIN \"student_course_registrations\" ON \"student_course_attendance\".\"course_id\" = \"student_course_registrations\".\"course_id\" JOIN \"students\" ON \"student_course_registrations\".\"student_id\" = \"students\".\"student_id\" ORDER BY \"registration_date\" DESC"
    [1]:
      id: S2
      path_id: P1
      tables[4]: courses,student_course_registrations,student_course_attendance,students
      join_keys[3]:
        [0]: ["courses","course_id","student_course_registrations","course_id"]
        [1]: ["student_course_registrations","course_id","student_course_attendance","course_id"]
        [2]: ["student_course_attendance","student_id","students","student_id"]
      path_score: 0.7250000000000001
      sql: "SELECT \"students\".\"student_id\" AS \"students__student_id\", \"students\".\"student_details\" AS \"student_details\", \"student_course_registrations\".\"registration_date\" AS \"registration_date\" FROM \"courses\" JOIN \"student_course_registrations\" ON \"courses\".\"course_id\" = \"student_course_registrations\".\"course_id\" JOIN \"student_course_attendance\" ON \"student_course_registrations\".\"course_id\" = \"student_course_attendance\".\"course_id\" JOIN \"students\" ON \"student_course_attendance\".\"student_id\" = \"students\".\"student_id\" ORDER BY \"registration_date\" DESC"
  evidence_groups[2]:
    [0]:
      source: S1
      schema: C1
      rows[9]{id,student_id,student_details,registration_date}:
        E1,121,Martin,"2008-11-14 10:35:13"
        E6,161,Sarah,"2008-11-07 10:35:13"
        E7,121,Martin,"2008-10-04 10:35:13"
        E8,141,Nikhil,"2008-11-06 10:35:13"
        E11,141,Nikhil,"2008-11-08 10:35:13"
        E14,151,John,"2008-11-07 10:35:13"
        E16,171,Joe,"2008-11-07 10:35:13"
        E19,131,Barry,"2008-11-05 10:35:13"
        E20,111,Marry,"2008-11-04 10:35:13"
    [1]:
      source: S2
      schema: C1
      rows[11]{id,student_id,student_details,registration_date}:
        E2,121,Martin,"2008-11-07 10:35:13"
        E3,121,Martin,"2008-11-04 10:35:13"
        E4,141,Nikhil,"2008-11-07 10:35:13"
        E5,141,Nikhil,"2008-11-04 10:35:13"
        E9,121,Martin,"2008-11-05 10:35:13"
        E10,121,Martin,"2008-11-08 10:35:13"
        E12,161,Sarah,"2008-11-06 10:35:13"
        E13,141,Nikhil,"2008-10-04 10:35:13"
        E15,171,Joe,"2008-11-04 10:35:13"
        E17,171,Joe,"2008-10-04 10:35:13"
        E18,171,Joe,"2008-11-08 10:35:13"
  schemas[1]:
    [0]:
      id: C1
      field_map:
        student_id: students__student_id
        student_details: student_details
        registration_date: registration_date
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The students who registered most recently are Martin (student ID 121) who registered on 2008-11-14, followed by Nikhil (student ID 141) and Sarah (student ID 161) who both registered on 2008-11-08 and 2008-11-07 respectively.","cited_ids":["E1","E10","E11","E6","E4","E12"]}
```

**Parsed Answer Used For Scoring**

```text
The students who registered most recently are Martin (student ID 121) who registered on 2008-11-14, followed by Nikhil (student ID 141) and Sarah (student ID 161) who both registered on 2008-11-08 and 2008-11-07 respectively.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.300` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `5.196` |
| `evidence_count` | `20` |
| `join_count` | `2` |
| `context_tokens` | `160` |
| `prompt_tokens` | `194` |
| `api_input_tokens` | `1347` |
| `api_output_tokens` | `99` |
| `cited_ids` | `E1, E10, E11, E6, E4, E12` |
| `error` | `[none]` |
