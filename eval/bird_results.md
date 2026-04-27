# LakePrompt Evaluation Report

## Run Metadata
- Generated at: `2026-04-27T05:02:09.933112+00:00`
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
- `cited_ids`: Evidence IDs cited by the answer when present.
- `error`: Execution or API error captured for that example, if any.

## Aggregate Metrics

### no_context

| Metric | Value |
| --- | --- |
| `n` | `5` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.0` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.0` |
| `mean_latency_seconds` | `4.685` |
| `mean_context_tokens` | `0.0` |
| `mean_prompt_tokens` | `25.0` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### single_table

| Metric | Value |
| --- | --- |
| `n` | `5` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.0004` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.0` |
| `mean_latency_seconds` | `4.4842` |
| `mean_context_tokens` | `392.4` |
| `mean_prompt_tokens` | `423.4` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### naive_multitable

| Metric | Value |
| --- | --- |
| `n` | `5` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.0004` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.0` |
| `mean_latency_seconds` | `4.6222` |
| `mean_context_tokens` | `858.0` |
| `mean_prompt_tokens` | `892.0` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### schema_baseline

| Metric | Value |
| --- | --- |
| `n` | `5` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.0003` |
| `mean_faithfulness` | `0.0` |
| `mean_semantic_score` | `0.0` |
| `mean_latency_seconds` | `6.6912` |
| `mean_context_tokens` | `1015.0` |
| `mean_prompt_tokens` | `1055.0` |
| `mean_join_count` | `0.0` |
| `error_rate` | `0.0` |

### join_no_ranking

| Metric | Value |
| --- | --- |
| `n` | `5` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.4` |
| `mean_faithfulness` | `0.4` |
| `mean_semantic_score` | `0.0` |
| `mean_latency_seconds` | `11.6384` |
| `mean_context_tokens` | `22.0` |
| `mean_prompt_tokens` | `58.6` |
| `mean_join_count` | `0.8` |
| `error_rate` | `0.0` |

### lakeprompt_ranked

| Metric | Value |
| --- | --- |
| `n` | `5` |
| `exact_match_rate` | `0.0` |
| `mean_ground_truth_coverage` | `0.2049` |
| `mean_faithfulness` | `0.8` |
| `mean_semantic_score` | `0.4` |
| `mean_latency_seconds` | `7.5826` |
| `mean_context_tokens` | `22.0` |
| `mean_prompt_tokens` | `58.6` |
| `mean_join_count` | `0.8` |
| `error_rate` | `0.0` |

## Examples

## Example 1

- Schema: `california_schools`
- Question: What is the highest eligible free rate for K-12 students in the schools in Alameda County?
- Ground Truth: 1.0

**Expected SQL Query**

```sql
SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT MAX("frpm"."Percent (%) Eligible Free (K-12)") AS "agg_0__max__frpm__Percent (%) Eligible Free (K-12)" FROM "frpm" WHERE "frpm"."County Name" = 'Alameda'
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?
```

**Response**

```text
I don't have access to current data about free lunch eligibility rates for K-12 students in Alameda County schools. To find this information, you would need to check:

1. **California Department of Education** - They maintain school-level data on free and reduced-price meal eligibility
2. **Alameda County Office of Education** - May have compiled district and school-level statistics
3. **Individual school district websites** - Often publish their own demographic and eligibility data
4. **DataQuest** - California's educational data system that provides school accountability information

The free meal eligibility rate is typically based on the percentage of students who qualify for free or reduced-price meals under federal guidelines, which is often used as an indicator of the socioeconomic status of the student population.

If you're looking for the most current data, I'd recommend starting with the California Department of Education's DataQuest system or contacting the Alameda County Office of Education directly.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.844` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `25` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?
```

**Response**

```text
Looking at the provided data for schools in Alameda County, I need to examine the "Percent (%) Eligible Free (K-12)" column for each school:

1. FAME Public Charter: 0.519779208831647 (51.98%)
2. Envision Academy for Arts & Technology: 0.470886075949367 (47.09%)
3. Aspire California College Preparatory Academy: 0.549180327868853 (54.92%)

The highest eligible free rate for K-12 students among these Alameda County schools is **0.549180327868853 (54.92%)** at Aspire California College Preparatory Academy.

Note: This answer is based only on the three schools provided in the data rows. There may be other schools in Alameda County not included in this dataset.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.322` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `397` |
| `prompt_tokens` | `428` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?
```

**Response**

```text
Looking at the frpm table data for schools in Alameda County, I can find the "Percent (%) Eligible Free (K-12)" values for each school:

1. FAME Public Charter: 0.519779208831647 (51.98%)
2. Envision Academy for Arts & Technology: 0.470886075949367 (47.09%)
3. Aspire California College Preparatory Academy: 0.549180327868853 (54.92%)

The highest eligible free rate for K-12 students among these schools in Alameda County is **0.549180327868853** (approximately 54.92%), which belongs to Aspire California College Preparatory Academy.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.293` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `858` |
| `prompt_tokens` | `892` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: frpm
Columns: CDSCode, Academic Year, County Code, District Code, School Code, County Name, District Name, School Name, District Type, School Type, Educational Option Type, NSLP Provision Status, Charter School (Y/N), Charter School Number, Charter Funding Type, IRC, Low Grade, High Grade, Enrollment (K-12), Free Meal Count (K-12), Percent (%) Eligible Free (K-12), FRPM Count (K-12), Percent (%) Eligible FRPM (K-12), Enrollment (Ages 5-17), Free Meal Count (Ages 5-17), Percent (%) Eligible Free (Ages 5-17), FRPM Count (Ages 5-17), Percent (%) Eligible FRPM (Ages 5-17), 2013-14 CALPADS Fall 1 Certification Status
Sample rows:
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
Columns: cds, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite, NumGE1500
Sample rows:
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
Columns: CDSCode, NCESDist, NCESSchool, StatusType, County, District, School, Street, StreetAbr, City, Zip, State, MailStreet, MailStrAbr, MailCity, MailZip, MailState, Phone, Ext, Website, OpenDate, ClosedDate, Charter, CharterNum, FundingType, DOC, DOCType, SOC, SOCType, EdOpsCode, EdOpsName, EILCode, EILName, GSoffered, GSserved, Virtual, Magnet, Latitude, Longitude, AdmFName1, AdmLName1, AdmEmail1, AdmFName2, AdmLName2, AdmEmail2, AdmFName3, AdmLName3, AdmEmail3, LastUpdate
Sample rows:
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?
```

**Response**

```text
Looking at the data provided, I can see information about eligible free rates for K-12 students in the frpm table, specifically in the "Percent (%) Eligible Free (K-12)" column.

From the sample rows shown, all three schools are in Alameda County, and their eligible free rates for K-12 students are:

1. FAME Public Charter: 0.519779208831647 (approximately 52.0%)
2. Envision Academy for Arts & Technology: 0.470886075949367 (approximately 47.1%)
3. Aspire California College Preparatory Academy: 0.549180327868853 (approximately 54.9%)

Based on these sample rows, the highest eligible free rate for K-12 students among the schools shown is **0.549180327868853 (approximately 54.9%)** at Aspire California College Preparatory Academy.

However, I should note that this is based only on the sample rows provided. To give you the definitive highest rate across all schools in Alameda County, I would need to see the complete dataset rather than just these three sample records.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.46` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `1015` |
| `prompt_tokens` | `1055` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"max_Percent (%) Eligible Free (K-12)": 1.0}

Question:
What is the highest eligible free rate for K-12 students in the schools in Alameda County?
```

**Response**

```text
Based on the provided evidence, I cannot answer the question about the highest eligible free rate for K-12 students in schools in Alameda County.

The evidence is insufficient because:
1. The single evidence row (E1) shows a maximum percentage of 1.0% (or 100% if interpreted as a decimal), but it lacks crucial context
2. There is no indication that this data specifically relates to Alameda County
3. There is no geographic identifier or school district information in the evidence
4. The evidence does not provide enough detail to determine if this represents the actual highest rate for the specified location

To properly answer this question, I would need evidence that explicitly identifies schools or data from Alameda County along with their corresponding eligible free rates for K-12 students.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `10.939` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `7` |
| `prompt_tokens` | `43` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: frpm
      join_keys[0]:
      path_score: 0.9005
      sql: "SELECT MAX(\"frpm\".\"Percent (%) Eligible Free (K-12)\") AS \"agg_0__max__frpm__Percent (%) Eligible Free (K-12)\" FROM \"frpm\" WHERE \"frpm\".\"County Name\" = 'Alameda'"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,max_Percent (%) Eligible Free (K-12)}:
        E1,1.0
  schemas[1]:
    [0]:
      id: C1
      field_map:
        max_Percent (%) Eligible Free (K-12): "agg_0__max__frpm__Percent (%) Eligible Free (K-12)"
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"The highest eligible free rate for K-12 students in schools in Alameda County is 100%.","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
The highest eligible free rate for K-12 students in schools in Alameda County is 100%.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `1.0` |
| `latency_seconds` | `6.058` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `7` |
| `prompt_tokens` | `43` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 2

- Schema: `california_schools`
- Question: Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
- Ground Truth: [0.043478260869565216, 0.07042253521126761, 0.11363636363636363]

**Expected SQL Query**

```sql
SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
```

**Response**

```text
I don't have access to any data about free rates for students aged 5-17 in continuation schools. To answer your question about the lowest three eligible free rates, I would need access to a dataset or document that contains this specific information about continuation school rates and student eligibility criteria.

If you have this data available, please share it and I'll be happy to help identify the lowest three rates for the specified age group.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `2.757` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `24` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Question:
Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
```

**Response**

```text
Based on the provided rows, I cannot answer this question about the lowest three eligible free rates for students aged 5-17 in continuation schools.

The data provided contains information for only 3 schools, and none of them are continuation schools. The schools listed are:

1. FAME Public Charter - K-12 Schools (Public)
2. Envision Academy for Arts & Technology - High Schools (Public) 
3. Aspire California College Preparatory Academy - High Schools (Public)

To answer your question about continuation schools specifically, I would need data rows that include schools with "School Type" indicating they are continuation schools.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.351` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `397` |
| `prompt_tokens` | `427` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
```

**Response**

```text
I need to look for continuation schools in the provided data and find their eligible free rates for students aged 5-17.

Looking at the data provided, I can see information from the frpm table that includes "Percent (%) Eligible Free (Ages 5-17)" for different schools. However, I need to identify which schools are continuation schools.

From the schools table, I can see the "SOCType" field which indicates the school type:
- FAME Public Charter: "K-12 Schools (Public)"
- Envision Academy for Arts & Technology: "High Schools (Public)"
- Alameda County Office of Education: "County Office of Education (COE)"

None of the schools in the provided data are identified as continuation schools. The school types shown are:
- K-12 Schools (Public)
- High Schools (Public)
- County Office of Education (COE)

Since there are no continuation schools in the provided data rows, I cannot answer the question about the lowest three eligible free rates for students aged 5-17 in continuation schools. The data provided is insufficient to answer this question.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.863` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `858` |
| `prompt_tokens` | `891` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: frpm
Columns: CDSCode, Academic Year, County Code, District Code, School Code, County Name, District Name, School Name, District Type, School Type, Educational Option Type, NSLP Provision Status, Charter School (Y/N), Charter School Number, Charter Funding Type, IRC, Low Grade, High Grade, Enrollment (K-12), Free Meal Count (K-12), Percent (%) Eligible Free (K-12), FRPM Count (K-12), Percent (%) Eligible FRPM (K-12), Enrollment (Ages 5-17), Free Meal Count (Ages 5-17), Percent (%) Eligible Free (Ages 5-17), FRPM Count (Ages 5-17), Percent (%) Eligible FRPM (Ages 5-17), 2013-14 CALPADS Fall 1 Certification Status
Sample rows:
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
Columns: cds, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite, NumGE1500
Sample rows:
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
Columns: CDSCode, NCESDist, NCESSchool, StatusType, County, District, School, Street, StreetAbr, City, Zip, State, MailStreet, MailStrAbr, MailCity, MailZip, MailState, Phone, Ext, Website, OpenDate, ClosedDate, Charter, CharterNum, FundingType, DOC, DOCType, SOC, SOCType, EdOpsCode, EdOpsName, EILCode, EILName, GSoffered, GSserved, Virtual, Magnet, Latitude, Longitude, AdmFName1, AdmLName1, AdmEmail1, AdmFName2, AdmLName2, AdmEmail2, AdmFName3, AdmLName3, AdmEmail3, LastUpdate
Sample rows:
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
```

**Response**

```text
I cannot answer this question confidently from the provided information. 

While I can see that the frpm table contains the "Percent (%) Eligible Free (Ages 5-17)" column which would provide the free rates for students aged 5-17, I cannot identify continuation schools from the available data.

The sample rows shown are all from regular charter schools (K-12 and high schools), and none of the available columns in any of the tables clearly indicate which schools are continuation schools. The School Type and SOCType columns in the samples show "K-12 Schools (Public)" and "High Schools (Public)", but there's no indication of continuation school designation in the provided schema or sample data.

To answer this question, I would need either:
1. Sample data that includes continuation schools, or
2. A clear indication of which column or value identifies continuation schools in the existing schema
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `9.84` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `1015` |
| `prompt_tokens` | `1054` |
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
Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
```

**Response**

```text
The evidence is insufficient to answer this question. No evidence rows were provided to determine the lowest three eligible free rates for students aged 5-17 in continuation schools.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `5.02` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `38` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Please list the lowest three eligible free rates for students aged 5-17 in continuation schools."
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
| `latency_seconds` | `3.658` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `38` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

## Example 3

- Schema: `california_schools`
- Question: Please list the zip code of all the charter schools in Fresno County Office of Education.
- Ground Truth: ["93726-5309", "93628-9602", "93706-2611", "93726-5208", "93706-2819"]

**Expected SQL Query**

```sql
SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1
```

**LakePrompt Generated Query**

```sql
SELECT "schools"."Zip" AS "Zip" FROM "schools" WHERE "schools"."District" = 'Fresno County Office of Education' AND "schools"."Charter" = 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Please list the zip code of all the charter schools in Fresno County Office of Education.
```

**Response**

```text
I don't have access to a current database or dataset containing the specific zip codes of charter schools in Fresno County Office of Education. To get this information, I'd recommend:

1. **Fresno County Office of Education website** - They likely maintain a directory of charter schools in their jurisdiction
2. **California Department of Education's school directory** - This would have comprehensive information about all public schools, including charter schools, with their addresses and zip codes
3. **DataQuest (CDE)** - California's education data system that allows you to search for schools by county and type

If you have access to a specific dataset about these schools, I'd be happy to help analyze it to extract the zip code information you're looking for.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.935` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `25` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Question:
Please list the zip code of all the charter schools in Fresno County Office of Education.
```

**Response**

```text
I cannot answer this question using only the provided rows. The data shown is only for schools in Alameda County Office of Education (County Code "01"), but the question asks about charter schools in Fresno County Office of Education. Additionally, the provided data does not include zip code information for any of the schools - the zip code field is not present in these rows.

To answer this question, I would need:
1. Rows containing data for Fresno County Office of Education
2. Zip code information included in the data fields
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.551` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `397` |
| `prompt_tokens` | `428` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
Please list the zip code of all the charter schools in Fresno County Office of Education.
```

**Response**

```text
I cannot answer this question using only the provided data. The rows I have access to only contain information about schools in Alameda County Office of Education, not Fresno County Office of Education. All the schools in the provided data are located in Alameda County (County Code "01"), while you're asking about charter schools in Fresno County Office of Education.

To answer your question, I would need data rows that contain information about schools in Fresno County Office of Education.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `3.502` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `858` |
| `prompt_tokens` | `892` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: frpm
Columns: CDSCode, Academic Year, County Code, District Code, School Code, County Name, District Name, School Name, District Type, School Type, Educational Option Type, NSLP Provision Status, Charter School (Y/N), Charter School Number, Charter Funding Type, IRC, Low Grade, High Grade, Enrollment (K-12), Free Meal Count (K-12), Percent (%) Eligible Free (K-12), FRPM Count (K-12), Percent (%) Eligible FRPM (K-12), Enrollment (Ages 5-17), Free Meal Count (Ages 5-17), Percent (%) Eligible Free (Ages 5-17), FRPM Count (Ages 5-17), Percent (%) Eligible FRPM (Ages 5-17), 2013-14 CALPADS Fall 1 Certification Status
Sample rows:
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
Columns: cds, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite, NumGE1500
Sample rows:
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
Columns: CDSCode, NCESDist, NCESSchool, StatusType, County, District, School, Street, StreetAbr, City, Zip, State, MailStreet, MailStrAbr, MailCity, MailZip, MailState, Phone, Ext, Website, OpenDate, ClosedDate, Charter, CharterNum, FundingType, DOC, DOCType, SOC, SOCType, EdOpsCode, EdOpsName, EILCode, EILName, GSoffered, GSserved, Virtual, Magnet, Latitude, Longitude, AdmFName1, AdmLName1, AdmEmail1, AdmFName2, AdmLName2, AdmEmail2, AdmFName3, AdmLName3, AdmEmail3, LastUpdate
Sample rows:
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
Please list the zip code of all the charter schools in Fresno County Office of Education.
```

**Response**

```text
I cannot answer this question confidently from the information provided. 

While I can see the schema includes the necessary columns (County, District, Charter, and Zip fields), the sample data only shows schools from Alameda County Office of Education, not Fresno County Office of Education. Without sample rows from Fresno County, I cannot determine which schools in that county are charter schools or provide their zip codes.

To answer this question, I would need data rows where the County field is "Fresno" and the District field is "Fresno County Office of Education", along with Charter = "1" to identify charter schools.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.239` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `1015` |
| `prompt_tokens` | `1055` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E6: {"Zip": "93662"}
E2: {"Zip": "93726-5208"}
E5: {"Zip": "93706-2611"}
E3: {"Zip": "93726-5309"}
E4: {"Zip": "93628-9602"}
E1: {"Zip": "93706-2819"}

Question:
Please list the zip code of all the charter schools in Fresno County Office of Education.
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence rows only contain zip codes (93662, 93726-5208, 93706-2611, 93726-5309, 93628-9602, and 93706-2819), but there is no information indicating:
1. Whether these zip codes are associated with charter schools
2. Whether they are located in Fresno County Office of Education
3. Any connection between the zip codes and educational institutions

To properly answer this question, I would need evidence that explicitly identifies charter schools and their relationship to the Fresno County Office of Education.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `13.872` |
| `evidence_count` | `6` |
| `join_count` | `1` |
| `context_tokens` | `18` |
| `prompt_tokens` | `54` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Please list the zip code of all the charter schools in Fresno County Office of Education."
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: schools
      join_keys[0]:
      path_score: 0.9005
      sql: "SELECT \"schools\".\"Zip\" AS \"Zip\" FROM \"schools\" WHERE \"schools\".\"District\" = 'Fresno County Office of Education' AND \"schools\".\"Charter\" = 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[6]{id,Zip}:
        E1,93706-2819
        E2,93726-5208
        E3,93726-5309
        E4,93628-9602
        E5,93706-2611
        E6,93662
  schemas[1]:
    [0]:
      id: C1
      field_map:
        Zip: Zip
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"93706-2819, 93726-5208, 93726-5309, 93628-9602, 93706-2611, 93662","cited_ids":["E1","E2","E3","E4","E5","E6"]}
```

**Parsed Answer Used For Scoring**

```text
93706-2819, 93726-5208, 93726-5309, 93628-9602, 93706-2611, 93662
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `1.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `8.28` |
| `evidence_count` | `6` |
| `join_count` | `1` |
| `context_tokens` | `18` |
| `prompt_tokens` | `54` |
| `cited_ids` | `E1, E2, E3, E4, E5, E6` |
| `error` | `[none]` |

## Example 4

- Schema: `california_schools`
- Question: What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?
- Ground Truth: 14429 South Downey Avenue

**Expected SQL Query**

```sql
SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1
```

**LakePrompt Generated Query**

```sql
SELECT "schools"."MailStreet" AS "MailStreet" FROM "schools" JOIN "frpm" ON "schools"."CDSCode" = "frpm"."CDSCode" WHERE "frpm"."School Type" = 'K-12 Schools (Public)' ORDER BY "frpm"."FRPM Count (K-12)" DESC LIMIT 1
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?
```

**Response**

```text
I don't have access to any data about schools, their FRPM (Free and Reduced Price Meal) counts, or their mailing addresses. To answer this question, I would need access to a dataset containing:

- School information including names and addresses
- FRPM enrollment data for K-12 students
- The ability to identify which school has the highest FRPM count

Without this data, I cannot determine which school has the highest FRPM count or provide its mailing address. If you have access to such a dataset, please share it and I'll be happy to help analyze it to find the answer.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.235` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `27` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Question:
What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?
```

**Response**

```text
I cannot answer this question using only the provided rows. The data includes FRPM counts for K-12 students for three schools:

- FAME Public Charter: 715.0
- Envision Academy for Arts & Technology: 186.0  
- Aspire California College Preparatory Academy: 175.0

FAME Public Charter has the highest FRPM count (715.0) for K-12 students, but none of the provided rows contain mailing street address information for any of these schools. The data only includes school identification codes, names, enrollment figures, meal program statistics, and administrative details, but no address information.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.931` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `397` |
| `prompt_tokens` | `430` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?
```

**Response**

```text
Looking at the FRPM data from the frpm table, I need to compare the "FRPM Count (K-12)" values for each school:

- FAME Public Charter: 715.0
- Envision Academy for Arts & Technology: 186.0  
- Aspire California College Preparatory Academy: 175.0

FAME Public Charter has the highest FRPM count for K-12 students with 715.0.

From the schools table, I can find the mailing street address for FAME Public Charter (CDSCode: 01100170109835):

The unabbreviated mailing street address is: **39899 Balentine Drive, Suite 335**
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `5.662` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `858` |
| `prompt_tokens` | `894` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: frpm
Columns: CDSCode, Academic Year, County Code, District Code, School Code, County Name, District Name, School Name, District Type, School Type, Educational Option Type, NSLP Provision Status, Charter School (Y/N), Charter School Number, Charter Funding Type, IRC, Low Grade, High Grade, Enrollment (K-12), Free Meal Count (K-12), Percent (%) Eligible Free (K-12), FRPM Count (K-12), Percent (%) Eligible FRPM (K-12), Enrollment (Ages 5-17), Free Meal Count (Ages 5-17), Percent (%) Eligible Free (Ages 5-17), FRPM Count (Ages 5-17), Percent (%) Eligible FRPM (Ages 5-17), 2013-14 CALPADS Fall 1 Certification Status
Sample rows:
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
Columns: cds, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite, NumGE1500
Sample rows:
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
Columns: CDSCode, NCESDist, NCESSchool, StatusType, County, District, School, Street, StreetAbr, City, Zip, State, MailStreet, MailStrAbr, MailCity, MailZip, MailState, Phone, Ext, Website, OpenDate, ClosedDate, Charter, CharterNum, FundingType, DOC, DOCType, SOC, SOCType, EdOpsCode, EdOpsName, EILCode, EILName, GSoffered, GSserved, Virtual, Magnet, Latitude, Longitude, AdmFName1, AdmLName1, AdmEmail1, AdmFName2, AdmLName2, AdmEmail2, AdmFName3, AdmLName3, AdmEmail3, LastUpdate
Sample rows:
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?
```

**Response**

```text
Looking at the data provided, I can see the FRPM Count (K-12) values for the schools in the sample:

- FAME Public Charter: 715.0
- Envision Academy for Arts & Technology: 186.0  
- Aspire California College Preparatory Academy: 175.0

From this sample, FAME Public Charter has the highest FRPM count for K-12 students with 715.0.

Looking at the schools table for FAME Public Charter (CDSCode: 01100170109835), the unabbreviated mailing street address is:

**39899 Balentine Drive, Suite 335**

However, I should note that I can only answer based on the sample rows provided. There may be other schools in the complete dataset with higher FRPM counts that aren't shown in these samples, so I cannot confidently say this is the school with the highest FRPM count across the entire dataset.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.743` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `1015` |
| `prompt_tokens` | `1057` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E1: {"MailStreet": "13330 Vaughn Street"}

Question:
What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence (E1) only contains a single street address "13330 Vaughn Street" without any information about:
- Which school this address belongs to
- FRPM (Free and Reduced Price Meal) counts for any schools
- K-12 student enrollment data
- Any comparison data between multiple schools

To answer the question about which school has the highest FRPM count for K-12 students and what its unabbreviated mailing street address is, I would need evidence containing FRPM data for multiple schools along with their corresponding addresses.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `13.553` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `43` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?"
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[2]: schools,frpm
      join_keys[1]:
        [0]: ["schools","CDSCode","frpm","CDSCode"]
      path_score: 0.8005
      sql: "SELECT \"schools\".\"MailStreet\" AS \"MailStreet\" FROM \"schools\" JOIN \"frpm\" ON \"schools\".\"CDSCode\" = \"frpm\".\"CDSCode\" WHERE \"frpm\".\"School Type\" = 'K-12 Schools (Public)' ORDER BY \"frpm\".\"FRPM Count (K-12)\" DESC LIMIT 1"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[1]{id,MailStreet}:
        E1,"13330 Vaughn Street"
  schemas[1]:
    [0]:
      id: C1
      field_map:
        MailStreet: MailStreet
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"13330 Vaughn Street","cited_ids":["E1"]}
```

**Parsed Answer Used For Scoring**

```text
13330 Vaughn Street
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `7.932` |
| `evidence_count` | `1` |
| `join_count` | `1` |
| `context_tokens` | `5` |
| `prompt_tokens` | `43` |
| `cited_ids` | `E1` |
| `error` | `[none]` |

## Example 5

- Schema: `california_schools`
- Question: Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.
- Ground Truth: [null, "(510) 596-8901", null, "(510) 686-4131", "(510) 452-2063", "(510) 842-1181", "(510) 748-4008", "(510) 748-4017", "(510) 995-4300", "(510) 748-4314", "(510) 809-9800", "(510) 809-9800", "(510) 300-1340", "(510) 300-1560", "(510) 931-7868", "(510) 543-4124", "(510) 370-3334", "(925) 443-1690", "(510) 635-7170", "(510) 562-5238", "(510) 382-9932", "(510) 562-8225", "(510) 658-2900", "(510) 904-6440", "(510) 893-8701", "(510) 893-8701", "(510) 285-7511", "(510) 893-8700", "(510) 874-7255", "(510) 436-5487", "(510) 992-7800", "(510) 879-1282", "(510) 562-8030", "(510) 436-9760", "(510) 452-2100", "(510) 633-0750", "(510) 562-8030", "(510) 689-2035", "(510) 444-1702", "(510) 577-9557", "(510) 594-3900", "(510) 562-8801", "(510) 635-7737", "(510) 638-9445", "(510) 873-8800", "(510) 568-3101", "(510) 655-0540", "(510) 879-3140", "(510) 258-0106", "(510) 317-2330", "(530) 894-3952", "(530) 891-3138", "(530) 891-3154", "(530) 345-6600", "(530) 892-1676", "(530) 879-7483", "(530) 532-1165", "(530) 636-4479", "(530) 534-1633", "(530) 872-4100", "(510) 262-1511", "(510) 685-9886", "(510) 435-2032", "(925) 755-1252", "(209) 365-4060", "(510) 235-4522", "(510) 235-2066", "(510) 235-2465", "(510) 235-2465", "(510) 222-3500", null, null, "(707) 464-9828", "(530) 672-3095", "(916) 936-2333", "(530) 644-4552", "(559) 222-8439", "(559) 305-7565", "(559) 420-1234", null, "(559) 225-7737", null, "(559) 490-4290", "(559) 278-8263", null, "(559) 492-7717", "(559) 229-3529", null, "(805) 581-0202", "(855) 937-4227", "(559) 255-9017", "(559) 222-8439", "(626) 921-8200", "(559) 486-1166", "(530) 361-6480", null, "(707) 822-3348", "(707) 822-4845", "(707) 832-4194", "(707) 825-8804", "(707) 822-5626", "(707) 269-9490", "(707) 268-0854", "(707) 768-1928", "(760) 353-0140", "(760) 592-7250", "(661) 454-3000", "(661) 855-8200", "(805) 581-0202", "(805) 581-0202", "(661) 364-3811", "(760) 375-1010", "(805) 581-0202", null, "(559) 925-3552", "(707) 987-3063", "(949) 306-8498", "(530) 233-3861", "(530) 256-2994", "(213) 614-1745", "(323) 585-1153", "(323) 257-1499", "(323) 277-2901", null, "(310) 425-1605", null, "(310) 216-6800", "(310) 425-1605", "(323) 443-3100", "(800) 985-0770", "(661) 942-3357", "(626) 962-3311", "(626) 814-0161", "(310) 355-0001", "(310) 355-0001", "(626) 921-8200", "(909) 307-6312", "(310) 671-5578", "(310) 671-5578", "(310) 680-7599", "(323) 298-6420", "(323) 298-6425", "(310) 677-3014", "(310) 910-0230", "(310) 673-0956", "(661) 579-2970", "(661) 722-4287", "(310) 214-3400", "(310) 680-5600", "(310) 412-2286", "(323) 752-8834", "(310) 216-3277", "(562) 912-7017", null, "(213) 413-9183", "(323) 406-7155", "(818) 609-2640", "(323) 846-6694", "(323) 235-6343", "(818) 368-1557", "(213) 471-4686", "(323) 264-7737", "(323) 290-6975", "(323) 778-0125", "(323) 293-3917", "(323) 780-1259", "(323) 259-5700", "(310) 827-5511", "(323) 223-1786", "(323) 779-0544", "(818) 485-0340", "(818) 710-2640", "(213) 765-9084", "(818) 716-0771", "(323) 235-7960", "(213) 240-8700", "(310) 392-8751", "(323) 585-3312", "(213) 745-8141", "(818) 834-5805", "(310) 832-9201", "(818) 994-7614", "(213) 413-5741", "(323) 920-6125", "(323) 732-6613", "(323) 656-6418", "(323) 923-1588", null, "(818) 882-2496", "(818) 350-5000", "(213) 487-0600", "(323) 508-0155", "(213) 385-4015", "(213) 342-2874", "(213) 342-2870", "(323) 905-1377", "(323) 232-9436", "(323) 846-5800", "(323) 756-3930", "(323) 596-2290", "(323) 987-2144", null, "(323) 222-5010", null, "(323) 539-2810", "(323) 294-2008", null, "(424) 789-8337", null, null, "(323) 583-5421", "(213) 413-9600", "(323) 232-9588", "(310) 637-3806", "(818) 485-5900", "(818) 808-0158", "(323) 232-0245", "(818) 758-0300", "(818) 897-1187", "(323) 298-6400", "(323) 231-1202", "(323) 344-0160", "(323) 859-0750", "(323) 817-6550", "(323) 972-9010", "(323) 739-0560", "(323) 939-6400", "(310) 473-2464", "(818) 705-5676", "(310) 842-8555", "(818) 886-0585", "(213) 896-2640", "(323) 846-2225", "(323) 780-3900", "(310) 999-6162", "(323) 298-6400", "(323) 290-6968", "(323) 290-6997", null, "(310) 306-3484", "(323) 420-2100", "(323) 254-1703", "(213) 201-0440", "(323) 947-7311", "(818) 830-1700", "(323) 451-3009", "(323) 352-8034", null, "(323) 418-0600", "(323) 923-0383", "(323) 583-5421", "(323) 565-4800", "(310) 642-9474", "(818) 994-2904", null, "(323) 744-2122", "(323) 276-5545", "(323) 750-2279", "(323) 589-1450", "(213) 448-9747", "(323) 777-2068", "(323) 232-1857", "(323) 565-3251", "(323) 464-4292", "(323) 730-7160", null, "(818) 356-2591", "(323) 249-5750", "(323) 249-5740", "(213) 481-3200", "(323) 344-3622", "(323) 589-2800", "(213) 201-4000", "(323) 826-3925", "(818) 810-6713", null, "(818) 988-9128", "(213) 736-5542", "(323) 352-3206", "(213) 241-8533", "(323) 266-4371", "(818) 753-2712", "(323) 223-9184", "(323) 565-3245", "(323) 568-8613", "(323) 600-6000", "(323) 600-6099", "(323) 780-8300", "(424) 789-8338", "(323) 846-4716", "(323) 235-8400", "(323) 235-8400", "(323) 235-8400", "(213) 381-8484", "(213) 353-5300", "(323) 568-4136", "(323) 905-1210", "(323) 276-5525", "(323) 537-2060", "(323) 584-6664", "(323) 292-2272", "(323) 800-5220", "(213) 929-1046", "(323) 348-4276", null, "(323) 821-1393", "(213) 201-5940", "(323) 462-2840", "(323) 545-1100", "(424) 248-0544", "(323) 826-9616", "(310) 221-0430", "(323) 800-5218", "(323) 756-1426", "(323) 481-2334", "(323) 294-4937", "(818) 934-8910", "(213) 736-5566", "(323) 800-2741", "(213) 377-5708", "(818) 456-4590", "(818) 514-4566", "(323) 269-2156", "(323) 484-0450", "(323) 238-7270", "(323) 641-7283", "(323) 263-3600", "(818) 655-9602", "(818) 646-4992", "(323) 406-8000", "(323) 565-4450", "(310) 410-9940", "(323) 406-8007", "(818) 492-1880", "(818) 492-1890", "(323) 750-8471", "(323) 984-9008", "(323) 537-7225", "(213) 487-7736", "(818) 716-2887", "(323) 751-1460", "(323) 227-7719", "(323) 469-0971", "(213) 413-4245", null, "(818) 716-5783", "(818) 346-5100", "(213) 748-8868", null, "(818) 609-0507", "(323) 780-8900", "(213) 239-6605", "(323) 290-6961", "(661) 952-6028", "(661) 285-1600", "(626) 797-1438", "(626) 797-1438", "(626) 797-7704", "(626) 564-2871", "(626) 797-7704", "(805) 581-0202", null, "(661) 200-5595", "(661) 705-4820", "(661) 702-0755", "(888) 997-4722", null, "(661) 272-1225", null, "(626) 241-3057", "(661) 269-2214", null, "(855) 937-4227", null, null, null, "(951) 250-4553", "(661) 268-1660", "(760) 945-1227", "(310) 943-5444", "(909) 622-0699", "(310) 605-2510", "(323) 566-1965", null, "(310) 725-5800", "(310) 725-5800", "(310) 725-5800", "(310) 725-5800", "(209) 375-6383", "(559) 674-1192", "(559) 675-2070", "(559) 675-6626", "(559) 868-8659", "(559) 642-1422", "(415) 331-7530", "(209) 742-6222", "(707) 964-1128", "(707) 467-1855", "(707) 467-0500", "(707) 463-7080", "(707) 462-0913", "(707) 459-6344", "(707) 459-1400", "(209) 675-7699", "(831) 655-4638", "(831) 704-6805", "(831) 755-0300", "(831) 424-9003", "(831) 582-9820", "(831) 583-2165", "(831) 324-4573", "(707) 252-5522", "(530) 268-9990", "(530) 268-2200", "(714) 619-0245", "(949) 461-1667", "(949) 831-4272", "(949) 305-6111", "(949) 448-7232", "(949) 248-1282", "(714) 836-9023", "(714) 569-0948", "(714) 558-2787", "(714) 560-9000", "(714) 543-0023", "(714) 557-7004", "(530) 346-8340", "(916) 780-6800", "(530) 581-1036", "(530) 581-1036", "(916) 778-4550", "(530) 414-5326", "(916) 408-5200", "(916) 632-6580", "(916) 630-1510", "(916) 778-4544", "(916) 632-6580", "(951) 252-8800", "(760) 391-9200", "(951) 222-8931", "(951) 266-0220", "(951) 672-2400", "(877) 544-1429", "(951) 601-6620", "(951) 275-8820", "(714) 569-0948", "(951) 678-5217", "(951) 926-6776", "(916) 287-4470", "(916) 714-9699", "(916) 744-1200", "(916) 649-7850", "(916) 421-0600", "(916) 277-6200", "(916) 739-8520", "(916) 277-7137", "(916) 325-0910", "(916) 433-5057", "(916) 508-6634", "(916) 533-4861", "(916) 971-2432", "(916) 987-6141", "(916) 979-1788", "(916) 979-1788", "(916) 286-5103", "(916) 286-5170", "(916) 286-1902", "(866) 992-9033", "(916) 286-5161", "(916) 286-5183", "(916) 387-8063", "(916) 844-2283", "(831) 313-0772", "(909) 386-2300", "(760) 947-0006", "(760) 843-6622", "(909) 464-2672", "(951) 252-8800", "(661) 272-1225", null, "(760) 245-3222", "(909) 475-3322", "(909) 382-2211", "(909) 888-3132", "(909) 888-3300", "(909) 888-8390", "(626) 685-9300", "(909) 888-8390", "(909) 884-1410", "(909) 881-1100", "(909) 266-1762", "(909) 473-5152", "(909) 887-7002", "(909) 446-1100", "(760) 949-9202", "(760) 949-8002", "(760) 244-3764", "(760) 956-2632", "(760) 948-4333", "(530) 295-3566", "(619) 579-7232", null, "(760) 767-5850", null, "(661) 272-1225", "(619) 401-4150", "(619) 401-4150", "(619) 420-0066", "(619) 934-0300", null, "(619) 286-0312", "(619) 792-9000", "(760) 743-7880", "(760) 224-0758", "(970) 258-0518", "(760) 743-7880", "(760) 737-3111", "(760) 737-3154", "(760) 480-9845", "(619) 660-3550", "(619) 286-0312", "(951) 252-8800", null, "(760) 631-5842", "(619) 443-0948", "(760) 295-1117", "(619) 303-2782", "(760) 591-0217", null, null, null, "(855) 937-4227", "(760) 494-9646", "(619) 336-0808", "(619) 814-5060", "(619) 233-3242", "(619) 243-5014", "(619) 463-6849", "(619) 243-5014", "(619) 265-2411", "(619) 398-8632", "(619) 525-7320", "(619) 525-7320", "(619) 644-1300", "(619) 795-1190", "(619) 744-3828", "(619) 528-9070", "(619) 677-3017", "(619) 788-4668", "(858) 271-1414", "(619) 704-1020", "(619) 263-2171", "(619) 674-6019", "(619) 600-5321", "(858) 874-4338", "(619) 574-6225", "(619) 694-4809", "(619) 795-3137", null, "(619) 677-2180", "(858) 386-0887", "(619) 546-0000", "(858) 496-1613", "(619) 528-9070", "(858) 678-4812", "(858) 292-1304", "(858) 751-4774", "(619) 243-5014", "(858) 678-2050", "(619) 795-3600", "(858) 496-1613", "(619) 232-2825", "(619) 795-1190", "(805) 581-0202", "(805) 581-0202", "(619) 476-0749", null, "(760) 295-5564", "(760) 945-1227", "(760) 598-0782", "(760) 631-8500", "(760) 820-9501", "(760) 631-4020", "(760) 471-0847", "(760) 749-4375", "(760) 749-5982", "(619) 243-5014", "(619) 243-5014", "(619) 243-5014", "(619) 243-5014", "(619) 243-5014", "(619) 243-5014", null, "(415) 467-2522", "(415) 440-4306", "(415) 734-3310", "(415) 841-2200", "(415) 734-3310", "(415) 734-3310", "(415) 922-1001", "(415) 643-6951", null, "(209) 468-9079", "(209) 468-8140", "(209) 468-5940", "(209) 838-7177", "(209) 955-1477", "(209) 368-4934", "(209) 956-8100", "(209) 835-2597", "(209) 478-1600", "(805) 581-0202", "(209) 477-7014", "(209) 477-7013", null, "(209) 830-6363", "(209) 830-6363", "(916) 866-9033", "(209) 253-1208", "(209) 944-5590", "(209) 943-2389", "(209) 948-4511", "(209) 943-2389", "(209) 390-9861", "(209) 390-9861", "(209) 466-3861", "(209) 462-2282", "(209) 831-5240", "(209) 831-5240", "(209) 831-5240", "(805) 467-2095", "(805) 581-0202", "(650) 556-1110", "(650) 562-7190", "(650) 714-2631", "(650) 556-1110", "(650) 366-1050", "(650) 893-8900", "(805) 348-3333", "(805) 291-1303", "(805) 686-7360", "(805) 734-5600", null, "(805) 966-7392", null, "(650) 947-4939", "(408) 243-9800", "(408) 723-1839", "(408) 286-3330", "(408) 729-3920", "(408) 824-5180", null, "(408) 684-4028", null, "(408) 271-1730", "(408) 217-8951", "(408) 729-1981", "(408) 659-4785", "(408) 708-4664", "(408) 931-6838", "(408) 243-9800", "(669) 600-5695", "(408) 780-0831", "(408) 708-5744", "(408) 926-5477", "(408) 780-0831", null, "(408) 785-0657", "(408) 275-7191", "(408) 937-3752", "(408) 831-3104", "(408) 251-1362", "(408) 479-0253", "(408) 729-2281", "(408) 459-6404", "(408) 275-7190", "(408) 361-1960", "(408) 361-3829", "(877) 806-0920", "(408) 899-2607", "(408) 655-6364", "(408) 729-3420", "(408) 337-5445", "(408) 463-0618", "(408) 271-8120", "(408) 271-8120", "(831) 728-6208", "(530) 295-3566", "(530) 547-9726", "(530) 247-7307", "(530) 949-0154", "(530) 225-0456", null, "(530) 222-9280", null, "(530) 926-5800", "(707) 678-8953", "(707) 453-7234", "(707) 552-6482", "(707) 568-7504", "(707) 585-6510", "(805) 581-0202", "(707) 843-4676", "(707) 778-6414", "(707) 762-9020", "(707) 778-4740", "(707) 522-3320", "(707) 545-0102", "(707) 539-2897", "(707) 823-8618", "(707) 996-3849", "(707) 664-0600", "(855) 937-4227", "(707) 524-2848", "(209) 576-2283", "(209) 538-8082", "(209) 521-3010", "(209) 544-8722", "(209) 874-9463", "(209) 669-3410", "(209) 667-0327", "(530) 295-3566", "(805) 581-0202", "(530) 755-2872", "(530) 822-9667", "(559) 625-8527", "(559) 592-9160", "(559) 713-1324", "(559) 970-5894", "(559) 622-3236", "(209) 606-2213", "(805) 581-0202", "(209) 532-9781", "(805) 486-5449", "(805) 648-5503", "(805) 492-3569", "(805) 777-7999", "(805) 495-7037", "(805) 642-3435", "(855) 937-4227", null, "(805) 640-4421", "(805) 384-1415", "(805) 437-1410", null, "(805) 482-4608", "(805) 222-5188", "(916) 375-7680", "(916) 596-6422", "(530) 692-2210", "(530) 742-2505"]

**Expected SQL Query**

```sql
SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`Charter School (Y/N)` = 1 AND T2.OpenDate > '2000-01-01'
```

**LakePrompt Generated Query**

```sql
SELECT "schools"."Phone" AS "Phone" FROM "schools" WHERE "schools"."Charter" = 1 AND "schools"."FundingType" = 'Directly funded' AND "schools"."OpenDate" > '2000-01-01'
```

### no_context

**Prompt**

```text
Answer the following question as best you can.

Question:
Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.
```

**Response**

```text
I don't have access to a database or dataset containing information about charter-funded schools, their opening dates, or their phone numbers. To answer your question about direct charter-funded schools opened after January 1, 2000, I would need access to:

1. A comprehensive database of charter schools
2. Information about their funding type (specifically "direct charter-funded")
3. Their opening/authorization dates
4. Current contact information including phone numbers

To find this information, you might want to check:
- Your state's Department of Education website
- National charter school databases
- Local school district records
- Charter school authorizer websites

If you have access to a specific dataset with this information, please share it and I'd be happy to help analyze it.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `7.654` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `0` |
| `prompt_tokens` | `24` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### single_table

**Prompt**

```text
Use only the following rows from a single table to answer the question.

Rows:
Table: schools
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.
```

**Response**

```text
Looking at the provided data, I need to identify schools that are:
1. Charter schools (Charter = "1")
2. Directly funded (FundingType = "Directly funded") 
3. Opened after 2000/1/1
4. Have phone numbers available

From the three rows provided:

1. **Alameda County Office of Education** - Not a charter school (Charter = "")
2. **FAME Public Charter** - Charter school, directly funded, opened 2005-08-29 (after 2000/1/1), but Phone = "" (no phone number)
3. **Envision Academy for Arts & Technology** - Charter school, directly funded, opened 2006-08-28 (after 2000/1/1), Phone = "(510) 596-8901"

**Answer:** Based on the provided data, there is one direct charter-funded school opened after 2000/1/1 with an available phone number:

- Envision Academy for Arts & Technology: (510) 596-8901

Note: FAME Public Charter also meets the criteria but does not have a phone number listed in the data.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.002` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `7.266` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `374` |
| `prompt_tokens` | `404` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### naive_multitable

**Prompt**

```text
Use only the following rows from multiple tables to answer the question. Tables are not joined.

Rows:
Table: frpm
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.
```

**Response**

```text
Looking at the provided data, I need to identify direct charter-funded schools that opened after 2000/1/1 and find their phone numbers.

From the schools table, I can see the following direct charter-funded schools (where Charter="1" and FundingType="Directly funded"):

1. **FAME Public Charter** (CDSCode: 01100170109835)
   - OpenDate: "2005-08-29" (after 2000/1/1)
   - Phone: "" (empty/no phone number listed)

2. **Envision Academy for Arts & Technology** (CDSCode: 01100170112607)
   - OpenDate: "2006-08-28" (after 2000/1/1)
   - Phone: "(510) 596-8901"

Based on the available data, here are the phone numbers for direct charter-funded schools opened after 2000/1/1:

- **Envision Academy for Arts & Technology**: (510) 596-8901
- **FAME Public Charter**: No phone number listed in the data
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.002` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `4.791` |
| `evidence_count` | `3` |
| `join_count` | `0` |
| `context_tokens` | `858` |
| `prompt_tokens` | `891` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### schema_baseline

**Prompt**

```text
Use the following schema and sample rows to answer the question.
If you cannot answer confidently from this information alone, say so.

Schema:
Table: frpm
Columns: CDSCode, Academic Year, County Code, District Code, School Code, County Name, District Name, School Name, District Type, School Type, Educational Option Type, NSLP Provision Status, Charter School (Y/N), Charter School Number, Charter Funding Type, IRC, Low Grade, High Grade, Enrollment (K-12), Free Meal Count (K-12), Percent (%) Eligible Free (K-12), FRPM Count (K-12), Percent (%) Eligible FRPM (K-12), Enrollment (Ages 5-17), Free Meal Count (Ages 5-17), Percent (%) Eligible Free (Ages 5-17), FRPM Count (Ages 5-17), Percent (%) Eligible FRPM (Ages 5-17), 2013-14 CALPADS Fall 1 Certification Status
Sample rows:
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170109835", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0728", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "1070.0", "Enrollment (K-12)": "1087.0", "FRPM Count (Ages 5-17)": "702.0", "FRPM Count (K-12)": "715.0", "Free Meal Count (Ages 5-17)": "553.0", "Free Meal Count (K-12)": "565.0", "High Grade": "12", "IRC": "1", "Low Grade": "K", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.65607476635514", "Percent (%) Eligible FRPM (K-12)": "0.657773689052438", "Percent (%) Eligible Free (Ages 5-17)": "0.516822429906542", "Percent (%) Eligible Free (K-12)": "0.519779208831647", "School Code": "0109835", "School Name": "FAME Public Charter", "School Type": "K-12 Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170112607", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "0811", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "376.0", "Enrollment (K-12)": "395.0", "FRPM Count (Ages 5-17)": "182.0", "FRPM Count (K-12)": "186.0", "Free Meal Count (Ages 5-17)": "182.0", "Free Meal Count (K-12)": "186.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible FRPM (K-12)": "0.470886075949367", "Percent (%) Eligible Free (Ages 5-17)": "0.484042553191489", "Percent (%) Eligible Free (K-12)": "0.470886075949367", "School Code": "0112607", "School Name": "Envision Academy for Arts & Technology", "School Type": "High Schools (Public)"}
{"2013-14 CALPADS Fall 1 Certification Status": "1", "Academic Year": "2014-2015", "CDSCode": "01100170118489", "Charter Funding Type": "Directly funded", "Charter School (Y/N)": "1", "Charter School Number": "1049", "County Code": "01", "County Name": "Alameda", "District Code": "10017", "District Name": "Alameda County Office of Education", "District Type": "County Office of Education (COE)", "Educational Option Type": "Traditional", "Enrollment (Ages 5-17)": "230.0", "Enrollment (K-12)": "244.0", "FRPM Count (Ages 5-17)": "168.0", "FRPM Count (K-12)": "175.0", "Free Meal Count (Ages 5-17)": "128.0", "Free Meal Count (K-12)": "134.0", "High Grade": "12", "IRC": "1", "Low Grade": "9", "NSLP Provision Status": "", "Percent (%) Eligible FRPM (Ages 5-17)": "0.730434782608696", "Percent (%) Eligible FRPM (K-12)": "0.717213114754098", "Percent (%) Eligible Free (Ages 5-17)": "0.556521739130435", "Percent (%) Eligible Free (K-12)": "0.549180327868853", "School Code": "0118489", "School Name": "Aspire California College Preparatory Academy", "School Type": "High Schools (Public)"}

Table: satscores
Columns: cds, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite, NumGE1500
Sample rows:
{"AvgScrMath": "418", "AvgScrRead": "418", "AvgScrWrite": "417", "NumGE1500": "14", "NumTstTakr": "88", "cds": "1100170000000", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "398", "rtype": "D", "sname": ""}
{"AvgScrMath": "546", "AvgScrRead": "503", "AvgScrWrite": "505", "NumGE1500": "9", "NumTstTakr": "17", "cds": "1100170109835", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "62", "rtype": "S", "sname": "FAME Public Charter"}
{"AvgScrMath": "387", "AvgScrRead": "397", "AvgScrWrite": "395", "NumGE1500": "5", "NumTstTakr": "71", "cds": "1100170112607", "cname": "Alameda", "dname": "Alameda County Office of Education", "enroll12": "75", "rtype": "S", "sname": "Envision Academy for Arts & Technology"}

Table: schools
Columns: CDSCode, NCESDist, NCESSchool, StatusType, County, District, School, Street, StreetAbr, City, Zip, State, MailStreet, MailStrAbr, MailCity, MailZip, MailState, Phone, Ext, Website, OpenDate, ClosedDate, Charter, CharterNum, FundingType, DOC, DOCType, SOC, SOCType, EdOpsCode, EdOpsName, EILCode, EILName, GSoffered, GSserved, Virtual, Magnet, Latitude, Longitude, AdmFName1, AdmLName1, AdmEmail1, AdmFName2, AdmLName2, AdmEmail2, AdmFName3, AdmLName3, AdmEmail3, LastUpdate
Sample rows:
{"AdmEmail1": "lkmonroe@acoe.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "L Karen", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Monroe", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170000000", "Charter": "", "CharterNum": "", "City": "Hayward", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "", "EILName": "", "EdOpsCode": "", "EdOpsName": "", "Ext": "", "FundingType": "", "GSoffered": "", "GSserved": "", "LastUpdate": "2015-06-23", "Latitude": "37.658212", "Longitude": "-122.09713", "Magnet": "", "MailCity": "Hayward", "MailState": "CA", "MailStrAbr": "313 West Winton Ave.", "MailStreet": "313 West Winton Avenue", "MailZip": "94544-1136", "NCESDist": "0691051", "NCESSchool": "", "OpenDate": "", "Phone": "(510) 887-0152", "SOC": "", "SOCType": "", "School": "", "State": "CA", "StatusType": "Active", "Street": "313 West Winton Avenue", "StreetAbr": "313 West Winton Ave.", "Virtual": "", "Website": "www.acoe.org", "Zip": "94544-1136"}
{"AdmEmail1": "", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170109835", "Charter": "1", "CharterNum": "0728", "City": "Newark", "ClosedDate": "2015-07-31", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "ELEMHIGH", "EILName": "Elementary-High Combination", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "K-12", "GSserved": "K-12", "LastUpdate": "2015-09-01", "Latitude": "37.521436", "Longitude": "-121.99391", "Magnet": "0", "MailCity": "Newark", "MailState": "CA", "MailStrAbr": "39899 Balentine Dr., Ste. 335", "MailStreet": "39899 Balentine Drive, Suite 335", "MailZip": "94560-5359", "NCESDist": "0691051", "NCESSchool": "10546", "OpenDate": "2005-08-29", "Phone": "", "SOC": "65", "SOCType": "K-12 Schools (Public)", "School": "FAME Public Charter", "State": "CA", "StatusType": "Closed", "Street": "39899 Balentine Drive, Suite 335", "StreetAbr": "39899 Balentine Dr., Ste. 335", "Virtual": "P", "Website": "", "Zip": "94560-5359"}
{"AdmEmail1": "laura@envisionacademy.org", "AdmEmail2": "", "AdmEmail3": "", "AdmFName1": "Laura", "AdmFName2": "", "AdmFName3": "", "AdmLName1": "Robell", "AdmLName2": "", "AdmLName3": "", "CDSCode": "01100170112607", "Charter": "1", "CharterNum": "0811", "City": "Oakland", "ClosedDate": "", "County": "Alameda", "DOC": "00", "DOCType": "County Office of Education (COE)", "District": "Alameda County Office of Education", "EILCode": "HS", "EILName": "High School", "EdOpsCode": "TRAD", "EdOpsName": "Traditional", "Ext": "", "FundingType": "Directly funded", "GSoffered": "9-12", "GSserved": "9-12", "LastUpdate": "2015-06-18", "Latitude": "37.80452", "Longitude": "-122.26815", "Magnet": "0", "MailCity": "Oakland", "MailState": "CA", "MailStrAbr": "1515 Webster St.", "MailStreet": "1515 Webster Street", "MailZip": "94612", "NCESDist": "0691051", "NCESSchool": "10947", "OpenDate": "2006-08-28", "Phone": "(510) 596-8901", "SOC": "66", "SOCType": "High Schools (Public)", "School": "Envision Academy for Arts & Technology", "State": "CA", "StatusType": "Active", "Street": "1515 Webster Street", "StreetAbr": "1515 Webster St.", "Virtual": "N", "Website": "www.envisionacademy.org/", "Zip": "94612-3355"}

Question:
Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.
```

**Response**

```text
Based on the schema and sample data provided, I can identify schools that meet your criteria, but I cannot provide a complete answer confidently.

From the sample data, I can see schools that are direct charter-funded and opened after 2000/1/1:

1. **FAME Public Charter** (CDSCode: 01100170109835)
   - Charter: Yes (Charter = "1")
   - Funding Type: "Directly funded"
   - Open Date: 2005-08-29 (after 2000/1/1)
   - Phone: "" (empty/no phone number listed)

2. **Envision Academy for Arts & Technology** (CDSCode: 01100170112607)
   - Charter: Yes (Charter = "1") 
   - Funding Type: "Directly funded"
   - Open Date: 2006-08-28 (after 2000/1/1)
   - Phone: "(510) 596-8901"

However, I cannot answer this question confidently because:
1. The sample data only shows 3 schools total, which is clearly not representative of the full dataset
2. One of the qualifying schools (FAME Public Charter) has no phone number listed
3. There are likely many more direct charter-funded schools opened after 2000 in the complete dataset that aren't shown in these samples

To get a complete and accurate list, you would need to query the full dataset.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.001` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `6.174` |
| `evidence_count` | `0` |
| `join_count` | `0` |
| `context_tokens` | `1015` |
| `prompt_tokens` | `1054` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### join_no_ranking

**Prompt**

```text
Use only the following LakePrompt evidence rows to answer the question.
If the evidence is insufficient, say so.

Evidence:
E3: {"Phone": "(530) 247-6933"}
E19: {"Phone": "(619) 490-9439"}
E9: {"Phone": "(619) 303-2782"}
E20: {"Phone": "(530) 581-1036"}
E7: {"Phone": "(530) 247-7307"}
E6: {"Phone": "(559) 305-7565"}
E12: {"Phone": "(310) 425-1605"}
E13: {"Phone": "(800) 985-0770"}
E14: {"Phone": "(626) 241-3057"}
E1: {"Phone": "(559) 278-8263"}
E16: {"Phone": "(310) 392-8751"}
E18: {"Phone": "(619) 498-8830"}
E5: {"Phone": "(714) 558-2787"}
E8: {"Phone": "(559) 490-4290"}
E4: {"Phone": "(916) 979-1788"}
E15: {"Phone": "(310) 410-9940"}
E17: {"Phone": "(408) 337-5445"}
E2: {"Phone": "(301) 789-5469"}
E10: {"Phone": "(916) 286-1902"}
E11: {"Phone": "(310) 273-2489"}

Question:
Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.
```

**Response**

```text
The evidence is insufficient to answer this question. 

The provided evidence rows only contain phone numbers, but they do not include any information about:
- Whether these are charter-funded schools
- Whether they are direct charter-funded schools specifically
- When these schools were opened
- Any dates or opening information

To answer the question about direct charter-funded schools opened after 2000/1/1, I would need evidence that includes school type/funding information and opening dates, which is not present in the provided data.
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.000` |
| `faithfulness` | `0.000` |
| `semantic_score` | `0.0` |
| `latency_seconds` | `14.808` |
| `evidence_count` | `20` |
| `join_count` | `1` |
| `context_tokens` | `80` |
| `prompt_tokens` | `115` |
| `cited_ids` | `[none]` |
| `error` | `[none]` |

### lakeprompt_ranked

**Prompt**

```text
The following context is encoded as TOON (Token-Oriented Object Notation). Use it exactly.

task: "Answer the question using only the evidence rows. Use sources and schemas only as metadata. Cite the supporting evidence row IDs."
context:
  question: "Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1."
  sources[1]:
    [0]:
      id: S1
      path_id: P1
      tables[1]: schools
      join_keys[0]:
      path_score: 0.9005
      sql: "SELECT \"schools\".\"Phone\" AS \"Phone\" FROM \"schools\" WHERE \"schools\".\"Charter\" = 1 AND \"schools\".\"FundingType\" = 'Directly funded' AND \"schools\".\"OpenDate\" > '2000-01-01'"
  evidence_groups[1]:
    [0]:
      source: S1
      schema: C1
      rows[20]{id,Phone}:
        E1,"(559) 278-8263"
        E2,"(301) 789-5469"
        E3,"(530) 247-6933"
        E4,"(916) 979-1788"
        E5,"(714) 558-2787"
        E6,"(559) 305-7565"
        E7,"(530) 247-7307"
        E8,"(559) 490-4290"
        E9,"(619) 303-2782"
        E10,"(916) 286-1902"
        E11,"(310) 273-2489"
        E12,"(310) 425-1605"
        E13,"(800) 985-0770"
        E14,"(626) 241-3057"
        E15,"(310) 410-9940"
        E16,"(310) 392-8751"
        E17,"(408) 337-5445"
        E18,"(619) 498-8830"
        E19,"(619) 490-9439"
        E20,"(530) 581-1036"
  schemas[1]:
    [0]:
      id: C1
      field_map:
        Phone: Phone
output_format: "{\"answer\":\"Plain-text answer here\",\"cited_ids\":[\"id1\",\"id2\"]}"
rules[2]: "Be concise.","Return only the requested JSON."
```

**Response**

```text
{"answer":"(559) 278-8263, (301) 789-5469, (530) 247-6933, (916) 979-1788, (714) 558-2787, (559) 305-7565, (530) 247-7307, (559) 490-4290, (619) 303-2782, (916) 286-1902, (310) 273-2489, (310) 425-1605, (800) 985-0770, (626) 241-3057, (310) 410-9940, (310) 392-8751, (408) 337-5445, (619) 498-8830, (619) 490-9439, (530) 581-1036","cited_ids":["E1","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11","E12","E13","E14","E15","E16","E17","E18","E19","E20"]}
```

**Parsed Answer Used For Scoring**

```text
(559) 278-8263, (301) 789-5469, (530) 247-6933, (916) 979-1788, (714) 558-2787, (559) 305-7565, (530) 247-7307, (559) 490-4290, (619) 303-2782, (916) 286-1902, (310) 273-2489, (310) 425-1605, (800) 985-0770, (626) 241-3057, (310) 410-9940, (310) 392-8751, (408) 337-5445, (619) 498-8830, (619) 490-9439, (530) 581-1036
```

**Scoring Metrics**

| Metric | Value |
| --- | --- |
| `exact_match` | `False` |
| `ground_truth_coverage` | `0.024` |
| `faithfulness` | `1.000` |
| `semantic_score` | `0.5` |
| `latency_seconds` | `11.985` |
| `evidence_count` | `20` |
| `join_count` | `1` |
| `context_tokens` | `80` |
| `prompt_tokens` | `115` |
| `cited_ids` | `E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20` |
| `error` | `[none]` |
