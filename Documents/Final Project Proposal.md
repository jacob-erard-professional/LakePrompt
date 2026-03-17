## **LakePrompt: LLM Context Engineering Using Joined Tuples**

TLDR: “**RAG, but the retrieval unit is a joined tuple instead of a document chunk.”**

### **1) Motivation**

LLMs are strong at *language*, but weak at *grounded answering* when the answer lives inside a messy collection of CSVs. In a **data lake** (a folder of tables), answering a question often requires:

* finding the right table(s),  
* understanding which columns correspond to the question,  
* **joining multiple tables**, and  
* selecting the **right joined rows** (not just any rows) to use as context.

This project builds a system that helps an LLM answer questions about a data lake by **retrieving and packaging “joined tuples” as context**.

---

## **2) Problem Statement**

Given:

* A data lake: tables (T\_1, T\_2, \\dots, T\_n) (e.g., CSV files)  
* A user question in natural language (q)

Return:

1. An answer (a) produced by an LLM **grounded in lake data**  
2. The supporting evidence: **a small set of rows**, potentially from **multi-table joins**, that justify the answer

Key challenge: If the question requires joining multiple tables, the system must identify and execute the relevant joins and select the most useful joined tuples under a strict token budget.

---

## **3) Core Tasks**

###  **Identifying Joinable Tables Using Column Similarity**

To determine whether columns from two tables are joinable, we evaluate the **overlap of values** using **Jaccard similarity**. Jaccard similarity measures the proportion of shared values between two sets and is a good indicator of potential column matches for joins. If the similarity score exceeds a given threshold, we consider the columns to be joinable.

* **Jaccard Similarity Formula:**  
  J(A, B) \= |A B| / |A B|

* Here, A and B are the sets of values from the two columns being compared. A higher Jaccard score indicates greater overlap and thus a higher likelihood of a successful join.

* **Example:**  
   Consider the column `Customer ID` in a **purchase history table**. If another table in the data lake contains a column with a high Jaccard similarity (such as `User ID`), we can join them to access additional features like `User Age Group` or `Membership Status`.

---

### **Task B — Map Question → Relevant Tables/Columns**

Given question (q), identify:

* candidate tables relevant to the question  
* candidate columns that represent the entities/filters/targets (e.g., “population”, “year”, “county”, “diagnosis”)

Possible approaches:

* keyword / BM25 emebeddings over table+column descriptions and sample values  
* embedding retrieval over column “cards” (name \+ samples \+ summary)  
* hybrid retrieval (keyword \+ embedding)

Output: Top (k) tables and column hints.

In order to index vector embeddings and find the most similar embeddings to a given query embedding, use an HNSW index:

[https://esteininger.medium.com/building-a-vector-search-engine-using-hnsw-and-cosine-similarity-753fb5268839](https://esteininger.medium.com/building-a-vector-search-engine-using-hnsw-and-cosine-similarity-753fb5268839)

---

### **Task C — Discover the Join Path(s) Needed**

If multiple tables are relevant, propose join paths:

* Build candidate join paths in the join graph:  
  * path length 1–3 (2–4 tables)  
* Score join paths using:  
  * how well involved columns match the question  
  * key uniqueness / selectivity (prefer many-to-one)  
  * estimated output size

Output: Top (p) join plans (e.g., `(A join B on …) join C on …`).

---

### **Task D — Retrieve and Rank Joined Tuples** 

Execute join plans (or partial joins) and select the most relevant joined tuples.

You want a small set of context rows that are maximally useful for answering.

Techniques you can implement:

* Pushdown filters inferred from the question:  
  * detect constraints like dates, locations, categories, “top”, “after 2019”, etc.  
* Relevance scoring of joined tuples:  
  * embedding similarity between tuple representation and question  
  * “coverage” objective: select tuples that cover different aspects/mentions in the question  
* Diversity constraints:  
  * avoid returning 20 near-duplicate joined rows

Output: Top (r) joined tuples \+ provenance (table names \+ join keys \+ join path used).

---

### **Task E — Context Packaging (“Prompt Engineering” but data-first)**

Transform joined tuples into LLM-friendly context with:

1. **Flattened joined tuple text**  
* “Row context 1: …”  
2. **Mini-table format**  
* show only the columns needed  
4. **Cited answers**  
* require the model to cite evidence IDs (E1, E2, …)

Deliverable: a prompt template that reliably enforces grounded answers.

---

## **4) What counts as “success”**

Your system should answer questions more accurately **than baselines** on questions that require multi-table reasoning.

Also: answers should be *faithful* to returned evidence (low hallucination).

---

## **5) Evaluation Plan (make it measurable)**

### **Build a small benchmark**

You can start with \~50–200 questions with labels:

* answer type:  
  * single value  
  * list  
* difficulty:  
  * 1-table  
  * 2-table join  
  * 3+ table join  
* ground truth:  
  * SQL query (ideal)  
  * or expected answer \+ evidence rows

### **Metrics**

* **Answer accuracy**  
  * exact match / F1 (for strings)  
  * numeric tolerance (for numbers)  
  * execution accuracy if SQL ground truth exists  
* **Faithfulness / grounding**  
  * “Is the answer supported by evidence rows?”  
  * automated checks when possible (e.g., if answer claims X, verify X exists in evidence)  
  * LLM-judge rubric *with* evidence shown (optional but common)  
* **Efficiency**  
  * latency  
  * number of joins executed  
  * context length (tokens)  
* **Ablations**  
  * remove join retrieval → measure drop  
  * remove filtering → measure drop  
  * different context formats → measure change

---

## **6) Baselines**

At minimum:

1. **No context** (LLM only)  
2. **Single-table retrieval** (top rows from one best table, no joins)  
3. **Naive multi-table context** (top rows from relevant tables concatenated, still no join)  
4. **Join but no ranking** (random/sample joined tuples)

**Datasets**

To get started, you could use any of the public table repositories (a collection of CSVs). For instance:  
[https://github.com/superctj/spider-join-data](https://github.com/superctj/spider-join-data)  
[https://bird-bench.github.io/](https://bird-bench.github.io/)  
[https://github.com/anonymous-repo1/LakeBench](https://github.com/anonymous-repo1/LakeBench)

**7) Example Scenario**

## **Example data lake (3 CSVs)**

### **`customers.csv`**

| customer\_id | name | city |
| ----- | ----- | ----- |
| 101 | Sara | Denver |
| 102 | Omar | Provo |
| 103 | Lina | Denver |

### **`orders.csv`**

| order\_id | customer\_id | order\_date |
| ----- | ----- | ----- |
| 9001 | 101 | 2025-01-15 |
| 9002 | 101 | 2025-02-02 |
| 9003 | 102 | 2025-01-20 |

### **`order_items.csv`**

| order\_id | product | price |
| ----- | ----- | ----- |
| 9001 | bike helmet | 45 |
| 9001 | water bottle | 12 |
| 9002 | gloves | 25 |
| 9003 | bike light | 30 |

---

## **User question**

“How much did customers in Denver spend in January 2025?”

This **cannot** be answered from a single table:

* “Denver” lives in `customers`  
* “January 2025” filter lives in `orders`  
* “spent” (sum of prices) lives in `order_items`

So we need a **join** across all three.

---

## **LakePrompt workflow (what the system does)**

### **Step 1 — Identify relevant tables/columns**

From the question:

* entity/location: “Denver” → likely `customers.city`  
* time constraint: “January 2025” → likely `orders.order_date`  
* spending: “spent” → likely `order_items.price`

Candidate tables selected: `customers`, `orders`, `order_items`

---

### **Step 2 — Find join paths (join graph)**

From profiling / inferred keys:

* `orders.customer_id = customers.customer_id`  
* `order_items.order_id = orders.order_id`

Proposed join plan:  
\[  
customers \\bowtie\_{customer\_id} orders \\bowtie\_{order\_id} order\_items  
\]

---

### **Step 3 — Push down filters**

Extract constraints from question:

* `customers.city = 'Denver'`  
* `orders.order_date in [2025-01-01, 2025-02-01)`

Apply them early to avoid join explosion.

Filtered rows:

* customers in Denver: customer\_id ∈ {101, 103}  
* orders in Jan 2025 for those customers: order\_id ∈ {9001} (since 9002 is Feb)

---

### **Step 4 — Execute joins \+ compute candidate evidence/context tuples**

Join results (joined tuples) after filters:

Evidence tuple E1 (joined row, flattened):

* customer\_id=101, name=Sara, city=Denver  
* order\_id=9001, order\_date=2025-01-15  
* product=bike helmet, price=45

Evidence tuple E2:

* customer\_id=101, name=Sara, city=Denver  
* order\_id=9001, order\_date=2025-01-15  
* product=water bottle, price=12

(Notice customer 103 has no Jan orders, so no joined tuples appear.)

---

### **Step 5 — Rank/select tuples for context (token budget)**

System picks top evidence tuples that:

* match Denver \+ Jan 2025 \+ have price  
* cover all needed facts for the sum  
* avoid irrelevant columns (drop `product` if not needed)

So it might package only:

* (order\_id, customer\_id, city, order\_date, price)

---

### **Step 6 — Context packaging to LLM**

Example prompt context:

**Task:** Answer the question using only the evidence rows. If the evidence is insufficient, say so. Provide the numeric result and cite evidence IDs.

**Evidence rows:**

* E1: city=Denver, order\_date=2025-01-15, order\_id=9001, price=45  
* E2: city=Denver, order\_date=2025-01-15, order\_id=9001, price=12

**Question:** How much did customers in Denver spend in January 2025?

---

### **Step 7 — LLM answer (grounded \+ cited)**

Customers in Denver spent **$57** in January 2025 (45 \+ 12). **Evidence:** E1, E2.

---

**AI Acknowledgement**

ChatGPT was used to polish and enrich this document  
