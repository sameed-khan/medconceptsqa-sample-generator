# MedConceptsQA Max-Coverage Sampling Script

## Overview

`sample_medconceptsqa.py` performs **maximum-coverage hierarchical sampling** 
of the MedConceptsQA dataset. 
What this means is that the sampling is done in a hierarchy-stratified manner
where the sampler tries to optimize for the maximum coverage across all levels
of the ICD-9 or ICD-10 hierarchy for any sample size that the user passes in.

### Key Features

- **Greedy Coverage Algorithm**: Prioritizes covering unique codes at all hierarchy levels
- **No Duplicate Subcategories**: Ensures maximum diversity by never sampling the same subcategory twice
- **Hierarchy-Aware**: Optimizes coverage across chapters → categories → subcategories → full codes
- **Single Vocabulary Focus**: Sample from either ICD-9-CM or ICD-10-CM (default: ICD-10-CM)
- **Dev/Test Splits**: Includes dev splits from original MedConceptsQA for validation
- **Multiple Configs**: Creates 4 subsets (easy, medium, hard, all) mirroring MedConceptsQA structure
- **Hub-Ready**: Auto-generates upload script for easy publishing to HuggingFace Hub
- **Reproducible**: Seed-based sampling for identical results

## Quick Start

Generate a 15,000-sample dataset from ICD-10-CM:
```bash
# Sample 15,000 questions from ICD-10-CM
uv run python sample_medconceptsqa.py \
  --size-quota 15000 \
  --vocab ICD10CM \
  --output-name medconceptsqa_15k \
  --seed 42
```

This creates a folder `medconceptsqa-sample_medconceptsqa_15k/` containing:
- **4 dataset configs** (subsets): `icd10cm_easy/`, `icd10cm_medium/`, `icd10cm_hard/`, `all/`
- **Each config has 2 splits**: `dev` (4-12 examples) and `test` (sampled examples)
- `medconceptsqa_15k_plan.json` - Detailed sampling plan with coverage statistics
- `medconceptsqa_15k_report.txt` - Human-readable coverage report
- `upload_to_hub.py` - Auto-generated script for uploading to HuggingFace Hub

## Command-Line Arguments

### Required Arguments

**`--size-quota INT`**
- Total number of samples to generate across all difficulties
- Distributed equally by default (e.g., 15,000 → 5,000 easy, 5,000 medium, 5,000 hard)
- Example: `--size-quota 15000` creates 15,000 total samples

### Optional Arguments

**`--vocab {ICD9CM,ICD10CM}`**
- Which vocabulary to sample from
- **Default: ICD10CM** (recommended - most current coding system)
- **ICD10CM**: Modern, detailed hierarchy
  - 26 chapters, 1,914 categories, 46,380 subcategories, 95,513 full codes
  - Used in current medical coding (post-2015)
- **ICD9CM**: Legacy coding system
  - 1,041 categories, 6,406 subcategories, 17,552 full codes
  - Used for historical/public health research

**`--difficulty-split-strategy {equal,proportional}`**
- How to distribute the size quota across easy/medium/hard difficulties
- **Default: equal** (recommended for balanced datasets)
  - Splits quota evenly: 15,000 → 5,000 each
- **proportional**: Splits based on original dataset proportions
  - ICD10: ~36% easy, 31% medium, 33% hard
  - Example: 15,000 → 5,400 easy, 4,650 medium, 4,950 hard

**`--output-name STR`**
- Name for output files (default: `medconceptsqa_sample`)
- Creates: `{output_name}/`, `{output_name}_plan.json`, `{output_name}_report.txt`

**`--seed INT`**
- Random seed for reproducibility (default: 42)
- Use the same seed to generate identical samples

**`--log-level {DEBUG,INFO,WARNING,ERROR,QUIET}`**
- Logging verbosity (default: INFO)
- **DEBUG**: Show detailed diagnostic information
- **INFO**: Standard progress updates with coverage statistics
- **WARNING**: Only show warnings and errors
- **ERROR**: Only show errors
- **QUIET**: Minimal output (only final summary)

**`--plan-only`**
- Generate sampling plan JSON without creating the dataset
- Useful for reviewing coverage statistics before execution
- Creates only `{output_name}_plan.json`

## How the Greedy Max-Coverage Algorithm Works

The script uses a greedy algorithm to maximize unique code coverage across the hierarchy. 
Here's a detailed walkthrough:

### Algorithm Overview

```mermaid
flowchart TD
    A[Start: All questions available] --> B[Calculate coverage score for each question]
    B --> C{Filter: Exclude questions with<br/>already-sampled subcategories}
    C --> D[Score each remaining question:<br/>chapter=1000, category=100<br/>subcategory=10, full_code=1]
    D --> E{Multiple questions<br/>with max score?}
    E -->|Yes| F[Randomly select one from<br/>top-scoring questions]
    E -->|No| G[Select the top scorer]
    F --> H[Add question to sample]
    G --> H
    H --> I[Mark covered:<br/>chapter, category<br/>subcategory, full_code]
    I --> J{Quota<br/>reached?}
    J -->|No| B
    J -->|Yes| K[End: Return sampled questions]
```

### Step-by-Step Example

Let's walk through sampling with a quota of 15,000 from ICD-10-CM:

**1. Quota Distribution** (Equal strategy)
```
Total quota: 15,000
├── Easy: 5,000
├── Medium: 5,000
└── Hard: 5,000
```

**2. First Iteration (Easy difficulty)**

Available questions pool: 94,576 questions

The algorithm examines each question and calculates a **coverage score**:

| Question | Code | Chapter | Category | Subcategory | Full Code | Score Calculation |
|----------|------|---------|----------|-------------|-----------|-------------------|
| Q1 | S02.0XXA | S | S02 | S02.0 | S02.0XXA | 1000+100+10+1 = **1111** |
| Q2 | S02.1XXA | S | S02 | S02.1 | S02.1XXA | 0+0+10+1 = **11** (S and S02 already covered) |
| Q3 | I10 | I | I10 | None | I10 | 1000+100+0+1 = **1101** |

**Selection**: Q1 is selected (highest score: 1111)
- **Covered after iteration 1**: 1 chapter (S), 1 category (S02), 1 subcategory (S02.0), 1 full code

**3. Second Iteration**

| Question | Code | Chapter | Category | Subcategory | Full Code | Score |
|----------|------|---------|----------|-------------|-----------|-------|
| Q2 | S02.1XXA | S | S02 | S02.1 | S02.1XXA | 0+0+10+1 = **11** |
| Q3 | I10 | I | I10 | None | I10 | 1000+100+0+1 = **1101** |
| Q4 | M79.3 | M | M79 | M79.3 | M79.3 | 1000+100+10+1 = **1111** |

**Selection**: Q4 is selected (score 1111, new chapter)
- **Covered after iteration 2**: 2 chapters, 2 categories, 2 subcategories, 2 full codes

**4. Continue Until Quota Reached...**

After 5,000 iterations for "easy":
- **All 26 chapters covered** (100%)
- **Typically ~5,000 categories covered** (depending on quota and hierarchy distribution)
- **Exactly 5,000 unique subcategories covered** (no duplicates!)
- **5,000 unique full codes**

### Coverage Progression

As sampling progresses, the algorithm naturally prioritizes:

1. **Phase 1 (Early)**: Covers all chapters quickly (first ~26 samples)
2. **Phase 2 (Middle)**: Covers many categories (first ~1,000-2,000 samples)
3. **Phase 3 (Late)**: Maximizes subcategory coverage (remaining quota)

```
Samples    Chapters  Categories  Subcategories  Full Codes
     100         26         100            100         100
   1,000         26       1,000          1,000       1,000
   5,000         26       ~4,500          5,000       5,000
  10,000         26       ~9,000         10,000      10,000
```
## Usage Examples

### Example 1: Recommended Configuration for RL Training

```bash
uv run python sample_medconceptsqa.py \
  --size-quota 15000 \
  --vocab ICD10CM \
  --difficulty-split-strategy equal \
  --output-name rl_training_15k \
  --seed 42
```

**Result:**
- 4 configs: `icd10cm_easy`, `icd10cm_medium`, `icd10cm_hard`, `all`
- 15,000 total test samples (5,000 per difficulty) + 12 dev samples
- Maximum subcategory diversity across the hierarchy
- All 26 ICD-10-CM chapters covered
- ~25% of all subcategories covered
- Ready to upload to HuggingFace Hub with generated script

### Example 2: Small Sample for Testing

```bash
uv run python sample_medconceptsqa.py \
  --size-quota 300 \
  --vocab ICD10CM \
  --output-name test_300 \
  --seed 42
```

**Result:**
- 4 configs with 300 total test samples (100 per difficulty) + 12 dev samples
- Quick test dataset for validating pipelines

### Example 3: ICD-9-CM Legacy Dataset

```bash
uv run python sample_medconceptsqa.py \
  --size-quota 6000 \
  --vocab ICD9CM \
  --output-name icd9_legacy_6k \
  --seed 42
```

**Result:**
- 4 configs: `icd9cm_easy`, `icd9cm_medium`, `icd9cm_hard`, `all`
- 6,000 total test samples from ICD-9-CM + 12 dev samples
- Higher coverage percentage due to smaller ICD-9 hierarchy

### Example 4: Proportional Difficulty Split

```bash
uv run python sample_medconceptsqa.py \
  --size-quota 12000 \
  --vocab ICD10CM \
  --difficulty-split-strategy proportional \
  --output-name proportional_12k \
  --seed 123
```

**Result:**
- ~4,320 easy, ~3,720 medium, ~3,960 hard
- Matches original MedConceptsQA difficulty distribution

### Example 5: Plan-Only Mode (Review Coverage First)

```bash
# Generate plan to see coverage statistics
uv run python sample_medconceptsqa.py \
  --size-quota 15000 \
  --vocab ICD10CM \
  --plan-only \
  --output-name my_sample

# Review the coverage statistics
cat my_sample_plan.json

# If satisfied, run again without --plan-only
uv run python sample_medconceptsqa.py \
  --size-quota 15000 \
  --vocab ICD10CM \
  --output-name my_sample
```

### Example 6: Quiet Mode for Automation

```bash
uv run python sample_medconceptsqa.py \
  --size-quota 10000 \
  --log-level QUIET \
  --output-name automated_10k
```

Only shows final summary, no progress bars.

## Output Files

### 1. Dataset Configs (`medconceptsqa-sample_{output_name}/`)

The script creates 4 dataset configurations (subsets), each in its own folder:

```
medconceptsqa-sample_medconceptsqa_15k/
├── icd10cm_easy/           # Easy difficulty questions
│   ├── dev/                # 4 dev examples
│   └── test/               # 5,000 sampled test examples
├── icd10cm_medium/         # Medium difficulty questions
│   ├── dev/                # 4 dev examples
│   └── test/               # 5,000 sampled test examples
├── icd10cm_hard/           # Hard difficulty questions
│   ├── dev/                # 4 dev examples
│   └── test/               # 5,000 sampled test examples
├── all/                    # Combined all difficulties
│   ├── dev/                # 12 dev examples (4+4+4)
│   └── test/               # 15,000 sampled test examples
├── medconceptsqa_15k_plan.json
├── medconceptsqa_15k_report.txt
└── upload_to_hub.py
```

Each config folder contains a HuggingFace `DatasetDict` with `dev` and `test` splits:

```python
from datasets import load_from_disk

# Load a specific config
dataset = load_from_disk("medconceptsqa-sample_medconceptsqa_15k/icd10cm_easy")
print(dataset)
# DatasetDict({
#     dev: Dataset({
#         features: ['question_id', 'question', 'answer', 'answer_id',
#                    'option1', 'option2', 'option3', 'option4',
#                    'vocab', 'level'],
#         num_rows: 4
#     }),
#     test: Dataset({
#         features: [...same as above...],
#         num_rows: 5000
#     })
# })

# Access a specific example from the test split
example = dataset["test"][0]
print(example["question"])
# "What is the description of the medical code S02.0XXA in ICD-10-CM?"

print(example["answer"])
# "Fracture of vault of skull, initial encounter for closed fracture"

# Load the combined "all" config
all_dataset = load_from_disk("medconceptsqa-sample_medconceptsqa_15k/all")
print(f"Dev examples: {len(all_dataset['dev'])}")    # 12
print(f"Test examples: {len(all_dataset['test'])}")  # 15,000
```

### Dataset Columns

The sampled dataset contains the same columns as the original MedConceptsQA dataset:

| Column | Description | Example Value |
|--------|-------------|---------------|
| `question_id` | Unique identifier for the question | 12345 |
| `question` | The question text | "What is the description of the medical code S02.0XXA in ICD-10-CM?" |
| `answer` | The correct answer | "Fracture of vault of skull, initial encounter for closed fracture" |
| `answer_id` | Index of the correct answer (0-3) | 2 |
| `option1` | First answer option | "..." |
| `option2` | Second answer option | "..." |
| `option3` | Third answer option | "..." |
| `option4` | Fourth answer option | "..." |
| `vocab` | Vocabulary system | "ICD-10-CM" |
| `level` | Difficulty level | "easy" |

**Notes:**
- The sampling algorithm uses ICD hierarchy codes internally (chapter, category, subcategory, full_code) to maximize coverage, but these hierarchy columns are **not** included in the final dataset output. Only the original MedConceptsQA columns are preserved.
- **Dev splits** are sourced from the original MedConceptsQA dataset's `dev` split, filtered for the specified vocabulary (e.g., ICD10CM). Each difficulty config contains 4 dev examples, and the `all` config contains 12 dev examples (concatenated from all three difficulties).
- **Test splits** contain the hierarchically-sampled questions generated by the max-coverage algorithm.

### 2. Sampling Plan JSON (`{output_name}_plan.json`)

Complete record of sampling with coverage statistics:

```json
{
  "metadata": {
    "created_at": "2025-10-22T20:45:03",
    "vocab": "ICD10CM",
    "size_quota": 15000,
    "difficulty_strategy": "equal",
    "total_sampled": 15000,
    "seed": 42
  },
  "difficulty_quotas": {
    "easy": 5000,
    "medium": 5000,
    "hard": 5000
  },
  "total_available": {
    "chapters": 26,
    "categories": 1914,
    "subcategories": 46380,
    "full_codes": 95513
  },
  "sampling_results": {
    "easy": {
      "quota": 5000,
      "sampled_count": 5000,
      "coverage": {
        "chapters": 26,
        "categories": 1891,
        "subcategories": 5000,
        "full_codes": 5000
      },
      "question_ids": [...]
    },
    ...
  }
}
```

**Key Fields:**
- `total_available`: How many unique codes exist in the dataset
- `coverage`: How many unique codes were sampled
- Calculate coverage percentage: `(coverage / total_available) × 100`

### 3. Text Report (`{output_name}_report.txt`)

Human-readable summary:

```
================================================================================
MEDCONCEPTSQA MAX-COVERAGE SAMPLING REPORT
================================================================================

Generated: 2025-10-22T20:45:03
Runtime: 77.2 seconds

CONFIGURATION
--------------------------------------------------------------------------------
Vocabulary           : ICD10CM
Size Quota           : 15000
Difficulty Strategy  : equal
Total Sampled        : 15000
Random Seed          : 42

DIFFICULTY QUOTAS
--------------------------------------------------------------------------------
EASY     : 5,000 / 5,000 (100.0%)
MEDIUM   : 5,000 / 5,000 (100.0%)
HARD     : 5,000 / 5,000 (100.0%)

COVERAGE STATISTICS
================================================================================

Total samples: 15,000

ICD10CM:

  Available nodes in dataset:
    Chapters     : 26
    Categories   : 1914
    Subcategories: 46380
    Full codes   : 95513

  ALL (combined across all difficulties):
    Chapters     : 26 / 26 (100.0%)
    Categories   : 1914 / 1914 (100.0%)
    Subcategories: 15000 / 46380 (32.8%)
    Full codes   : 10122 / 95513 (10.6%)

  Coverage by difficulty:

  EASY:
    Chapters     : 26 / 26 (100.0%)
    Categories   : 1891 / 1914 (98.8%)
    Subcategories: 5000 / 46380 (10.8%)
    Full codes   : 5000 / 95513 (5.2%)

  MEDIUM:
    Chapters     : 25 / 26 (96.2%)
    Categories   : 1856 / 1914 (97.0%)
    Subcategories: 5000 / 46380 (10.8%)
    Full codes   : 5000 / 95513 (5.2%)

  HARD:
    Chapters     : 25 / 26 (96.2%)
    Categories   : 1872 / 1914 (97.8%)
    Subcategories: 5000 / 46380 (10.8%)
    Full codes   : 5000 / 95513 (5.2%)

================================================================================
```

### 4. Upload Script (`upload_to_hub.py`)

The script automatically generates a Python upload script for publishing to HuggingFace Hub:

```bash
# Navigate to the output directory
cd medconceptsqa-sample_medconceptsqa_15k

# Set your HuggingFace token
export HF_TOKEN='your_token_here'

# Upload to HuggingFace Hub
python upload_to_hub.py --repo-name username/dataset-name

# Optional: make the repository private
python upload_to_hub.py --repo-name username/dataset-name --private
```

The upload script will:
1. Upload all 4 configs (`icd10cm_easy`, `icd10cm_medium`, `icd10cm_hard`, `all`)
2. Upload both `dev` and `test` splits for each config
3. Show progress for each config/split combination
4. Provide the final dataset URL

**Example output:**
```
Uploading medconceptsqa_15k to username/dataset-name...
Private: False

Loading icd10cm_easy...
  Uploading icd10cm_easy/dev (4 examples)...
    ✓ Successfully uploaded icd10cm_easy/dev
  Uploading icd10cm_easy/test (5000 examples)...
    ✓ Successfully uploaded icd10cm_easy/test

Loading icd10cm_medium...
  Uploading icd10cm_medium/dev (4 examples)...
    ✓ Successfully uploaded icd10cm_medium/dev
  Uploading icd10cm_medium/test (5000 examples)...
    ✓ Successfully uploaded icd10cm_medium/test

...

================================================================================
Upload complete!
View your dataset at: https://huggingface.co/datasets/username/dataset-name
================================================================================
```

**Notes:**
- The upload script requires `datasets` and `huggingface_hub` packages. These are already installed if you used `uv` to run the sampling script.
- The generated dataset structure mirrors the original MedConceptsQA repository, making it compatible with existing code and workflows.
- After uploading, users can load your dataset using:
  ```python
  from datasets import load_dataset

  # Load a specific config and split
  easy_test = load_dataset("username/dataset-name", "icd10cm_easy", split="test")

  # Load all splits of a config
  easy_data = load_dataset("username/dataset-name", "icd10cm_easy")

  # Load the combined "all" config
  all_data = load_dataset("username/dataset-name", "all")
  ```

## Understanding ICD Hierarchy

The ICD (International Classification of Diseases) organizes medical diagnoses hierarchically:

### ICD-10-CM Example: "Fracture of occiput"

```
Level           Code        Description
────────────────────────────────────────────────────────────────
Chapter         S           Injury, poisoning and certain other
                            consequences of external causes

Category        S02         Fracture of skull and facial bones

Subcategory     S02.0       Fracture of vault of skull

Full Code       S02.0XXA    Fracture of vault of skull, initial
                            encounter for closed fracture
```
**Not included:** Most granular laterality/anatomy (S02.0XXB, S02.0XXD, etc.)

### ICD-9-CM Example: "Fracture of tibia and fibula"

```
Level           Code        Description
────────────────────────────────────────────────────────────────
Chapter/        823         Fracture of tibia and fibula
Category

Subcategory     823.2       Fracture of shaft of tibia and fibula

Full Code       823.21      Fracture of shaft of tibia and fibula,
                            open
```

**Note:** ICD-9 has simpler structure (no separate chapter level)

## Citation

If you use this sampling script in your research, please cite MedConceptsQA:

```bibtex
@article{medconceptsqa2024,
  title={MedConceptsQA: Open Source Medical Concepts QA Benchmark},
  author={Pal, Ofir Ben Shoham and Hollander, Nadav},
  journal={Computers in Biology and Medicine},
  year={2024},
  publisher={Elsevier},
  url={https://www.sciencedirect.com/science/article/pii/S0010482524011740}
}
```
## License

This sampling script is provided for research purposes. 
MedConceptsQA dataset has its own license - please refer to the official dataset page.
