#!/usr/bin/env python3
"""
MedConceptsQA Hierarchical Max-Coverage Sampling Script - V2 OPTIMIZED

This version implements tier-specific candidate pools for O(n + m*k) complexity
where k is the number of questions affected by coverage changes (k << n).

Key optimizations:
1. Build tier-specific pools at start (O(n))
2. Select from highest priority tier (O(1))
3. Incrementally update tier pools when coverage changes (O(k) where k << n)
4. No iteration through all questions after initial setup

Author: Generated via Claude Code, checked and verified by Sameed Khan
Date: October 22, 2025
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from pyhealth.medcode import InnerMap
from tqdm import tqdm

# ============================================================================
# CONSTANTS
# ============================================================================

CODE_EXTRACTION_PATTERN = r"medical code ([A-Z0-9.]+) in"
VALID_VOCABS = ["ICD9CM", "ICD10CM"]
VALID_DIFFICULTY_STRATEGIES = ["equal", "proportional"]

# Hierarchy level weights for greedy scoring
COVERAGE_WEIGHTS = {
    "chapter": 1000,
    "category": 100,
    "subcategory": 10,
    "full_code": 1,
}

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================


class Logger:
    """Simple logger with level support."""

    LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "QUIET": 4}

    def __init__(self, level: str = "INFO"):
        self.level = self.LEVELS.get(level.upper(), 1)

    def debug(self, msg: str):
        if self.level <= 0:
            print(f"[DEBUG] {msg}")

    def info(self, msg: str):
        if self.level <= 1:
            print(f"[INFO] {msg}")

    def warning(self, msg: str):
        if self.level <= 2:
            print(f"⚠ [WARNING] {msg}")

    def error(self, msg: str):
        if self.level <= 3:
            print(f"✗ [ERROR] {msg}", file=sys.stderr)

    def section(self, title: str):
        if self.level <= 1:
            print(f"\n{'=' * 80}")
            print(title)
            print("=" * 80)

    def subsection(self, title: str):
        if self.level <= 1:
            print(f"\n{title}")
            print("-" * 80)


# ============================================================================
# VALIDATION
# ============================================================================


def validate_vocab(vocab: str) -> Tuple[bool, str]:
    """Validate vocabulary parameter."""
    if vocab not in VALID_VOCABS:
        return (
            False,
            f"Invalid vocabulary '{vocab}'. Must be one of: {', '.join(VALID_VOCABS)}",
        )
    return True, ""


def validate_size_quota(size_quota: int) -> Tuple[bool, str]:
    """Validate size quota parameter."""
    if size_quota < 1:
        return False, f"size_quota must be >= 1, got {size_quota}"
    if size_quota < 100:
        return False, f"size_quota={size_quota} is too small. Minimum recommended: 100"
    return True, ""


def validate_difficulty_strategy(strategy: str) -> Tuple[bool, str]:
    """Validate difficulty split strategy parameter."""
    if strategy not in VALID_DIFFICULTY_STRATEGIES:
        return (
            False,
            f"Invalid difficulty strategy '{strategy}'. Must be one of: {', '.join(VALID_DIFFICULTY_STRATEGIES)}",
        )
    return True, ""


# ============================================================================
# HIERARCHY DECOMPOSITION
# ============================================================================


def decompose_icd10_code(code: str) -> Dict[str, Optional[str]]:
    """Decompose ICD-10 code into hierarchy levels."""
    clean_code = code.replace(".", "")
    return {
        "full_code": code,
        "chapter": clean_code[0] if len(clean_code) >= 1 else None,
        "category": clean_code[:3] if len(clean_code) >= 3 else None,
        "subcategory": clean_code[: min(6, len(clean_code))]
        if len(clean_code) > 3
        else None,
    }


def decompose_icd9_code(code: str) -> Dict[str, Optional[str]]:
    """Decompose ICD-9 code into hierarchy levels."""
    clean_code = code.replace(".", "")
    return {
        "full_code": code,
        "chapter": clean_code[:3] if len(clean_code) >= 3 else None,
        "category": clean_code[:3] if len(clean_code) >= 3 else None,
        "subcategory": clean_code[:4] if len(clean_code) >= 4 else None,
    }


# ============================================================================
# DATASET LOADING AND CODE EXTRACTION
# ============================================================================


def load_medconceptsqa(vocab: str, logger: Logger) -> Dict:
    """Load ICD configurations for specified vocabulary from MedConceptsQA."""
    logger.subsection("PHASE 1: LOADING DATASET")
    logger.info(f"Loading MedConceptsQA ({vocab}) from HuggingFace...")

    # Determine which configs to load based on vocabulary
    vocab_prefix = vocab.lower()
    configs = [f"{vocab_prefix}_easy", f"{vocab_prefix}_medium", f"{vocab_prefix}_hard"]

    datasets = {}
    for config in configs:
        try:
            data = load_dataset("ofir408/MedConceptsQA", config, split="test")
            datasets[config] = data
            logger.info(f"  ✓ {config} ({len(data):,} questions)")
        except Exception as e:
            logger.error(f"Failed to load {config}: {e}")
            raise

    total = sum(len(d) for d in datasets.values())
    logger.info(f"Total: {total:,} {vocab} questions loaded")

    return datasets


def extract_codes(
    datasets: Dict, vocab: str, logger: Logger
) -> Tuple[Dict[str, List[Dict]], int]:
    """Extract ICD codes from all questions and build question list."""
    logger.subsection("PHASE 2: CODE EXTRACTION")
    logger.info("Extracting ICD codes using regex pattern...")

    pattern = re.compile(CODE_EXTRACTION_PATTERN)
    questions_by_config = {}
    total_failures = 0

    # Select decomposition function based on vocabulary
    if vocab == "ICD9CM":
        decompose_fn = decompose_icd9_code
    else:
        decompose_fn = decompose_icd10_code

    for config_name, data in datasets.items():
        questions = []
        failures = 0
        difficulty = config_name.split("_")[1]

        # Extract codes
        for example in tqdm(
            data, desc=f"  {config_name}", disable=logger.level > 1, leave=False
        ):
            match = pattern.search(example["question"])
            if match:
                code = match.group(1)
                decomp = decompose_fn(code)

                questions.append(
                    {
                        "question_id": example["question_id"],
                        "code": code,
                        "vocab": vocab,
                        "difficulty": difficulty,
                        "chapter": decomp["chapter"],
                        "category": decomp["category"],
                        "subcategory": decomp["subcategory"],
                        "full_code": decomp["full_code"],
                        "original_data": example,
                    }
                )
            else:
                failures += 1

        questions_by_config[config_name] = questions
        total_failures += failures

    total_extracted = sum(len(q) for q in questions_by_config.values())
    success_rate = (
        total_extracted / (total_extracted + total_failures)
        if (total_extracted + total_failures) > 0
        else 0
    )

    logger.info(
        f"Successfully extracted {total_extracted:,} codes ({total_failures} failures, {success_rate:.1%} success rate)"
    )

    if total_failures > 0:
        logger.warning(
            f"{total_failures} codes could not be extracted (likely range codes like 'V83-V84.99')"
        )

    return questions_by_config, total_failures


# ============================================================================
# HIERARCHY MAPPING
# ============================================================================


def build_hierarchy_mappings(
    questions_by_config: Dict[str, List[Dict]], vocab: str, logger: Logger
) -> Dict:
    """Build hierarchy mappings and load PyHealth vocabulary."""
    logger.subsection("PHASE 3: HIERARCHY ANALYSIS")
    logger.info(f"Loading PyHealth vocabulary for {vocab}...")

    # Load PyHealth vocabulary
    try:
        vocab_map = InnerMap.load(vocab)
        logger.info(f"  ✓ {vocab} ({len(vocab_map.graph.nodes)} codes in hierarchy)")
    except Exception as e:
        logger.error(f"Failed to load PyHealth vocabulary: {e}")
        raise

    # Organize questions by difficulty
    questions_by_difficulty = {"easy": [], "medium": [], "hard": []}

    for config_name, questions in questions_by_config.items():
        difficulty = config_name.split("_")[1]
        questions_by_difficulty[difficulty].extend(questions)

    logger.info("Hierarchy analysis complete")

    return {
        "questions_by_difficulty": questions_by_difficulty,
        "vocabulary": vocab_map,
    }


# ============================================================================
# V2 OPTIMIZED GREEDY MAX-COVERAGE SAMPLING ALGORITHM
# ============================================================================


def build_reverse_indices(questions: List[Dict], logger: Logger) -> Dict:
    """
    Build reverse indices mapping hierarchy nodes to question indices.

    This enables O(1) lookup of which questions have a given chapter/category/subcategory.
    """
    logger.debug("Building reverse indices for fast lookup...")

    indices = {
        "chapter_to_questions": defaultdict(set),
        "category_to_questions": defaultdict(set),
        "subcategory_to_questions": defaultdict(set),
    }

    for idx, q in enumerate(questions):
        if q.get("chapter"):
            indices["chapter_to_questions"][q["chapter"]].add(idx)
        if q.get("category"):
            indices["category_to_questions"][q["category"]].add(idx)
        if q.get("subcategory"):
            indices["subcategory_to_questions"][q["subcategory"]].add(idx)

    logger.debug(
        f"  Indices built: {len(indices['chapter_to_questions'])} chapters, "
        f"{len(indices['category_to_questions'])} categories, "
        f"{len(indices['subcategory_to_questions'])} subcategories"
    )

    return indices


def build_initial_tier_pools(
    questions: List[Dict],
    indices: Dict,
    covered: Dict[str, Set],
    logger: Logger,
) -> Dict[int, Set[int]]:
    """
    Build initial tier pools based on current coverage state.

    Tier 1: Questions with uncovered chapters (highest priority)
    Tier 2: Questions with uncovered categories (but covered chapters)
    Tier 3: Questions with uncovered subcategories (but covered categories)
    Tier 4: All other questions

    Complexity: O(n)
    """
    logger.debug("Building initial tier pools...")

    tier_pools = {1: set(), 2: set(), 3: set(), 4: set()}

    for idx, q in enumerate(questions):
        chapter = q.get("chapter")
        category = q.get("category")
        subcategory = q.get("subcategory")

        # Skip if subcategory already covered
        if subcategory and subcategory in covered["subcategories"]:
            continue

        # Assign to tier
        if chapter and chapter not in covered["chapters"]:
            tier_pools[1].add(idx)
        elif category and category not in covered["categories"]:
            tier_pools[2].add(idx)
        elif subcategory and subcategory not in covered["subcategories"]:
            tier_pools[3].add(idx)
        else:
            tier_pools[4].add(idx)

    logger.debug(
        f"  Tier sizes: T1={len(tier_pools[1])}, T2={len(tier_pools[2])}, "
        f"T3={len(tier_pools[3])}, T4={len(tier_pools[4])}"
    )

    return tier_pools


def update_tier_pools_after_selection(
    selected_question: Dict,
    questions: List[Dict],
    indices: Dict,
    tier_pools: Dict[int, Set[int]],
    covered: Dict[str, Set],
    newly_covered: Dict[str, bool],
) -> int:
    """
    Incrementally update tier pools after a question is selected.

    Key optimization: Only update questions affected by newly covered nodes.

    Returns: number of questions moved between tiers (for stats)
    """
    moves_count = 0

    # Remove all questions with this subcategory from all tiers
    subcat = selected_question.get("subcategory")
    if subcat and subcat in indices["subcategory_to_questions"]:
        for idx in indices["subcategory_to_questions"][subcat]:
            for tier in [1, 2, 3, 4]:
                if idx in tier_pools[tier]:
                    tier_pools[tier].remove(idx)
                    moves_count += 1

    # If a new chapter was covered, move questions from Tier 1 to Tier 2
    if newly_covered.get("chapter"):
        chapter = selected_question.get("chapter")
        if chapter and chapter in indices["chapter_to_questions"]:
            for idx in list(indices["chapter_to_questions"][chapter]):
                if idx in tier_pools[1]:
                    tier_pools[1].remove(idx)
                    # Check if question's category is still uncovered
                    q = questions[idx]
                    category = q.get("category")
                    if category and category not in covered["categories"]:
                        # Still in tier 2 (uncovered category)
                        if q.get("subcategory") not in covered["subcategories"]:
                            tier_pools[2].add(idx)
                    elif q.get("subcategory") not in covered["subcategories"]:
                        tier_pools[3].add(idx)
                    moves_count += 1

    # If a new category was covered, move questions from Tier 2 to Tier 3
    if newly_covered.get("category"):
        category = selected_question.get("category")
        if category and category in indices["category_to_questions"]:
            for idx in list(indices["category_to_questions"][category]):
                if idx in tier_pools[2]:
                    tier_pools[2].remove(idx)
                    # Move to tier 3 if subcategory not covered
                    q = questions[idx]
                    if q.get("subcategory") not in covered["subcategories"]:
                        tier_pools[3].add(idx)
                    moves_count += 1

    return moves_count


def greedy_coverage_sampling_v2(
    questions: List[Dict],
    quota: int,
    seed: int,
    logger: Logger,
) -> Tuple[List[Dict], Dict]:
    """
    V2 OPTIMIZED hierarchical coverage sampling with tier-specific pools.

    Key optimizations:
    1. Build tier pools once at start - O(n)
    2. Select from highest priority tier - O(1)
    3. Incrementally update only affected questions - O(k) where k << n

    Complexity:
    - Preprocessing: O(n)
    - Per iteration: O(k) where k = questions affected by coverage change
    - Total: O(n + m*k_avg) where k_avg << n

    Expected speedup vs original O(n*m): 50-100x

    Returns: (sampled_questions, coverage_statistics)
    """
    random.seed(seed)

    # Track coverage
    covered = {
        "chapters": set(),
        "categories": set(),
        "subcategories": set(),
        "full_codes": set(),
    }

    # Build reverse indices (O(n))
    indices = build_reverse_indices(questions, logger)

    # Build initial tier pools (O(n))
    tier_pools = build_initial_tier_pools(questions, indices, covered, logger)

    # Sampled question indices
    sampled_indices = set()

    # Stats tracking
    total_tier_updates = 0
    tier_selections = {1: 0, 2: 0, 3: 0, 4: 0}

    # Progress bar
    pbar = tqdm(
        total=quota,
        desc="  V2 optimized sampling",
        disable=logger.level > 1,
        leave=False,
    )

    iteration_count = 0

    while len(sampled_indices) < quota:
        iteration_count += 1

        # Select from highest priority tier with candidates (O(1))
        selected_idx = None

        for tier in [1, 2, 3, 4]:
            if tier_pools[tier]:
                # Convert to list for random.choice (O(k) where k = tier size)
                candidates = list(tier_pools[tier])
                selected_idx = random.choice(candidates)
                tier_selections[tier] += 1
                break

        if selected_idx is None:
            logger.debug(f"No more valid questions at iteration {iteration_count}")
            break

        # Add to sampled
        selected = questions[selected_idx]
        sampled_indices.add(selected_idx)

        # Track what gets newly covered
        newly_covered = {
            "chapter": False,
            "category": False,
            "subcategory": False,
        }

        # Update covered sets
        if selected.get("chapter") and selected["chapter"] not in covered["chapters"]:
            covered["chapters"].add(selected["chapter"])
            newly_covered["chapter"] = True

        if (
            selected.get("category")
            and selected["category"] not in covered["categories"]
        ):
            covered["categories"].add(selected["category"])
            newly_covered["category"] = True

        if selected.get("subcategory"):
            covered["subcategories"].add(selected["subcategory"])
            newly_covered["subcategory"] = True

        if selected.get("full_code"):
            covered["full_codes"].add(selected["full_code"])

        # Incrementally update tier pools (O(k) where k << n)
        moves = update_tier_pools_after_selection(
            selected, questions, indices, tier_pools, covered, newly_covered
        )
        total_tier_updates += moves

        # Update progress
        pbar.update(1)

    pbar.close()

    # Calculate average tier pool sizes
    avg_pool_size = sum(len(pool) for pool in tier_pools.values()) / 4

    # Log performance statistics
    logger.debug("  V2 Optimization stats:")
    logger.debug(f"    Iterations: {iteration_count}")
    logger.debug(f"    Total tier updates: {total_tier_updates:,}")
    logger.debug(
        f"    Avg updates/iteration: {total_tier_updates / iteration_count:.1f}"
    )
    logger.debug(f"    Avg tier pool size: {avg_pool_size:.0f}")
    logger.debug(
        f"    Tier selections: T1={tier_selections[1]}, T2={tier_selections[2]}, T3={tier_selections[3]}, T4={tier_selections[4]}"
    )
    logger.debug(
        f"    Original would scan: {len(questions) * iteration_count:,} questions"
    )
    logger.debug(
        f"    Theoretical speedup: {(len(questions) * iteration_count) / max(total_tier_updates, 1):.1f}x"
    )

    # Build sampled questions list
    sampled = [questions[idx] for idx in sampled_indices]

    # Build statistics
    stats = {
        "sampled_count": len(sampled),
        "coverage": {
            "chapters": len(covered["chapters"]),
            "categories": len(covered["categories"]),
            "subcategories": len(covered["subcategories"]),
            "full_codes": len(covered["full_codes"]),
        },
        "optimization_stats": {
            "algorithm_version": "v2_tier_pools",
            "iterations": iteration_count,
            "total_tier_updates": total_tier_updates,
            "avg_updates_per_iteration": total_tier_updates / iteration_count
            if iteration_count > 0
            else 0,
            "tier_selections": tier_selections,
            "original_would_evaluate": len(questions) * iteration_count,
            "speedup_factor": (
                (len(questions) * iteration_count) / max(total_tier_updates, 1)
            ),
        },
    }

    return sampled, stats


# ============================================================================
# SAMPLING PLAN GENERATION
# ============================================================================


def calculate_difficulty_quotas(
    total_quota: int,
    questions_by_difficulty: Dict[str, List],
    strategy: str,
) -> Dict[str, int]:
    """Calculate how to distribute quota across difficulties."""
    if strategy == "equal":
        # Equal split across three difficulties
        base_quota = total_quota // 3
        remainder = total_quota % 3
        quotas = {"easy": base_quota, "medium": base_quota, "hard": base_quota}
        # Distribute remainder
        for i, diff in enumerate(["easy", "medium", "hard"]):
            if i < remainder:
                quotas[diff] += 1
    else:  # proportional
        total_questions = sum(len(qs) for qs in questions_by_difficulty.values())
        quotas = {}
        allocated = 0
        for diff in ["easy", "medium", "hard"]:
            proportion = len(questions_by_difficulty[diff]) / total_questions
            quota = int(total_quota * proportion)
            quotas[diff] = quota
            allocated += quota
        # Handle rounding remainder
        remainder = total_quota - allocated
        if remainder > 0:
            quotas["easy"] += remainder

    return quotas


def calculate_total_available_nodes(questions: List[Dict]) -> Dict[str, int]:
    """Calculate total unique nodes available in the question pool."""
    all_chapters = set()
    all_categories = set()
    all_subcategories = set()
    all_full_codes = set()

    for q in questions:
        if q["chapter"]:
            all_chapters.add(q["chapter"])
        if q["category"]:
            all_categories.add(q["category"])
        if q["subcategory"]:
            all_subcategories.add(q["subcategory"])
        if q["full_code"]:
            all_full_codes.add(q["full_code"])

    return {
        "chapters": len(all_chapters),
        "categories": len(all_categories),
        "subcategories": len(all_subcategories),
        "full_codes": len(all_full_codes),
    }


def generate_sampling_plan(
    hierarchy_data: Dict,
    vocab: str,
    size_quota: int,
    difficulty_strategy: str,
    seed: int,
    logger: Logger,
) -> Dict:
    """Generate complete sampling plan using V2 optimized greedy max-coverage algorithm."""
    logger.subsection("PHASE 4: SAMPLING PLAN GENERATION")
    logger.info("Generating sampling plan with V2 optimization...")

    questions_by_difficulty = hierarchy_data["questions_by_difficulty"]
    hierarchy_data["vocabulary"]

    # Calculate difficulty quotas
    difficulty_quotas = calculate_difficulty_quotas(
        size_quota, questions_by_difficulty, difficulty_strategy
    )

    logger.info(f"Difficulty quotas ({difficulty_strategy} strategy):")
    for diff, quota in difficulty_quotas.items():
        logger.info(f"  • {diff}: {quota:,}")

    # Calculate total available nodes for reference
    all_questions = []
    for qs in questions_by_difficulty.values():
        all_questions.extend(qs)
    total_available = calculate_total_available_nodes(all_questions)

    logger.info(f"\nTotal available nodes in {vocab}:")
    logger.info(f"  • Chapters: {total_available['chapters']}")
    logger.info(f"  • Categories: {total_available['categories']}")
    logger.info(f"  • Subcategories: {total_available['subcategories']}")
    logger.info(f"  • Full codes: {total_available['full_codes']}")

    # Run V2 optimized greedy sampling for each difficulty
    sampling_results = {}
    total_sampled = 0

    for difficulty in ["easy", "medium", "hard"]:
        logger.info(
            f"\nSampling {difficulty} (quota: {difficulty_quotas[difficulty]:,})..."
        )

        sampled, stats = greedy_coverage_sampling_v2(
            questions=questions_by_difficulty[difficulty],
            quota=difficulty_quotas[difficulty],
            seed=seed + hash(difficulty),  # Different seed per difficulty
            logger=logger,
        )

        sampling_results[difficulty] = {
            "sampled_questions": sampled,
            "stats": stats,
            "quota": difficulty_quotas[difficulty],
        }

        total_sampled += len(sampled)
        logger.info(f"  ✓ Sampled {len(sampled):,} questions")
        logger.info(
            f"    Coverage: {stats['coverage']['chapters']} chapters, "
            f"{stats['coverage']['categories']} categories, "
            f"{stats['coverage']['subcategories']} subcategories"
        )
        logger.info(
            f"    Speedup: {stats['optimization_stats']['speedup_factor']:.1f}x "
            f"({stats['optimization_stats']['total_tier_updates']:,} updates "
            f"vs {stats['optimization_stats']['original_would_evaluate']:,} in original)"
        )

    logger.info(f"\n✓ Plan generated for {total_sampled:,} total samples")

    # Calculate aggregate coverage across all difficulties
    aggregate_chapters = set()
    aggregate_categories = set()
    aggregate_subcategories = set()
    aggregate_full_codes = set()

    for difficulty in ["easy", "medium", "hard"]:
        sampled_questions = sampling_results[difficulty]["sampled_questions"]
        for q in sampled_questions:
            if q.get("chapter"):
                aggregate_chapters.add(q["chapter"])
            if q.get("category"):
                aggregate_categories.add(q["category"])
            if q.get("subcategory"):
                aggregate_subcategories.add(q["subcategory"])
            if q.get("full_code"):
                aggregate_full_codes.add(q["full_code"])

    aggregate_coverage = {
        "chapters": len(aggregate_chapters),
        "categories": len(aggregate_categories),
        "subcategories": len(aggregate_subcategories),
        "full_codes": len(aggregate_full_codes),
    }

    logger.info("\nAggregate coverage across all difficulties:")
    logger.info(
        f"  • Chapters: {aggregate_coverage['chapters']} / {total_available['chapters']} "
        f"({aggregate_coverage['chapters'] / total_available['chapters'] * 100:.1f}%)"
    )
    logger.info(
        f"  • Categories: {aggregate_coverage['categories']} / {total_available['categories']} "
        f"({aggregate_coverage['categories'] / total_available['categories'] * 100:.1f}%)"
    )
    logger.info(
        f"  • Subcategories: {aggregate_coverage['subcategories']} / {total_available['subcategories']} "
        f"({aggregate_coverage['subcategories'] / total_available['subcategories'] * 100:.1f}%)"
    )
    logger.info(
        f"  • Full codes: {aggregate_coverage['full_codes']} / {total_available['full_codes']} "
        f"({aggregate_coverage['full_codes'] / total_available['full_codes'] * 100:.1f}%)"
    )

    # Build plan dictionary
    plan = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "vocab": vocab,
            "size_quota": size_quota,
            "difficulty_strategy": difficulty_strategy,
            "total_sampled": total_sampled,
            "seed": seed,
            "algorithm_version": "v2_tier_pools",
        },
        "difficulty_quotas": difficulty_quotas,
        "total_available": total_available,
        "aggregate_coverage": aggregate_coverage,
        "sampling_results": {
            diff: {
                "quota": res["quota"],
                "sampled_count": res["stats"]["sampled_count"],
                "coverage": res["stats"]["coverage"],
                "optimization_stats": res["stats"]["optimization_stats"],
                "question_ids": [q["question_id"] for q in res["sampled_questions"]],
            }
            for diff, res in sampling_results.items()
        },
    }

    return plan, sampling_results


# ============================================================================
# OUTPUT GENERATION (same as original)
# ============================================================================


def save_sampling_plan(plan: Dict, output_path: Path, logger: Logger):
    """Save sampling plan as JSON."""
    logger.info(f"Saving sampling plan to {output_path}...")

    try:
        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2, default=str)
        logger.info(
            f"✓ Saved sampling plan ({output_path.stat().st_size / 1024:.1f} KB)"
        )
    except Exception as e:
        logger.error(f"Failed to save sampling plan: {e}")
        raise


def create_dataset_from_plan(
    plan: Dict,
    sampling_results: Dict,
    vocab: str,
    vocab_map: InnerMap,
    logger: Logger,
) -> DatasetDict:
    """Create HuggingFace DatasetDict from sampling results (original MedConceptsQA columns only)."""
    logger.subsection("PHASE 5: DATASET CREATION")
    logger.info("Creating HuggingFace dataset from sampling plan...")

    dataset_splits = {}

    for difficulty in ["easy", "medium", "hard"]:
        config_name = f"{vocab.lower()}_{difficulty}"
        sampled_questions = sampling_results[difficulty]["sampled_questions"]

        # Extract only original MedConceptsQA data (no hierarchy columns)
        data_list = []
        for q in sampled_questions:
            data_item = dict(q["original_data"])  # Copy only original fields
            data_list.append(data_item)

        # Create HuggingFace Dataset
        dataset_splits[config_name] = Dataset.from_list(data_list)
        logger.info(f"  ✓ {config_name}: {len(data_list):,} samples")

    dataset_dict = DatasetDict(dataset_splits)

    total_samples = sum(len(split) for split in dataset_dict.values())
    logger.info(f"✓ Created dataset with {total_samples:,} total samples")

    return dataset_dict


def save_report(
    plan: Dict,
    output_path: Path,
    runtime: float,
    logger: Logger,
):
    """Save text report with coverage statistics and optimization metrics."""
    logger.info(f"Generating text report at {output_path}...")

    metadata = plan["metadata"]
    total_available = plan["total_available"]
    sampling_results = plan["sampling_results"]

    report = []
    report.append("=" * 80)
    report.append("MEDCONCEPTSQA MAX-COVERAGE SAMPLING REPORT (V2 OPTIMIZED)")
    report.append("=" * 80)
    report.append(f"\nGenerated: {metadata['created_at']}")
    report.append(f"Runtime: {runtime:.1f} seconds")

    report.append("\n\nCONFIGURATION")
    report.append("-" * 80)
    report.append(
        f"Algorithm Version    : {metadata.get('algorithm_version', 'original')}"
    )
    report.append(f"Vocabulary           : {metadata['vocab']}")
    report.append(f"Size Quota           : {metadata['size_quota']:,}")
    report.append(f"Difficulty Strategy  : {metadata['difficulty_strategy']}")
    report.append(f"Total Sampled        : {metadata['total_sampled']:,}")
    report.append(f"Random Seed          : {metadata['seed']}")

    report.append("\n\nOPTIMIZATION STATISTICS (V2 TIER POOLS)")
    report.append("-" * 80)
    for diff in ["easy", "medium", "hard"]:
        opt_stats = sampling_results[diff].get("optimization_stats", {})
        if opt_stats:
            speedup = opt_stats.get("speedup_factor", 1.0)
            report.append(f"\n{diff.upper()}:")
            report.append(
                f"  Iterations               : {opt_stats.get('iterations', 'N/A'):,}"
            )
            report.append(
                f"  Total tier updates       : {opt_stats.get('total_tier_updates', 'N/A'):,}"
            )
            report.append(
                f"  Avg updates/iteration    : {opt_stats.get('avg_updates_per_iteration', 0):.1f}"
            )
            report.append(
                f"  Original would evaluate  : {opt_stats.get('original_would_evaluate', 'N/A'):,}"
            )
            report.append(f"  Speedup factor           : {speedup:.1f}x")

    report.append("\n\nDIFFICULTY QUOTAS")
    report.append("-" * 80)
    for diff in ["easy", "medium", "hard"]:
        quota = plan["difficulty_quotas"][diff]
        sampled = sampling_results[diff]["sampled_count"]
        report.append(
            f"{diff.upper():8} : {sampled:,} / {quota:,} ({sampled / quota * 100:.1f}%)"
        )

    report.append("\n\nCOVERAGE STATISTICS")
    report.append("=" * 80)
    report.append(f"\nTotal samples: {metadata['total_sampled']:,}")
    report.append(f"\n{metadata['vocab']}:")

    report.append("\n  Available nodes in dataset:")
    report.append(f"    Chapters     : {total_available['chapters']}")
    report.append(f"    Categories   : {total_available['categories']}")
    report.append(f"    Subcategories: {total_available['subcategories']}")
    report.append(f"    Full codes   : {total_available['full_codes']}")

    # Aggregate coverage across all difficulties
    aggregate_coverage = plan.get("aggregate_coverage", {})
    if aggregate_coverage:
        report.append("\n  AGGREGATE COVERAGE (across all difficulties):")
        report.append(
            f"    Chapters     : {aggregate_coverage['chapters']} / {total_available['chapters']} "
            f"({aggregate_coverage['chapters'] / total_available['chapters'] * 100:.1f}%)"
        )
        report.append(
            f"    Categories   : {aggregate_coverage['categories']} / {total_available['categories']} "
            f"({aggregate_coverage['categories'] / total_available['categories'] * 100:.1f}%)"
        )
        report.append(
            f"    Subcategories: {aggregate_coverage['subcategories']} / {total_available['subcategories']} "
            f"({aggregate_coverage['subcategories'] / total_available['subcategories'] * 100:.1f}%)"
        )
        report.append(
            f"    Full codes   : {aggregate_coverage['full_codes']} / {total_available['full_codes']} "
            f"({aggregate_coverage['full_codes'] / total_available['full_codes'] * 100:.1f}%)"
        )

    report.append("\n  Coverage by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        coverage = sampling_results[diff]["coverage"]
        report.append(f"\n  {diff.upper()}:")
        report.append(
            f"    Chapters     : {coverage['chapters']} / {total_available['chapters']} "
            f"({coverage['chapters'] / total_available['chapters'] * 100:.1f}%)"
        )
        report.append(
            f"    Categories   : {coverage['categories']} / {total_available['categories']} "
            f"({coverage['categories'] / total_available['categories'] * 100:.1f}%)"
        )
        report.append(
            f"    Subcategories: {coverage['subcategories']} / {total_available['subcategories']} "
            f"({coverage['subcategories'] / total_available['subcategories'] * 100:.1f}%)"
        )
        report.append(
            f"    Full codes   : {coverage['full_codes']} / {total_available['full_codes']} "
            f"({coverage['full_codes'] / total_available['full_codes'] * 100:.1f}%)"
        )

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    with open(output_path, "w") as f:
        f.write("\n".join(report))

    logger.info(f"✓ Saved report ({output_path.stat().st_size / 1024:.1f} KB)")


# ============================================================================
# MAIN
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Max-coverage hierarchical sampling of MedConceptsQA dataset (V2 OPTIMIZED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 15,000 total samples from ICD-10CM with equal difficulty split
  python sample_medconceptsqa_v2_optimized.py --size-quota 15000 --vocab ICD10CM

  # Sample 6,000 samples from ICD-9CM with proportional difficulty split
  python sample_medconceptsqa_v2_optimized.py --size-quota 6000 --vocab ICD9CM --difficulty-split-strategy proportional

  # Generate plan only without creating dataset
  python sample_medconceptsqa_v2_optimized.py --size-quota 10000 --plan-only

  # Custom output name and seed
  python sample_medconceptsqa_v2_optimized.py --size-quota 12000 --output-name my_dataset --seed 123

V2 OPTIMIZATIONS (Tier-Specific Pools):
  - Build tier pools once at start: O(n)
  - Select from highest priority tier: O(1)
  - Update only affected questions: O(k) where k << n
  - Expected speedup: 50-100x faster than original implementation
        """,
    )

    parser.add_argument(
        "--size-quota",
        type=int,
        required=True,
        help="Total number of samples to generate (distributed across 3 difficulties)",
    )

    parser.add_argument(
        "--vocab",
        type=str,
        choices=VALID_VOCABS,
        default="ICD10CM",
        help="Vocabulary to sample from (default: ICD10CM)",
    )

    parser.add_argument(
        "--difficulty-split-strategy",
        type=str,
        choices=VALID_DIFFICULTY_STRATEGIES,
        default="equal",
        help="How to split quota across difficulties: 'equal' (default) or 'proportional'",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="medconceptsqa_sample_v2",
        help="Output dataset name (default: medconceptsqa_sample_v2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "QUIET"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Generate sampling plan only, do not create dataset",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    start_time = datetime.now()

    # Initialize logger
    logger = Logger(args.log_level)

    logger.section("MEDCONCEPTSQA MAX-COVERAGE SAMPLING (V2 OPTIMIZED)")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Display configuration
    logger.subsection("CONFIGURATION")
    logger.info("Algorithm Version     : V2 (tier-specific pools)")
    logger.info(f"Vocabulary            : {args.vocab}")
    logger.info(f"Size Quota            : {args.size_quota:,}")
    logger.info(f"Difficulty Strategy   : {args.difficulty_split_strategy}")
    logger.info(f"Output Name           : {args.output_name}")
    logger.info(f"Random Seed           : {args.seed}")
    logger.info(f"Plan Only Mode        : {args.plan_only}")

    try:
        # Validate parameters
        valid, msg = validate_vocab(args.vocab)
        if not valid:
            logger.error(msg)
            return 1

        valid, msg = validate_size_quota(args.size_quota)
        if not valid:
            logger.error(msg)
            return 1

        valid, msg = validate_difficulty_strategy(args.difficulty_split_strategy)
        if not valid:
            logger.error(msg)
            return 1

        # Load datasets
        datasets = load_medconceptsqa(args.vocab, logger)
        questions_by_config, failures = extract_codes(datasets, args.vocab, logger)

        # Build hierarchy mappings
        hierarchy_data = build_hierarchy_mappings(
            questions_by_config, args.vocab, logger
        )

        # Generate sampling plan
        plan, sampling_results = generate_sampling_plan(
            hierarchy_data=hierarchy_data,
            vocab=args.vocab,
            size_quota=args.size_quota,
            difficulty_strategy=args.difficulty_split_strategy,
            seed=args.seed,
            logger=logger,
        )

        dynamic_repo_path = Path(__file__).parent / Path(
            f"medconceptsqa-sample_{args.output_name}"
        )
        dynamic_repo_path.mkdir(exist_ok=True)

        # Save plan
        plan_path = dynamic_repo_path / Path(f"{args.output_name}_plan.json")
        save_sampling_plan(plan, plan_path, logger)

        # Create dataset (unless plan-only mode)
        if not args.plan_only:
            dataset = create_dataset_from_plan(
                plan=plan,
                sampling_results=sampling_results,
                vocab=args.vocab,
                vocab_map=hierarchy_data["vocabulary"],
                logger=logger,
            )

            # Save dataset
            logger.info(f"Saving dataset to {args.output_name}/...")
            dataset_path = dynamic_repo_path / Path(args.output_name)
            dataset.save_to_disk(dataset_path)
            logger.info("✓ Dataset saved")

            # Save report
            runtime = (datetime.now() - start_time).total_seconds()
            report_path = dynamic_repo_path / Path(f"{args.output_name}_report.txt")
            save_report(plan, report_path, runtime, logger)

        # Final summary
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        logger.section("COMPLETE")
        logger.info(f"Runtime: {runtime:.1f} seconds")

        if args.plan_only:
            logger.info(f"Sampling plan saved to out/{args.output_name}_plan.json")
        else:
            logger.info(f"Dataset saved to {dynamic_repo_path}/")
            logger.info(
                f"Report saved to {dynamic_repo_path}/{args.output_name}_report.txt"
            )

        return 0

    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
