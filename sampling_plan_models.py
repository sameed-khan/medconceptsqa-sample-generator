"""
Pydantic models for MedConceptsQA sampling plans.
Generated from Investigation 5.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class StratumInfo(BaseModel):
    """Information about a single stratum."""

    code: str = Field(description="The ICD code representing this stratum")
    description: Optional[str] = Field(
        None, description="Human-readable description of the code"
    )
    available_questions: int = Field(
        description="Total questions available in this stratum"
    )
    phase1_sample_count: int = Field(
        description="Questions sampled in Phase 1 (minimum coverage)"
    )
    phase2_sample_count: int = Field(
        description="Questions sampled in Phase 2 (proportional)"
    )
    question_ids: List[int] = Field(description="List of sampled question IDs")


class DifficultyPlan(BaseModel):
    """Sampling plan for one difficulty level."""

    quota: int = Field(description="Total quota for this difficulty")
    phase1_samples: int = Field(description="Total samples from Phase 1")
    phase2_samples: int = Field(description="Total samples from Phase 2")
    strata: Dict[str, StratumInfo] = Field(description="Stratum-level sampling info")


class VocabPlan(BaseModel):
    """Sampling plan for one vocabulary."""

    easy: DifficultyPlan
    medium: DifficultyPlan
    hard: DifficultyPlan


class SamplingMetadata(BaseModel):
    """Metadata about the sampling plan."""

    created_at: datetime = Field(default_factory=datetime.now)
    hierarchy_level: str = Field(description="Hierarchy level for stratification")
    min_per_stratum: int = Field(description="Minimum samples per stratum")
    size_quota_per_vocab: int = Field(description="Total quota per vocabulary")
    total_dataset_size: int = Field(description="Total size of sampled dataset")
    notes: Optional[str] = None


class Statistics(BaseModel):
    """Statistics about the sampling plan."""

    total_strata: int
    sparse_strata: List[str] = Field(
        description="Codes of strata with < min_per_stratum"
    )
    coverage: Dict[str, str] = Field(
        description="Coverage percentages at each hierarchy level"
    )


class VocabStatistics(BaseModel):
    """Statistics for one vocabulary."""

    ICD9CM: Statistics
    ICD10CM: Statistics


class SamplingPlan(BaseModel):
    """Complete sampling plan."""

    metadata: SamplingMetadata
    sampling_plan: Dict[str, VocabPlan] = Field(
        description="Plans for ICD9CM and ICD10CM"
    )
    statistics: VocabStatistics
    warnings: List[str] = Field(default_factory=list)
