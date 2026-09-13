"""Multi-model trust orchestration.

Route one prompt to two or more models, detect where they agree and disagree,
attach a verifiable source to every claim, and validate the structured output
before returning it.
"""

from .models import (
    AgreementVerdict,
    Citation,
    Claim,
    ClaimAssessment,
    ConsensusReport,
    ModelCandidate,
)
from .orchestrator import route

__all__ = [
    "AgreementVerdict",
    "Citation",
    "Claim",
    "ClaimAssessment",
    "ConsensusReport",
    "ModelCandidate",
    "route",
]

__version__ = "0.1.0"