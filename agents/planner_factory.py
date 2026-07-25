"""Small, UI-independent planner factory for fully offline modes."""

from agents.cleaning_agent import CleaningPlanner, RuleBasedCleaningAgent
from agents.multi_expert import MultiExpertCleaningAgent
from models.data_contract import DataContract
from models.policy import PreprocessingPolicy


DETERMINISTIC_ROUTED_MODE = "Deterministic routed planner"
OFFLINE_PLANNER_MODES = ("Rule-based baseline", DETERMINISTIC_ROUTED_MODE)


def build_offline_planner(
    planner_mode: str,
    policy: PreprocessingPolicy | None = None,
    contract: DataContract | None = None,
) -> CleaningPlanner:
    """Create a deterministic planner without network or Streamlit state."""
    from tools.policy import policy_with_contract
    policy = policy_with_contract(policy, contract)
    if planner_mode == "Rule-based baseline":
        return RuleBasedCleaningAgent(policy=policy)
    if planner_mode == DETERMINISTIC_ROUTED_MODE:
        return MultiExpertCleaningAgent(policy=policy, contract=contract)
    raise ValueError(f"Planner mode {planner_mode!r} is not an offline planner.")
