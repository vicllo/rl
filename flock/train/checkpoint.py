"""Save and load trained policy checkpoints.

Stores policy weights via equinox serialization, plus metadata (configs,
hyperparams, timestamp) as a JSON sidecar.

Checkpoints live under saves/checkpoints/{name}_{YYYYMMDD_HHMM}/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import equinox as eqx
import jax

from flock.env.types import EnvConfig, TeamConfig
from flock.train.ppo.policy import ActorCritic, make_policy

CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "saves" / "checkpoints"


def _make_dir(name: str) -> Path:
    """Create a timestamped checkpoint directory that never collides."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    d = CHECKPOINT_ROOT / f"{name}_{stamp}"
    # In the unlikely case of a collision within the same minute, append a suffix
    if d.exists():
        i = 2
        while (CHECKPOINT_ROOT / f"{name}_{stamp}_{i}").exists():
            i += 1
        d = CHECKPOINT_ROOT / f"{name}_{stamp}_{i}"
    d.mkdir(parents=True)
    return d


def save_checkpoint(
    name: str,
    policy: ActorCritic,
    team: TeamConfig,
    env_config: EnvConfig | None = None,
    metadata: dict | None = None,
) -> Path:
    """Save a trained policy checkpoint.

    >>> path = save_checkpoint("selfplay", pred_policy, rules.teams[0])

    Creates saves/checkpoints/{name}_{timestamp}/policy.eqx + meta.json.
    Returns the directory path.
    """
    d = _make_dir(name)

    eqx.tree_serialise_leaves(str(d / "policy.eqx"), policy)

    # Store everything needed to reconstruct the policy from scratch
    meta = {
        "name": name,
        "team": team._asdict(),
        "hidden": policy.trunk.out_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if env_config is not None:
        meta["env_config"] = env_config._asdict()
    if metadata is not None:
        meta.update(metadata)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved checkpoint: {d}")
    return d


def load_checkpoint(path: str | Path) -> tuple[ActorCritic, dict]:
    """Load a policy from a checkpoint directory.

    All architecture params are read from the saved metadata — no arguments needed.

    >>> policy, meta = load_checkpoint("selfplay_20260316_1430")
    """
    d = Path(path)
    if not d.is_absolute():
        d = CHECKPOINT_ROOT / d
    meta = json.loads((d / "meta.json").read_text())
    team = TeamConfig(**meta["team"])
    hidden = meta["hidden"]

    skeleton = make_policy(team, hidden=hidden, key=jax.random.key(0))
    policy = eqx.tree_deserialise_leaves(str(d / "policy.eqx"), skeleton)

    return policy, meta


def list_checkpoints(name: str | None = None) -> list[Path]:
    """List saved checkpoints, optionally filtered by name prefix. Newest first."""
    if not CHECKPOINT_ROOT.exists():
        return []
    dirs = sorted(CHECKPOINT_ROOT.iterdir(), reverse=True)
    if name is not None:
        dirs = [d for d in dirs if d.name.startswith(name + "_")]
    return [d for d in dirs if d.is_dir()]


def load_latest(name: str) -> tuple[ActorCritic, dict]:
    """Load the most recent checkpoint matching a name.

    >>> policy, meta = load_latest("selfplay")
    """
    matches = list_checkpoints(name)
    if not matches:
        raise FileNotFoundError(f"No checkpoints found for '{name}'")
    return load_checkpoint(matches[0])
