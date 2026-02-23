"""Custom reward function for SLIME RL training.

The reward is already computed inside slime_generate.py during the
agent-environment loop (deterministic scoring from env.py). This module
provides the reward_func interface that SLIME expects, simply returning
the pre-computed reward and subscores.
"""

import logging

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


async def reward_func(args, sample, **kwargs):
    """Return the reward already computed during generation.

    SLIME calls this after generate() — since our environment scoring is
    deterministic and already runs inside generate(), we just read it back.

    Returns:
        dict with "score" and optional "subscores"
    """
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample)}")

    reward = sample.reward if sample.reward is not None else 0.0
    subscores = {}
    if sample.metadata:
        subscores = sample.metadata.get("subscores", {})

    return {
        "score": reward,
        "subscores": subscores,
    }
