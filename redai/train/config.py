"""Configuration management for NeuralQuest training."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass
class EnvConfig:
    """Environment configuration."""
    frame_skip: int = 4
    sticky_p: float = 0.1
    seed: int = 1337
    max_episode_steps: int = 1500  # Shorter episodes to enable frontier sampling
    deterministic: bool = True
    headless: bool = True


@dataclass
class AlgoConfig:
    """Algorithm configuration."""
    gamma: float = 0.995
    gae_lambda: float = 0.95
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    batch_horizon: int = 1500  # Match episode length for proper episode completion
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    grad_clip: float = 5.0
    epochs_per_update: int = 4
    minibatch_size: int = 256
    hidden_dim: int = 256


@dataclass
class RNDConfig:
    """Random Network Distillation configuration."""
    beta: float = 0.2
    lr: float = 1e-3
    reward_clip: float = 5.0
    norm_ema: float = 0.99
    hidden_dim: int = 128


@dataclass
class ArchiveConfig:
    """Archive exploration configuration."""
    hash_bits: int = 64
    capacity: int = 20000
    p_frontier: float = 0.25
    novel_lru: int = 5000
    evict_on: str = "old_and_often"
    hamming_threshold: int = 2
    projection_dim: int = 64


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 10000
    ckpt_every: int = 100
    log_every: int = 10
    eval_every: int = 500
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    """Main configuration container."""
    env: EnvConfig
    algo: AlgoConfig
    rnd: RNDConfig
    archive: ArchiveConfig
    train: TrainConfig
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls(
            env=EnvConfig(),
            algo=AlgoConfig(),
            rnd=RNDConfig(),
            archive=ArchiveConfig(),
            train=TrainConfig()
        )
    
    @classmethod
    def from_toml(cls, path: str) -> "Config":
        """Load configuration from TOML file."""
        try:
            import tomllib
        except ImportError:
            # Fallback for Python < 3.11
            import tomli as tomllib
        
        with open(path, "rb") as f:
            data = tomllib.load(f)
        
        return cls(
            env=EnvConfig(**data.get("env", {})),
            algo=AlgoConfig(**data.get("algo", {})),
            rnd=RNDConfig(**data.get("rnd", {})),
            archive=ArchiveConfig(**data.get("archive", {})),
            train=TrainConfig(**data.get("train", {}))
        )
    
    def override(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides."""
        for key, value in overrides.items():
            if "." in key:
                section, param = key.split(".", 1)
                if hasattr(self, section):
                    config_section = getattr(self, section)
                    if hasattr(config_section, param):
                        setattr(config_section, param, value)
                    else:
                        raise ValueError(f"Unknown parameter: {param} in section {section}")
                else:
                    raise ValueError(f"Unknown section: {section}")
            else:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown parameter: {key}")
    
    def ensure_dirs(self) -> None:
        """Ensure output directories exist."""
        os.makedirs(self.train.save_dir, exist_ok=True)
        os.makedirs(self.train.log_dir, exist_ok=True)