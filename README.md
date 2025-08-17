# NeuralQuest

**Domain-agnostic reinforcement learning agent for Game Boy games using curiosity-driven exploration.**

NeuralQuest is a research project that implements a pure curiosity-driven RL agent capable of learning to play Game Boy games without any game-specific knowledge, hardcoded behaviors, or external rewards. The agent uses Random Network Distillation (RND) for intrinsic motivation and archive-based exploration for systematic state discovery.

## 🎯 Core Principles

- **Domain-Agnostic Learning**: No hardcoded game logic, waypoints, or ROM-specific features
- **Pure Curiosity**: All learning driven by intrinsic motivation (RND) with optional terminal rewards
- **Archive-Based Exploration**: Systematic discovery and revisiting of novel states
- **Minimal Dependencies**: NumPy-only neural networks, no PyTorch/TensorFlow required
- **Reproducible**: Fully deterministic execution with configurable seeds

## 🏗️ Architecture

### Core Components

- **Environment**: PyBoy Game Boy emulator wrapper with RAM-based observations
- **Policy**: Actor-Critic (A2C) with Generalized Advantage Estimation (GAE)
- **Curiosity**: Random Network Distillation for intrinsic rewards
- **Exploration**: SimHash-based state archive with frontier sampling
- **Networks**: Pure NumPy implementation with custom backpropagation

### Key Features

- **9-Action Discrete Control**: noop, up, down, left, right, A, B, start, select
- **RAM Observations**: Frame-stacked emulator RAM (no pixel processing)
- **Archive System**: 64-bit SimHash for state discretization and frontier-based resets
- **Persistent Learning**: Save/load networks and archive for continued exploration

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/neuralquest.git
cd neuralquest

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```bash
# Train agent on a Game Boy ROM (you must legally own the ROM)
python -m redai.cli.run_train path/to/your/game.gb --config configs/baseline.toml

# Resume training from checkpoint
python -m redai.cli.run_train path/to/your/game.gb --resume checkpoints/epoch_001000

# Evaluate trained agent
python -m redai.cli.eval path/to/your/game.gb --checkpoint checkpoints/epoch_001000 --episodes 50

# Quick smoke test (5 minutes)
python -m redai.cli.run_train path/to/your/game.gb --smoke-test
```

### Pokemon Red Specific Usage

```bash
# Train on Pokemon Red with RPG-optimized settings
python scripts/train_pokemon.py --mode standard

# Fast training for quick results
python scripts/train_pokemon.py --mode fast

# Maximum exploration to discover the entire game world
python scripts/train_pokemon.py --mode exploration

# Evaluate and watch the agent play
python scripts/eval_pokemon.py --mode standard --render --episodes 5

# Analyze exploration patterns and frontier cells
python scripts/eval_pokemon.py --mode standard --analyze-archive --frontier-analysis
```

### Configuration

Customize training via TOML configuration files:

```toml
[env]
frame_skip = 4
sticky_p = 0.1
seed = 1337

[algo]
gamma = 0.995
lr_policy = 3e-4
batch_horizon = 2048

[rnd]
beta = 0.2
lr = 1e-3
reward_clip = 5.0

[archive]
capacity = 20000
p_frontier = 0.25
novel_lru = 5000
```

Override any parameter from command line:
```bash
python -m redai.cli.run_train game.gb --override rnd.beta=0.3 --override algo.lr_policy=1e-3
```

## 📊 Monitoring & Evaluation

### Training Metrics

Monitor training progress via CSV logs:
- **Exploration**: Unique cells discovered per hour, archive growth
- **Performance**: Episode length distribution, policy entropy
- **Learning**: Policy/value losses, gradient norms, intrinsic rewards

### Success Criteria

The agent demonstrates domain-agnostic learning through:
1. **Increasing exploration rate**: More unique states discovered over time
2. **Improving survival**: Longer episodes before game over
3. **Persistent improvement**: Performance maintained after checkpoint resume
4. **Ablation validation**: Removing RND/archive collapses exploration

## 🧪 Testing

Run comprehensive test suite:

```bash
# All tests
python -m pytest redai/tests/

# Specific test categories
python -m pytest redai/tests/test_gradcheck.py    # Neural network gradients
python -m pytest redai/tests/test_archive.py     # Archive and exploration
python -m pytest redai/tests/test_rnd.py         # Random Network Distillation

# Manual gradient checking
python -m redai.tests.test_gradcheck

# Integration smoke test
python -m redai.cli.run_train path/to/rom.gb --smoke-test
```

## 🚀 Production Setup

### Repository Structure

The repository is production-ready with:

```
NeuralQuest/
├── redai/              # Core package
│   ├── algo/           # A2C algorithm
│   ├── envs/           # PyBoy environment
│   ├── explore/        # Archive & hashing
│   ├── nets/           # NumPy networks
│   ├── tests/          # Test suite
│   └── train/          # Training infrastructure
├── configs/            # TOML configurations
├── scripts/            # Training & evaluation scripts
├── roms/              # Game ROMs (gitignored)
├── requirements.txt   # Dependencies
├── setup.py          # Package installation
└── .gitignore        # Comprehensive exclusions
```

### .gitignore Coverage

The `.gitignore` excludes:
- **Training artifacts**: checkpoints, logs, save files
- **ROMs**: Game Boy files (*.gb, *.gbc, *.gba)
- **Python cache**: __pycache__, *.pyc, build artifacts
- **Development**: debug files, local configs, temp files
- **System**: OS-specific files (Windows, macOS, Linux)
- **Data**: Large model files, datasets, media files

### Git Best Practices

```bash
# Initialize repository
git init
git add .
git commit -m "Initial NeuralQuest implementation

🎮 Domain-agnostic RL agent for Game Boy games
🧠 RND curiosity + archive-based exploration  
🔬 Pure NumPy neural networks
✅ Comprehensive test suite"

# Add ROM files locally (not tracked)
cp your_game.gb roms/pokemon_red.gb

# Train with version control
git checkout -b experiment/pokemon-training
python scripts/train_pokemon.py --mode standard
git add configs/ -A
git commit -m "Add Pokemon Red training configuration"
```

### Deployment Considerations

1. **ROM Management**: ROMs are gitignored - distribute separately
2. **Checkpoint Storage**: Training artifacts excluded from VCS
3. **Configuration**: Use TOML files for reproducible experiments
4. **Dependencies**: Minimal requirements (NumPy, PyBoy, tomli)
5. **Testing**: Run full test suite before deployment

## 🔬 Research Features

### Random Network Distillation (RND)
- Fixed random target network and trainable predictor
- Prediction error serves as intrinsic reward for novel states
- Reward normalization via exponential moving averages

### Archive-Based Exploration
- SimHash (random projection + sign) for state discretization
- Frontier scoring: `α/(1+visits) + γ*age + ζ*depth`
- LRU novelty detection with Hamming distance thresholds
- Automatic capacity management with intelligent eviction

### Technical Implementation
- **NumPy-Only Networks**: Custom MLP with Adam optimizer and gradient clipping
- **Deterministic Execution**: Reproducible results across platforms when enabled
- **Efficient Archive**: Compressed savestates with hash-based collision detection
- **Performance Optimized**: >30k environment frames/minute target

## 📈 Performance Targets

- **Throughput**: ≥30,000 environment frames/minute (headless mode)
- **Exploration**: >100 unique cells/hour sustained discovery rate
- **Memory**: Archive capacity up to 20,000 states with compression
- **Determinism**: Identical results from identical seeds and configurations

## ⚖️ Legal & Ethics

- **ROM Ownership**: Users must legally own Game Boy ROM files
- **Research Use**: Single-player research applications only
- **No Distribution**: No ROM files or ROM-specific data included
- **Privacy**: Logs contain only emulator states and learned parameters

## 🛠️ Development

### Project Structure
```
redai/
├── envs/          # PyBoy environment wrapper
├── nets/          # NumPy neural networks (MLP, RND)  
├── algo/          # A2C algorithm implementation
├── explore/       # Archive system and hashing
├── train/         # Training loop and configuration
├── cli/           # Command-line interface
└── tests/         # Comprehensive test suite
```

### Contributing

1. Install development dependencies: `pip install -e .[dev]`
2. Run tests: `pytest`
3. Format code: `black redai/`
4. Type checking: `mypy redai/`
5. Linting: `flake8 redai/`

## 🔮 Roadmap

- **v1.0**: RAM-only observations, A2C+RND, archive exploration (current)
- **v1.1**: Optional PyTorch backend for acceleration
- **v2.0**: Pixel observations with convolutional networks
- **v3.0**: Hierarchical exploration with option-critic
- **v4.0**: Model-based planning with learned dynamics

## 📚 References

- [RND Paper](https://arxiv.org/abs/1810.12894): Random Network Distillation
- [Go-Explore](https://arxiv.org/abs/1901.10995): Archive-based exploration
- [PyBoy](https://github.com/Baekalfen/PyBoy): Game Boy emulator
- [GAE Paper](https://arxiv.org/abs/1506.02438): Generalized Advantage Estimation

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built with PyBoy emulator and inspired by OpenAI's curiosity-driven exploration research.