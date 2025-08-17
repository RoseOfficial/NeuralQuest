#!/usr/bin/env python3
"""Production readiness verification script."""

import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check that all core modules can be imported."""
    modules = [
        'redai',
        'redai.envs.pyboy_env',
        'redai.algo.a2c_numpy',
        'redai.nets.rnd_numpy',
        'redai.explore.archive',
        'redai.train.trainer',
        'redai.train.config'
    ]
    
    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"OK {module}")
        except ImportError as e:
            print(f"FAIL {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def check_files():
    """Check that all required files exist."""
    required_files = [
        'setup.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'configs/baseline.toml',
        'configs/smoke_test.toml',
        'scripts/train_pokemon.py',
        'scripts/eval_pokemon.py',
        'scripts/verify_production.py',
        'docs/PRD.txt'
    ]
    
    failed = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"OK {file_path}")
        else:
            print(f"MISSING {file_path}")
            failed.append(file_path)
    
    return len(failed) == 0

def check_gitignore():
    """Check that .gitignore covers important patterns."""
    required_patterns = [
        'checkpoints/',
        'logs/',
        '*.gb',
        '__pycache__/',
        '*.pyc',
        '.env',
        'debug_*.py'
    ]
    
    with open('.gitignore', 'r') as f:
        gitignore_content = f.read()
    
    failed = []
    for pattern in required_patterns:
        if pattern in gitignore_content:
            print(f"OK .gitignore covers: {pattern}")
        else:
            print(f"MISSING .gitignore missing: {pattern}")
            failed.append(pattern)
    
    return len(failed) == 0

def check_dependencies():
    """Check that dependencies can be imported."""
    deps = ['numpy', 'pyboy']
    
    failed = []
    for dep in deps:
        try:
            importlib.import_module(dep)
            print(f"OK {dep}")
        except ImportError:
            print(f"MISSING {dep} not installed")
            failed.append(dep)
    
    return len(failed) == 0

def main():
    """Run all production checks."""
    print("NeuralQuest Production Readiness Check")
    print("=" * 50)
    
    checks = [
        ("Core imports", check_imports),
        ("Required files", check_files),
        (".gitignore patterns", check_gitignore),
        ("Dependencies", check_dependencies)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        passed = check_func()
        all_passed = all_passed and passed
    
    print("\n" + "=" * 50)
    if all_passed:
        print("SUCCESS: Production ready! All checks passed.")
        print("\nNext steps:")
        print("1. git init")
        print("2. git add .")
        print("3. git commit -m 'Initial NeuralQuest implementation'")
        print("4. Add ROM files (not tracked)")
        print("5. Run training: python scripts/train_pokemon.py --mode smoke")
        return 0
    else:
        print("ERROR: Production issues found. Please fix before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())