# Contributing to VectorWaves

Thank you for contributing to VectorWaves!

## Setting Up for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/M-Soni-Physics/VectorWaves.git
   cd VectorWaves
   ```

2. **Install in editable mode with testing tools:**
   ```bash
   # Core + tests
   pip install -e ".[test]"

   # Or include extras if testing GPU/viz:
   pip install -e ".[test,viz,gpu]"
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b new-branch
   ```
2. Make your changes and add corresponding unit tests in `tests/`.
3. Run the test suite:
   ```bash
   pytest
   ```
4. Push your branch and open a Pull Request on GitHub.