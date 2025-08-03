# Contributing to Transistor Parameter Extraction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the deep learning transistor parameter extraction codebase.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature/fix
4. Set up the development environment:
   ```bash
   conda env create -f environment.yml
   conda activate transistor-param-extraction
   pip install -e .[dev]
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use black for code formatting: `black src/`
- Use flake8 for linting: `flake8 src/`
- Add docstrings to all functions and classes

### Testing
- Write tests for new functionality
- Run tests before submitting: `pytest tests/`
- Ensure all tests pass
- Aim for good test coverage

### Documentation
- Update documentation for any new features
- Include examples in docstrings
- Update README.md if necessary

### Commit Messages
- Use clear, descriptive commit messages
- Follow conventional commit format when possible
- Reference issue numbers in commits

## Submitting Changes

1. Ensure all tests pass
2. Update documentation as needed
3. Create a pull request with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/plots if applicable

## Development Setup

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
flake8 src/ tests/
```

### Building Documentation
```bash
cd docs/
sphinx-build -b html . _build/html
```

## Questions?

If you have questions about contributing, please:
1. Check existing issues
2. Create a new issue for discussion
3. Contact the maintainers at rkabenne@stanford.edu

Thank you for contributing!