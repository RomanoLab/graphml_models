Each model is contained in its own subfolder, where the subfolder's name indicates the general characteristics of the model.

In the future, we may apply additional hierarchical structure based on groups of related models.

# Model structure:

Within each folder containing a model, there should be:

- `main.py`: A main Python script that can be used to test the model, e.g., on a benchmark dataset.
- `model.py`: The actual model, provided as a Python class. This should contain the structure of the model (e.g., any DGL GNN components, etc.) and can optionally contain other stuff that the model needs to run.
- [optional] Other Python files: E.g., utilities for loading data, reporting results, transforming inputs/outputs. These things can also be rolled into `model.py`.
- `requirements.txt`: pip-formatted requirements.txt file listing any packages needed to run the model. For the correct format, see here: https://pip.pypa.io/en/stable/reference/requirements-file-format/