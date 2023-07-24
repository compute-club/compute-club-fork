### Env setup

1. install (poetry)[https://python-poetry.org/docs/] - this manages python packages
2. `$ poetry config virtualenvs.in-project true`
3. `$ poetry install`
4. `$ poetry shell`
5. Download the lid.176.bin model from from the (fasttext)[https://fasttext.cc/docs/en/language-identification.html#content] and place it in /filtering/models

**To run notebook in VSCode**

1. From VSCODE hit `CMD+SHIFT+P` -> "Python: Select Interpreter"
2. Add this path: "<PATH_TO_REPO>/compute-club-fork/filtering/.venv/bin/python3.11"
3. When you open a notebook, click "select kernel" on the top right, then select the .venv kernel

### To add a new package

`poetry add <package_name>` rather than `pip install`

### Usage instructions

Check `demo.ipynb` for usage instructions
