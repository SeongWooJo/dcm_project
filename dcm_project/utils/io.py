import os

def save_variable_to_file(variable: str, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(variable)

def load_variable_from_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
