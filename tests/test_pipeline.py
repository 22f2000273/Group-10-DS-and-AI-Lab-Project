import pathlib


def test_core_structure_exists():
    required_dirs = [
        "src/data",
        "src/models",
        "src/inference",
        "src/retrieval",
        "src/agents",
        "src/evaluation",
        "src/utils",
        "app",
        "outputs/predictions",
    ]
    for dir_path in required_dirs:
        assert pathlib.Path(dir_path).is_dir(), f"Missing directory: {dir_path}"
