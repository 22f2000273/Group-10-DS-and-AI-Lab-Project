import pathlib


def test_data_readme_exists():
    assert pathlib.Path("data/README.md").is_file()
