import pathlib


def test_model_config_exists():
    assert pathlib.Path("configs/model.yaml").is_file()
