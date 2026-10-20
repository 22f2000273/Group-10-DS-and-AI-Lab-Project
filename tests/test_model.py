import pathlib
import unittest


class TestModelStructure(unittest.TestCase):
    def test_model_config_exists(self):
        self.assertTrue(pathlib.Path("configs/model.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
