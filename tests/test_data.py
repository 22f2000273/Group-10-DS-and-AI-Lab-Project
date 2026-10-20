import pathlib
import unittest


class TestDataStructure(unittest.TestCase):
    def test_data_readme_exists(self):
        self.assertTrue(pathlib.Path("data/README.md").is_file())


if __name__ == "__main__":
    unittest.main()
