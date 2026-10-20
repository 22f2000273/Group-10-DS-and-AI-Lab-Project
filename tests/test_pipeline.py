import pathlib
import unittest


class TestPipelineStructure(unittest.TestCase):
    def test_core_structure_exists(self):
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
            self.assertTrue(pathlib.Path(dir_path).is_dir(), f"Missing directory: {dir_path}")


if __name__ == "__main__":
    unittest.main()
