import unittest

from decision_lora.finetune.cli import selftest


class FinetuneTests(unittest.TestCase):
    def test_split_merge_and_corpus(self):
        selftest()


if __name__ == "__main__":
    unittest.main()
