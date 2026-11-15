import unittest


class TestResultFormat(unittest.TestCase):
    def test_best_f_is_scientific_notation(self):
        from experiments.result_format import csv_rows

        rows = csv_rows([{"best_f": 54.0913963318}, {"best_f": None}])
        self.assertEqual(rows[0]["best_f"], "5.409140e+01")
        self.assertIsNone(rows[1]["best_f"])


if __name__ == "__main__":
    unittest.main()
