from ExFunction import retur, my_contains, my_first
import unittest, numpy as np
from colour import data, gamut, space


"""Imports the functions from the exfuntion.py file and uses it to run unittest on them and test
    each funtions invidualiy.
"""


class TestFunction(unittest.TestCase):
    def test_return(self):
        self.assertEquals(retur(), "true")

    def test_contains(self):
        self.assertTrue(my_contains(3, [1, 2, 3]))

    def test_first_number(self):
        self.assertEquals(my_first([1, 2, 3]), 1)

if __name__ == '__main__':
    unittest.main(exit=False)
