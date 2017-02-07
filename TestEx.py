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

    def test_gamut_initialize(self):
        # Test for convex hull
        n_data = np.array([[0, 0, 0],       # 0 vertecis
                           [10, 0, 0],      # 1 vertecis
                           [10, 10, 0],     # 2 vertecis
                           [0, 10, 0],      # 3 vertecis
                           [5, 5, 5],       # 4 non vertecis
                           [4, 6, 2],       # 5 non vertecis
                           [10, 10, 10],    # 6 vertecis
                           [1, 2, 3],       # 7 non vertecis
                           [10, 0, 10],     # 8 vertecis
                           [0, 0, 10],      # 9 vertecis
                           [0, 10, 10]])    # 10 vertecis

        c_data = data.Data(space.srgb, n_data)          # Generating the colour Data object
        g = gamut.Gamut(space.srgb, c_data)
        vertices = np.array([0, 1, 2, 3, 6, 8, 9, 10])  # Known indices of vertecis for the test case

        self.assertEqual(vertices.tolist(), g.vertices.tolist())    # Checking that the vertecis match

    def test_is_inside(self):       #Test for gamut.Gamut.is_inside

        n_data = np.array([[0, 0, 0],  # 0 vertecis
                           [10, 0, 0],  # 1 vertecis
                           [10, 10, 0],  # 2 vertecis
                           [0, 10, 0],  # 3 vertecis
                           [5, 5, 5],  # 4 non vertecis
                           [4, 6, 2],  # 5 non vertecis
                           [10, 10, 10],  # 6 vertecis
                           [1, 2, 3],  # 7 non vertecis
                           [10, 0, 10],  # 8 vertecis
                           [0, 0, 10],  # 9 vertecis
                           [0, 10, 10]])  # 10 vertecis
        c_data = data.Data(space.srgb, n_data)
        g = gamut.Gamut(space.srgb, c_data)

        points = np.array([[1, 1, 1],   # inside
                           [2, 2, 3],   # inside
                           [20, 2, 3],  # outside
                           [1, 2, 30]]) # outside
        c_points = data.Data(space.srgb, points)
        g.is_inside(space.srgb, c_points)


    #def test_get_vertices(self):


if __name__ == '__main__':
    unittest.main(exit=False)
