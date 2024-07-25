import unittest
import numpy as np
from src.hex.hexgrid import HexGrid
import os

out_folder = "./out"

class HexGridTests(unittest.TestCase):
    def setUp(self):
        self.grid = HexGrid(radius_grid_hx=4, width_tile_px=10, orientation_hx="FLAT", padding_grid=5, origin=np.array([0, 0, 0]), padding_tile=2)

    def test_initialize(self):
        self.assertEqual(self.grid.radius_grid_hx, 4)
        self.assertEqual(self.grid.width_tile_px, 10)
        self.assertEqual(self.grid.orientation_hx, "FLAT")
        self.assertEqual(self.grid.padding_grid, 5)
        self.assertTrue(np.array_equal(self.grid.origin, np.array([0, 0, 0])))
        self.assertEqual(self.grid.padding_tile, 2)
        self.assertEqual(len(self.grid), 61)  # Number of hexes in a grid with radius 4
        
        # The formula for number of hexes is 3 * radius^2 + 3 * radius + 1

    def test_set_and_get_hex(self):
        self.grid[0] = "A"
        self.assertEqual(self.grid[0], "A")

    def test_set_and_get_hex_by_ax(self):
        self.grid.set_by_ax((0, 0), "B")
        self.assertEqual(self.grid.val_by_ax((0, 0)), "B")

    def test_set_and_get_hex_by_cube(self):
        self.grid.set_by_cube((0, 0, 0), "C")
        self.assertEqual(self.grid.val_by_cube((0, 0, 0)), "C")

    def test_set_and_get_hex_by_px(self):
        self.grid.set_by_px((0, 0), "D")
        self.assertEqual(self.grid.val_by_px((0, 0)), "D")

    def test_update_tile_radius(self):
        self.grid.update_tile_radius(8)
        self.assertEqual(self.grid.radius_tile_px, 8)

    def test_update_grid_radius(self):
        self.grid.update_grid_radius(3)
        self.assertEqual(self.grid.radius_grid_hx, 3)

    def test_update_origin(self):
        self.grid.update_origin(np.array([1, 1, 1]))
        self.assertTrue(np.array_equal(self.grid.origin, np.array([1, 1, 1])))

    def test_update_padding(self):
        self.grid.update_padding(3)
        self.assertEqual(self.grid.padding_grid, 3)

    def test_update_orientation(self):
        self.grid.update_orientation("POINTY")
        self.assertEqual(self.grid.orientation_hx, "POINTY")

    def test_find_index(self):
        index = self.grid.find_index((0, 0, 0))
        self.assertEqual(index, 0)

    def test_str_neighbors(self):
        neighbors = self.grid.str_neighbors()
        self.assertIsInstance(neighbors, str)

    # def test_export_to_excel(self):
    #     self.grid.export_to_excel("hexgrid.xlsx")
    #     # Add assertions to check if the file was successfully exported

    def test_export_to_txt(self):
        # Insert items into the hexgrid to print to text
        for i in range(len(self.grid)):
            self.grid[i] = i
        # Print grid
        self.grid.export_to_txt(os.path.join(out_folder, "hexgrid.txt"))
        # Add assertions to check if the file was successfully exported

    def test_save_and_load(self):
        self.grid.save(os.path.join(out_folder, "hexgrid.pkl"))
        loaded_grid = HexGrid.load(os.path.join(out_folder, "hexgrid.pkl"))
        self.assertEqual(len(loaded_grid), len(self.grid))
        # Add more assertions to check if the loaded grid is the same as the original grid

    def test_to_json(self):
        json_str = self.grid.to_json()
        self.assertIsInstance(json_str, str)
        # Add assertions to check the JSON string

    def test_from_json(self):
        json_str = self.grid.to_json()
        loaded_grid = HexGrid.from_json(json_str)
        self.assertEqual(len(loaded_grid), len(self.grid))
        # Add more assertions to check if the loaded grid is the same as the original grid

if __name__ == "__main__":
    unittest.main()