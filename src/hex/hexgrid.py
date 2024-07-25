import pickle

import numpy as np
from typing import Literal
import math
import pandas as pd
import json

## Notes
# Implemented from https://www.redblobgames.com/grids/hexagons/#coordinates
# and https://www.redblobgames.com/grids/hexagons/implementation.html
# Note that the code is labeled from the perspective of POINTY orientation, but
# is usable for both, since they are just rotations of each other. This only 
# makes a difference in the direction names used; handy for visualization.

# region Constants

NE = np.array((0, -1, 1))
NW = np.array((-1, 0, 1))
W = np.array((-1, 1, 0))
SW = np.array((0, 1, -1))
SE = np.array((1, 0, -1))
E = np.array((1, -1, 0))
DIRECTIONS = np.array([NW, NE, E, SE, SW, W, ])
DIR_NAME_DICT = {str(NE): 'NE', str(NW): 'NW', str(W): 'W', str(SW): 'SW', str(SE): 'SE', str(E): 'E'}


# Matrix for converting axial coordinates to pixel coordinates
MAT_AX2PX_POINTY = np.array([[math.sqrt(3), math.sqrt(3) / 2], [0, 3 / 2.]])
MAT_AX2PX_FLAT = np.array([[3 / 2., 0],[math.sqrt(3) / 2, math.sqrt(3)]])

# Matrix for converting pixel coordinates to axial coordinates
MAT_PX2AX_POINTY = np.linalg.inv(MAT_AX2PX_POINTY)
MAT_PX2AX_FLAT = np.linalg.inv(MAT_AX2PX_FLAT)


# Hex Mapping Functions
################################################################################
def is_hex_in_grid_cube(coord_cube: np.array, radius: int) -> bool:
    """
    Returns True if a given hex is within a hex grid of a given radius.
    """
    
    # Find index of coord_cube
    idx = None
    for i, coord in enumerate(get_spiral_cube(np.array((0,0,0)), 0, radius)):
        if np.array_equal(coord_cube, coord):
            idx = i
            break
    
    return idx is not None

def is_hex_in_grid_cube_cache(coord_cube, radius, valid_hexes: set):
    """
    Returns True if a given hex is within a hex grid of a given radius.
    """
    return tuple(coord_cube) in valid_hexes

def is_hex_on_edge(coord_cube: np.array, radius: int, center: np.array = np.array((0,0,0))) -> bool:
    """
    Returns True if a given hex is on the edge of a hex grid of a given radius.
    """
    ring = get_ring_cube(center, radius)
    for hex in ring:
        if np.array_equal(hex, coord_cube):
            return True
    return False

def get_last_hex_in_direction(start_hex: np.array, direction: np.array, radius) -> np.array:
    """
    Returns the last hex in a given direction, starting at a given hex, and going
    for a given radius.
    """
    return start_hex + direction * radius

def get_direction_of_hex_cube(on_coord_cube, to_coord_cube):
    """
    Returns the direction of the to hex from the on hex
    """
    direction = to_coord_cube - on_coord_cube
    direction = direction / np.sum(np.abs(direction))
    # Return the equivalent direction
    return DIRECTIONS[np.argmin(np.sum(np.abs(DIRECTIONS - direction), axis=1))]

    # direction = to_coord_cube - on_coord_cube
    # direction = direction / np.sum(np.abs(direction))

def get_corner_hexes(start_hex: np.array, radius: int) -> np.array:
    """
    Returns the hexes at the corners of a hex grid of a given radius, starting at
    a given hex.
    """
    return np.array([get_last_hex_in_direction(start_hex, direction, radius) for direction in DIRECTIONS])

def get_hexes_radius_from_hx_count(hx_count: int) -> int:
    """
    Finds radius requred for a hex grid to contain a given number of hexes.
    """
    return np.ceil(math.sqrt((hx_count - 1) / 3 + 1 / 4) - 1 / 2)

def get_dist_cube(start_hx: np.array, end_hx: np.array):
    """
    Distance between two hexes, in hexes units, using cube coordinates.
    """
    return np.sum(np.abs(start_hx - end_hx) / 2)

def get_line_cube(start_hx, end_hx)-> np.array:
    """
    Returns a list of hexes in a straight line between two hexes
    Get hexes on line from start_hx to end_hx
    """
    hex_distance = get_dist_cube(start_hx, end_hx)
    if hex_distance < 1:
        return np.array([start_hx])

    # Set up linear system to compute linearly interpolated cube points
    bottom_row = np.array([i / hex_distance for i in np.arange(hex_distance)])
    x = np.vstack((1 - bottom_row, bottom_row))
    A = np.vstack((start_hx, end_hx)).T

    # linearly interpolate from a to b in n steps
    interpolated_points = A.dot(x)
    interpolated_points = np.vstack((interpolated_points.T, end_hx))
    return np.array(get_roundings_cube(interpolated_points))

def get_neighbor_coord_cube(coord_cube, direction):
    """
    Returns cube coord of neighbor in given direction
    """
    return coord_cube + direction

def get_neighbor_closest_to_origin(coord_cube):
    """
    Returns cube coord of neighbor closest to origin
    """
    return get_neighbor_coord_cube(coord_cube, np.argmin(np.sum(np.abs(coord_cube), axis=0)))

def get_ring_cube(center, radius) -> np.array:
    """
    Returns cube coords of all hexes in a ring of given radius around center
    """
    if radius < 0:
        return []
    if radius == 0:
        return [center]

    radius_hx = np.zeros((6 * radius, 3))
    count = 0
    for i in range(0, 6):
        for k in range(0, radius):
            radius_hx[count] = DIRECTIONS[i - 1] * (radius - k) + DIRECTIONS[i] * (k)
            count += 1

    return np.squeeze(radius_hx) + center

def get_cached_ring(coord, ring_cache: dict):
    """
    Get the ring for a given coordinate from the cache.
    :param coord: The coordinate of the center hex.
    :return: List of coordinates in the ring.
    """
    return ring_cache.get(tuple(coord), [])

def get_disk_cube(center, radius)-> np.array:
    """
    Returns cube coords of all hexes within given radius of center
    """
    return get_spiral_cube(center, 0, radius)

def get_spiral_cube(center, radius_start=1, radius_end=2)-> np.array:
    """
    Returns cube coords of all hexes between given radius start and radius end
    """
    hex_area = get_ring_cube(center, radius_start)
    for i in range(radius_start + 1, radius_end + 1):
        hex_area = np.append(hex_area, get_ring_cube(center, i), axis=0)
    return np.array(hex_area)

def get_spiral_cube2(center, radius_start=1, radius_end=2):
    """
    Returns cube coords of all hexes between given radius start and radius end.
    """
    # Calculate the total number of hexes
    total_hexes = 1 + sum(6 * r for r in range(radius_start, radius_end + 1)) if radius_end > 0 else 1
    hex_area = np.zeros((total_hexes, 3))

    # Populate hex_area with hex coordinates
    idx = 0
    if radius_start == 0:
        hex_area[idx] = center
        idx += 1

    for radius in range(max(radius_start, 1), radius_end + 1):
        ring = get_ring_cube(center, radius)
        ring_size = len(ring)
        hex_area[idx:idx + ring_size] = ring
        idx += ring_size

    return hex_area

# Coordinate conversions
################################################################################
def get_roundings_cube(coords_cube: np.array) -> np.array:
    """
    Returns roundings of cube coordinates to center of nearest hex
    :return: Cube coordinate of center of nearest hex
    """
    rounded = np.zeros((coords_cube.shape[0], 3))
    rounded_cubes = np.round(coords_cube)
    for i, coord in enumerate(rounded_cubes):
        (rx, ry, rz) = coord
        xdiff, ydiff, zdiff = np.abs(coord-coords_cube[i])
        if xdiff > ydiff and xdiff > zdiff:
            rx = -ry - rz
        elif ydiff > zdiff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        rounded[i] = (rx, ry, rz)
    return rounded

def get_roundings_ax(coords_ax: np.array) -> np.array:
    """
    Returns roundings of axial coordinates to center of nearest hex
    :return: Axial coordinate of center of nearest hex
    """
    return get_cube2ax(get_roundings_cube(get_ax2cube(coords_ax)))

def get_cube2ax(coords_cube)-> np.array:
    """
    Cube coords to axial coords
    """
    return np.vstack((coords_cube[:, 0], coords_cube[:, 2])).T

def get_ax2cube(coords_ax)-> np.array:
    """
    Axiel coords to cube coords
    """
    x = coords_ax[:, 0]
    z = coords_ax[:, 1]
    y = -x - z
    cube_coords = np.vstack((x, y, z)).T
    return cube_coords

def get_ax2px_pointy(axial, radius, padding=0)-> np.array:
    """
    Axial coords to pixel coords (pointy orientation)
    """
    pos = (radius + padding) * MAT_AX2PX_POINTY.dot(axial.T)
    return pos.T

def get_ax2px_flat(coords_ax, radius, padding=0)-> np.array:
    """
    Axial coords to pixel coords (flat orientation)
    """
    pos = (radius + padding) * MAT_AX2PX_FLAT.dot(coords_ax.T)
    return pos.T

def get_cube2px_pointy(coords_cube, radius, padding=0)-> np.array:
    """
    Cube coords to pixel coords (pointy orientation)
    """
    coords_ax = get_cube2ax(coords_cube)
    return get_ax2px_pointy(coords_ax, radius, padding)

def get_cube2px_flat(coords_cube, radius, padding=0)-> np.array:
    """
    Cube coords to pixel coords (flat orientation)
    """
    coords_ax = get_cube2ax(coords_cube)
    return get_ax2px_flat(coords_ax, radius, padding)

def get_px2cube_pointy(pixel, radius_hx) -> np.array:
    """
    Pixel coords to cube coords (pointy orientation)
    """
    coords_ax = MAT_PX2AX_POINTY.dot(pixel.T) / radius_hx
    return get_roundings_cube(get_ax2cube(coords_ax.T))

def get_px2cube_flat(pixel, radius_hx, padding=0) -> np.array:
    """
    Pixel coords to cube coords (flat orientation)
    """
    coords_ax = MAT_PX2AX_FLAT.dot(pixel.T) / (radius_hx + padding)
    return get_roundings_cube(get_ax2cube(coords_ax.T))

def get_px2ax_pointy(coords_px, radius, padding=0) -> np.array:
    """
    Pixel coords to axial coords (pointy orientation)
    """
    coords_cube = get_px2cube_pointy(coords_px, radius, padding)
    return get_cube2ax(coords_cube)

def get_px2ax_flat(coords_px, radius, padding=0) -> np.array:
    """
    Pixel coords to axial coords (flat orientation)
    """
    coords_cube = get_px2cube_flat(coords_px, radius, padding)
    return get_cube2ax(coords_cube)

# Helper functions
################################################################################
# The bases of the axial coordinate system
MAT_BASES = get_cube2ax(np.array([SE, E], dtype=int))
def make_key_from_coordinates(indexes):
    """
    Converts indexes to string for hashing
    :param indexes: the indexes of a hex. nx2, n=number of index pairs
    :return: key for hashing based on index.
    """
    return [str(int(index[0])) + ',' + str(int(index[1])) for index in indexes]


def solve_for_indexes(hexes):
    """
    We want to solve for the coefficients in the linear combos.
    :param hexes: The hexes whose indexes we want to solve for.
                  nx2, n=number of hexes
    :return: indexes of `hexes`
    """
    if hexes.shape[1] != 2:
        raise ValueError("Must be axial coordinates!")
    return np.linalg.solve(MAT_BASES, hexes.T).T

# endregion

# Hexgrid Class
################################################################################
class HexGrid(object):
    
    def __init__(
        self, 
        radius_grid_hx: int = 1,
        width_tile_px: int = 1,
        orientation_hx: Literal['FLAT', 'POINTY'] = "FLAT",
        padding_grid: int = 0, 
        origin: np.array = np.array((0, 0, 0)),
        initialize: Literal['indices', 'value', 'zeroes'] = None, val=None,
        padding_tile: int = 0,
        category = None
        ):

        # Update width_tile_px to ensure it is even
        if width_tile_px % 2 != 0:
            width_tile_px += 1
        
        self.category = category
        self.type_hexes = None
        self.origin = origin
        self.width_tile_px = width_tile_px
        self.radius_tile_px = width_tile_px / 2
        self.radius_grid_hx = radius_grid_hx
        if self.radius_grid_hx < 1:
            raise ValueError("Grid radius must be num of hexes > 0")
        if self.radius_tile_px < 1:
            raise ValueError("Tile radius must be num pixels > 0")
        self.orientation_hx = orientation_hx
        if not self.orientation_hx in ['FLAT', 'POINTY']:
            raise ValueError("Orientation must be either 'FLAT' or 'POINTY'")
        self.padding_grid = padding_grid
        if self.padding_grid < 0:
            raise ValueError("Padding must be num pixels >= 0")
        self.padding_tile = padding_tile
        if self.padding_tile < 0:
            raise ValueError("Padding must be num pixels >= 0")
        self.aux = {}
        self.coords_cube_to_index = {}
        self._hexes = {}
        self._neighbors = {}
        # self.ring_cache = {}  # Add a field variable for storing rings
        self.valid_hexes = set()
        self.shift_constant = 0
        self._set_coords(initialize, val)
        self.num_hexes = len(self._hexes)
        self.meta = None

# region Functions

    def _set_coords(self, initialize: Literal['indices', 'value', 'zeroes'] = None, val=None):
        """_summary_
        Initializes the grid coordinates into a spiral configuration.
        Raises:
            ValueError: _description_
        """
        self.coords_cube = get_spiral_cube(self.origin, 0, self.radius_grid_hx)
        self.coord_to_index = {tuple(coord): idx for idx, coord in enumerate(self.coords_cube)}
        self.coords_ax = get_cube2ax(self.coords_cube)
        if self.orientation_hx == 'POINTY':
            self.coords_px = get_cube2px_pointy(self.coords_cube, self.radius_tile_px, self.padding_tile)
        elif self.orientation_hx == 'FLAT':
            self.coords_px = get_cube2px_flat(self.coords_cube, self.radius_tile_px, self.padding_tile)
        else:
            raise ValueError("Orientation must be either 'FLAT' or 'POINTY'")
        self.shift_constant = self.padding_grid - self.coords_px.min(axis=0)
        self.coords_px_shifted = self.coords_px + self.shift_constant

        # Initializes object dict _hexes 
        if initialize == 'indices':
            self._initialize_to_indices()
        elif initialize == 'value':
            self._initialize_to_value(val)
        elif initialize == 'zeroes':
            self._initialize_to_zeroes()
        elif initialize is None:
            self._initialize_to_none()
        else:
            raise ValueError("Initialize must be 'indices', 'value', 'zeroes', or None")
        
        # self._initialize_neighbors()
        self._initialize_aux()
        # self._populate_ring_cache(self.radius_grid_hx)
        self.valid_hexes = set(tuple(coord) for coord in get_spiral_cube(np.array((0,0,0)), 0, self.radius_grid_hx))
    
    def _populate_ring_cache(self, radius):
        """Populate the ring cache for each hex in the grid."""
        for r in range(radius+1):
            ring = get_ring_cube(self.origin, r)
            for coord in ring:
                self.ring_cache[tuple(coord)] = [tuple(neighbor) for neighbor in get_ring_cube(coord, 1)]

    def _initialize_neighbors(self):
        """
        Initializes the neighbors of each hex in the grid.
        """
        for idx, coord in enumerate(self.coords_cube):
            lst = []
            for direction in DIRECTIONS:
                neighbor = get_neighbor_coord_cube(coord, direction)
                if np.any(np.all(self.coords_cube == neighbor, axis=1)):
                    lst.append(self.find_index(neighbor))
            self._neighbors[idx] = lst
    
    def _initialize_aux(self):
        """
        Initializes the aux of each hex in the grid.
        """
        for k, v in self.items():
            self.aux[k] = None

    def _initialize_to_indices(self):
        """
        Initializes the grid to the indices of each hex.
        """
        self._hexes = {idx: idx for idx in range(len(self.coords_ax))}
    
    def _initialize_to_none(self):
        """
        Initializes the grid to None.
        """
        self._hexes = {idx: None for idx in range(len(self.coords_ax))}
    
    def _initialize_to_value(self, value):
        """
        Initializes the grid to a given value.
        """
        self._hexes = {idx: value for idx in range(len(self.coords_ax))}
    
    def _initialize_to_zeroes(self):
        """
        Initializes the grid to 0.
        """
        self._hexes = {idx: 0 for idx in range(len(self.coords_ax))}
    
    def keys(self):
        yield from self._hexes.keys()

    def values(self):
        yield from self._hexes.values()

    def items(self):
        yield from self._hexes.items()

    def __len__(self):
        return self._hexes.__len__()

    def __iter__(self):
        yield from self._hexes

    def __setitem__(self, index, obj):
        self._hexes[index] = obj
        self.type_hexes = type(obj) 
    
    def find_index(self, coord_cube):
        return self.coord_to_index.get(tuple(coord_cube))
    
    def update_tile_radius(self, radius_tile_px):
        self.radius_tile_px = radius_tile_px
        self._set_coords()
    
    def update_grid_radius(self, radius_grid_hx):
        self.radius_grid_hx = radius_grid_hx
        self._set_coords()
    
    def update_origin(self, origin):
        self.origin = origin
        self._set_coords()
    
    def update_padding(self, padding_grid):
        self.padding_grid = padding_grid
        self._set_coords()
    
    def update_orientation(self, orientation_hx: Literal['FLAT', 'POINTY']):
        self.orientation_hx = orientation_hx
        self._set_coords()

    def __delitem__(self, index):
        self._hexes.pop(index)

    def __getitem__(self, index):
        return self._hexes.get(index)
    
    def __contains__(self, coord_cube):
        return coord_cube in self.coords_cube
    
    def get_coord_ax(self, idx):
        """
        Returns the axial coordinate at the index
        """
        return self.coord_ax[idx]
    
    def get_coord_cube(self, idx):
        """
        Returns the cubic coordinate at the index
        """
        return self.coords_cube[idx]
    
    def get_coord_px(self, idx):
        """
        Returns the pixel coordinate at the index
        """
        return self.coords_px[idx]

    def get_coord_px_shifted(self, idx):
        """
        Returs the pixel shifted coordinate at the index
        """
        return self.coords_px_shifted[idx]

    def set_by_ax(self, coord_ax, obj):
        """
        Sets the hex at the given axial coordinate to obj
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_ax):
            if np.array_equal(coord, coord_ax):
                self._hexes[idx] = obj
                return
        raise ValueError("Axial coordinate not found in HexGrid")
    
    def set_by_cube(self, coord_cube, obj):
        """
        Sets the hex at the given cube coordinate to obj
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_cube):
            if np.array_equal(coord, coord_cube):
                self._hexes[idx] = obj
                return
        raise ValueError("Cube coordinate not found in HexGrid")
    
    def set_by_px(self, coord_px, obj):
        """
        Sets the hex at the given pixel coordinate to obj
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_px):
            if np.array_equal(coord, coord_px):
                self._hexes[idx] = obj
                return
        raise ValueError("Pixel coordinate not found in HexGrid")

    def set_by_px_shifted(self, coord_px_shifted, obj):
        """
        Sets the hex at the given pixel coordinate to obj
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_px_shifted):
            if np.array_equal(coord, coord_px_shifted):
                self._hexes[idx] = obj
                return
        raise ValueError("Pixel coordinate not found in HexGrid")
    
    def val_by_ax(self, coord_ax):
        """
        Returns the value of the hex at the given axial coordinate
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_ax):
            if np.array_equal(coord, coord_ax):
                return self._hexes[idx]
        return None

    def val_by_cube(self, coord_cube):
        """
        Returns the value of the hex at the given cube coordinate
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_cube):
            if np.array_equal(coord, coord_cube):
                return self._hexes[idx]
        return None
    
    def val_by_px(self, coord_px):
        """
        Returns the value of the hex at the given pixel coordinate
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_px):
            if np.array_equal(coord, coord_px):
                return self._hexes[idx]
        return None
    
    def val_by_px_shifted(self, coord_px_shifted):
        """
        Returns the value of the hex at the given pixel coordinate
        """
        # Find the index of the coord_ax in _hexes
        for (idx, val), coord in zip(self.items(), self.coords_px_shifted):
            if np.array_equal(coord, coord_px_shifted):
                return self._hexes[idx]
        return None

    def str_neighbors(self):
        """
        Returns a string representation of the neighbors of each hex in the grid.
        """
        output = []
        for key, val in self._neighbors.items():
            output.append(f"{key}: {val}")
        return '\n'.join(output)
    
    def __str__(self):
        if not self._hexes:
            return "Empty HexMap"

        # Define the fixed length for each hex content
        content_length = 5  # You can adjust this based on your requirement
        ################ (Maatlock)
        # Note: If you want to adjust the content length, you need to adjust the offset below as well
        # The content_length should be equal to the number subtracted from it below.
        # Note: the fixed length is ONE MORE than the allowed length of the hex content
        ################ (Maatlock)
        # Note that if you want to print the coordinates, set at 12 and 12
        # If you want to print the hexes, set to 5 and 5

        # Find the bounds of the grid
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for coord in self.coords_ax:
            x, y = map(int, coord)

            min_x = min(min_x, x)
            max_y = max(max_y, y)
            max_x = max(max_x, x)
            min_y = min(min_y, y)

        # Create a 2D array representation
        grid = [[' ' * content_length for _ in range(min_x, max_x + 1)] for _ in range(min_y, max_y + 1)]

        # Populate the grid
        for (key, hex_tile), coord in zip(self._hexes.items(), self.coords_ax):
            x, y = map(int, coord)
            formatted_content = str(hex_tile).ljust(content_length)  # Adjust content to fixed length
            grid[y - min_y][x - min_x] = formatted_content

        # Format the grid into a string
        output = []
        for y, row in enumerate(grid):
            offset = ' ' * (y * content_length // 2)  # Adjust offset for content length
            line = offset + (' ' * (content_length-5)).join(row)  # Adjust space between contents
        ################ (Maatlock)
        # Note: What you subtract from the content length should be equal to content length
        # Leaving it here so it can be adjusted
        ################ (Maatlock)
        # Note: the fixed length is ONE MORE than the allowed length of the hex content
            output.append(line)
                   
        output.append("=" * len(line))
        
        header = "HexGrid: Orientation: {}, Radius: {}, Length: {} Tile Radius: {}, Padding: {}, Origin: {}".format(
            self.orientation_hx, self.radius_grid_hx, len(self), self.radius_tile_px, self.padding_grid, self.origin)
        output.append(header)
        
        output.append("=" * len(line))
        
        # Format the coordinates into a string
        for index in range(len(self.coords_ax)):  # Example loop
            # Assuming self.coords_ax, self.coords_cube, self.coords_px, and self.coords_px_shifted are defined
            ax = self.coords_ax[index].astype(int)
            cu = self.coords_cube[index].astype(int)
            px = self.coords_px[index].astype(int)
            px_shifted = self.coords_px_shifted[index].astype(int)
            val = str(self._hexes[index])

            # Format each coordinate component with fixed width for alignment
            line = ("Idx: {idx:4d}: "
                    "Ax: ({ax[0]:4d}, {ax[1]:4d}): "
                    "Cu: ({cu[0]:4d}, {cu[1]:4d}, {cu[2]:4d}): "
                    "Px: ({px[0]:4d}, {px[1]:4d}): "
                    "PxS: ({pxs[0]:4d}, {pxs[1]:4d}): "
                    "Val: {val:4s}").format(
                        idx=index, ax=ax, cu=cu, px=px, pxs=px_shifted, val=val
                    )

            output.append(line)
        

        return '\n'.join(output)

    def export_to_excel(self, file_path):
        # Create a DataFrame from the hex grid data
        data = {
            "Key": list(self._hexes.keys()),
            "Coord_Ax": [f"({int(ax[0])}, {int(ax[1])})" for ax in self.coords_ax],
            "Coord_Cube": [f"({int(cu[0])}, {int(cu[1])}, {int(cu[2])})" for cu in self.coords_cube],
            "Coord_Px": [f"({int(px[0])}, {int(px[1])})" for px in self.coords_px],
            "Coord_Shifted": [f"({int(ps[0])}, {int(ps[1])})" for ps in self.coords_px_shifted],
            "Value": [str(val) for val in self._hexes.values()]
        }
        df = pd.DataFrame(data)

        # Export to Excel
        df.to_excel(file_path, index=False)
    
    def export_to_txt(self, file_path):
        with open(file_path, 'w') as f:
            f.write(str(self))
    
    def save(self, filename):
        """Save the object to a file."""
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(filename):
        """Load the object from a file."""
        with open(filename, 'rb') as input:
            return pickle.load(input)

    def to_json(self, filename=None):
        """Save the object to a JSON file."""
        dd = {}
        meta = {"radius_grid_hx": self.radius_grid_hx, "width_tile_px": self.width_tile_px,
                "orientation_hx": self.orientation_hx, "padding_grid": self.padding_grid,
                "origin": self.origin.tolist(), "padding_tile": self.padding_tile}
        
        hexes = {k: v for k, v in self._hexes.items()}
        
        dd["meta"] = meta
        # data = {k: {"hex": hexes[k], "cube": coords_cube[k], "ax": coords_ax[k]} for k in coords_cube}
        data = {k: {"hex": hexes[k]} for k in hexes}
        dd["data"] = data
        
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(dd, f, indent=4)
            
        return json.dumps(dd)

    @staticmethod
    def from_json(filename, **hexgrid_kwargs):
        """Load the object from a JSON file."""
        try:
            with open(filename, 'r') as input:
                data = json.load(input)
        except FileNotFoundError:
            try:
                data = json.loads(filename)
            except:
                raise ValueError("File not found or invalid JSON string")
        
        meta = data["meta"]
        hexes = data["data"]
        
        radius_grid_hx = meta["radius_grid_hx"] if not "radius_grid_hx" in hexgrid_kwargs else hexgrid_kwargs["radius_grid_hx"]
        width_tile_px = meta["width_tile_px"] if not "width_tile_px" in hexgrid_kwargs else hexgrid_kwargs["width_tile_px"]
        orientation_hx = meta["orientation_hx"] if not "orientation_hx" in hexgrid_kwargs else hexgrid_kwargs["orientation_hx"]
        padding_grid = meta["padding_grid"] if not "padding_grid" in hexgrid_kwargs else hexgrid_kwargs["padding_grid"]
        origin = np.array(meta["origin"]) if not "origin" in hexgrid_kwargs else hexgrid_kwargs["origin"]
        padding_tile = meta["padding_tile"] if not "padding_tile" in hexgrid_kwargs else hexgrid_kwargs["padding_tile"]
        
        grid = HexGrid(radius_grid_hx, width_tile_px, orientation_hx, padding_grid, origin, padding_tile=padding_tile)
        
        for k, v in hexes.items():
            grid[int(k)] = v['hex']
        
        return grid
    

# endregion
        
        



if __name__ == "__main__":
    
    
    grid = HexGrid(10)


        
    