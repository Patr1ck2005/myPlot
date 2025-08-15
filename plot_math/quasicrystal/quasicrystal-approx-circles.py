import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

# Constants
sqrt2 = np.sqrt(2)
silver_ratio = 1 + sqrt2
side_length = 1.0  # Basic edge length, adjustable

class Rhomb:
    def __init__(self, center, rotation=0):
        self.center = np.array(center)
        self.rotation = rotation
        self.vertices = self.get_vertices()

    def get_vertices(self):
        angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4]) + self.rotation
        scales = [side_length / 2, side_length * sqrt2 / 2, side_length / 2, side_length * sqrt2 / 2]
        dx = scales * np.cos(angles)
        dy = scales * np.sin(angles)
        return self.center + np.vstack((dx, dy)).T  # Fixed: removed [:, None] for broadcasting

    def inflate(self):
        new_tiles = []
        scale_factor = 1 / silver_ratio
        new_side = side_length * scale_factor
        offset1 = new_side / 2
        offset2 = new_side * (1 + sqrt2 / 2)
        offset3 = new_side * (sqrt2 / 2)
        directions_r1 = [np.pi/4, 5*np.pi/4]
        for d in directions_r1:
            new_center = self.center + np.array([np.cos(self.rotation + d) * offset2, np.sin(self.rotation + d) * offset2])
            new_tiles.append(Rhomb(new_center, self.rotation + d))
        directions_s = [0, np.pi/2, np.pi]
        for d in directions_s:
            new_center = self.center + np.array([np.cos(self.rotation + d) * offset1, np.sin(self.rotation + d) * offset1])
            new_tiles.append(Square(new_center, self.rotation))
        directions_r2 = [3*np.pi/4, 7*np.pi/4]
        for d in directions_r2:
            new_center = self.center + np.array([np.cos(self.rotation + d) * offset3, np.sin(self.rotation + d) * offset3])
            new_tiles.append(Rhomb(new_center, self.rotation + d - np.pi/4))  # Adjust rotation for symmetry
        return new_tiles

class Square:
    def __init__(self, center, rotation=0):
        self.center = np.array(center)
        self.rotation = rotation
        self.vertices = self.get_vertices()

    def get_vertices(self):
        angles = np.array([ -np.pi/4, np.pi/4, 3*np.pi/4, 5*np.pi/4]) + self.rotation
        dist = side_length / sqrt2
        dx = dist * np.cos(angles)
        dy = dist * np.sin(angles)
        return self.center + np.vstack((dx, dy)).T  # Fixed: removed [:, None] for broadcasting

    def inflate(self):
        new_tiles = []
        scale_factor = 1 / silver_ratio
        new_side = side_length * scale_factor
        offset1 = new_side / 2
        offset2 = new_side * (1 + sqrt2 / 2)
        directions = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
        for d in directions:
            new_center = self.center + np.array([np.cos(self.rotation + d) * offset2, np.sin(self.rotation + d) * offset2])
            new_tiles.append(Rhomb(new_center, self.rotation + d))
        directions_s = [0, np.pi/2, np.pi, 3*np.pi/2]
        for d in directions_s:
            new_center = self.center + np.array([np.cos(self.rotation + d) * offset1, np.sin(self.rotation + d) * offset1])
            new_tiles.append(Square(new_center, self.rotation))
        new_tiles.append(Square(self.center, self.rotation))  # Central square
        return new_tiles

# Start with a central rhomb at (0,0) for high symmetry
tiles = [Rhomb([0, 0], rotation=0)]

# Inflate 2 times for second-order patch centered at z=8
for _ in range(2):
    new_tiles = []
    for tile in tiles:
        new_tiles.extend(tile.inflate())
    tiles = new_tiles

# Collect all unique vertices (points) from tiles
all_vertices = np.vstack([tile.vertices for tile in tiles])
points = np.unique(np.round(all_vertices, decimals=6), axis=0)  # Deduplicate with tolerance

# Compute z (coordination number) for each point
dist_threshold = side_length / silver_ratio**2 * 1.05  # Adjusted for inflated scale
z = np.zeros(len(points))
for i in range(len(points)):
    distances = np.linalg.norm(points - points[i], axis=1)
    z[i] = np.sum((distances > 0) & (distances < dist_threshold))  # Count neighbors

# Verify center (0,0) has z=8
center_idx = np.argmin(np.linalg.norm(points, axis=1))
print(f"Center point z: {z[center_idx]} (should be 8)")

# Circle radius
circle_radius = 0.1 * side_length

# Square boundary with padding
x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
width = x_max - x_min
height = y_max - y_min
side = max(width, height)
padding = 0.1 * side
boundary_x = -side/2 - padding
boundary_y = -side/2 - padding
boundary_side = side + 2 * padding

# Visualize
fig, ax = plt.subplots(figsize=(10, 10))
for tile in tiles:
    poly = Polygon(tile.vertices, closed=True, fill=None, edgecolor='gray')
    ax.add_patch(poly)
for point in points:
    ax.add_patch(Circle(point, circle_radius, fill=None, edgecolor='blue'))
ax.add_patch(Rectangle((boundary_x, boundary_y), boundary_side, boundary_side, fill=None, edgecolor='red', linewidth=2))
ax.scatter(0, 0, c='green', marker='*', s=100, label='Center (z=8)')
ax.set_aspect('equal')
ax.autoscale()
ax.legend()
plt.title('Inflation-Centered Quasicrystal (z=8 at Center)')
plt.savefig('quasicrystal_centered_inflation.png')
plt.show()

# DXF export
with open('quasicrystal_centered_inflation.dxf', 'w') as f:
    f.write(' 0\nSECTION\n 2\nHEADER\n 0\nENDSEC\n 0\nSECTION\n 2\nENTITIES\n')
    # Circles at points
    for point in points:
        f.write(' 0\nCIRCLE\n 8\nLayer0\n')
        f.write(' 10\n{:.6f}\n 20\n{:.6f}\n 30\n0.0\n'.format(point[0], point[1]))
        f.write(' 40\n{:.6f}\n'.format(circle_radius))
    # Tiles as polylines (for edges)
    for tile in tiles:
        f.write(' 0\nPOLYLINE\n 8\nLayer0\n 66\n1\n 70\n1\n')
        for v in np.vstack([tile.vertices, tile.vertices[0]]):  # Close
            f.write(' 0\nVERTEX\n 8\nLayer0\n 10\n{:.6f}\n 20\n{:.6f}\n 30\n0.0\n'.format(v[0], v[1]))
        f.write(' 0\nSEQEND\n')
    # Square boundary
    f.write(' 0\nPOLYLINE\n 8\nLayer0\n 66\n1\n 70\n1\n')
    boundary_points = [
        [boundary_x, boundary_y],
        [boundary_x + boundary_side, boundary_y],
        [boundary_x + boundary_side, boundary_y + boundary_side],
        [boundary_x, boundary_y + boundary_side],
        [boundary_x, boundary_y]
    ]
    for pt in boundary_points:
        f.write(' 0\nVERTEX\n 8\nLayer0\n 10\n{:.6f}\n 20\n{:.6f}\n 30\n0.0\n'.format(pt[0], pt[1]))
    f.write(' 0\nSEQEND\n')
    f.write(' 0\nENDSEC\n 0\nEOF\n')

print("Generated quasicrystal_centered_inflation.png and quasicrystal_centered_inflation.dxf. Grid expanded around z=8 center.")
