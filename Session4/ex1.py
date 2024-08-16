import open3d as o3d
import numpy as np
import os

# Function to get all point cloud .ply files in a directory
def get_ply_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.ply')]

# Function to load a point cloud from a given path
def load_point_cloud(path):
    return o3d.io.read_point_cloud(path)

# Function to visualize a point cloud using Open3D
def visualize(pcd, title="Point Cloud"):
    print(f"Visualizing: {title}")
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )

# Function to print information about the point cloud: min, max, range, center
def print_info(pcd):
    # Compute the minimum and maximum bounds of x, y, z coordinates
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Compute the center of the point cloud
    center = points.mean(axis=0)
    
    # Print info of point cloud
    print(f"Min bounds: {min_bound}")
    print(f"Max bounds: {max_bound}")
    print(f"Range: {max_bound - min_bound}")
    print(f"Center: {center}")

# Function to change color to blue with a decreasing intensity as we move away from center
def change_color(pcd):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    
    # Compute the center of the point cloud
    center = points.mean(axis=0)
    
    # Compute the distance of each point from the center
    distances = np.linalg.norm(points - center, axis=1)
    
    # Normalize distances for coloring
    max_distance = distances.max()
    distances_normalized = distances / max_distance
    
    # Create colors based on the distances (blue color intensity increases with distance)
    # Initialize colors array (RGB)
    colors = np.zeros((points.shape[0], 3))
    
    # Full intensity at the center (0 distance) and reduced intensity with distance
    # To achieve this, we reverse the normalized distance for the blue channel
    colors[:, 0] = 0                        # Red channel (always 0)
    colors[:, 1] = 0                        # Green channel (always 0)
    colors[:, 2] = 1 - distances_normalized  # Blue channel (intensity decreases with distance)
    
    # Assign colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the colored point cloud
    print("Colored Point Cloud:")
    visualize(pcd, "Colored Point Cloud")

def main():
    directory = "./data"
    ply_files = get_ply_files(directory)
    for file in ply_files:
        print(f"Processing file: {file}")
        pcd = load_point_cloud(file)
        visualize(pcd, "Original Point Cloud")
        print_info(pcd)
        change_color(pcd)
        
if __name__ == "__main__":
    main()