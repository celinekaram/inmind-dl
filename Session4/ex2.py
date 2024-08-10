import open3d as o3d
import numpy as np
import os

# Function to get all point clouds .ply files in a directory
def get_ply_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.ply')]

# Function to load point cloud from a given path
def load_point_cloud(path):
    return o3d.io.read_point_cloud(path)

# Function to visualize a point cloud using Open3D
def visualize(pcd, title="Point Cloud"):
    print(f"Visualizing: {title}")
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

# Function to drop a percentage of points from the point cloud
def drop_points(pcd, drop_percentage=30):
    # Convert points to numpy array & calculate nb points to drop
    points = np.asarray(pcd.points) 
    total_points = points.shape[0]
    num_points_to_drop = int(total_points * drop_percentage / 100)
    
    # Randomly select indices to drop with mask = 0
    indices_to_drop = np.random.choice(total_points, num_points_to_drop, replace=False)
    mask = np.ones(total_points, dtype=bool)
    mask[indices_to_drop] = False
    
    # Create a new point cloud with remaining points
    points_remaining = points[mask]
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points_remaining)
    
    # Retain colors and normals of original pointcloud if availaible
    if pcd.has_colors(): new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals(): new_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

    return new_pcd

# Function to add jitter (random displacement) to points in the point cloud
def add_jitter(pcd, jitter_strength=0.01):
    # Add to points a random noise array with same shape as the points array and
    # values between -jitter_strength and +jitter_strength giving a noise value in x, y, z
    points = np.asarray(pcd.points)
    noise = np.random.uniform(-jitter_strength, jitter_strength, size=points.shape)
    points_jittered = points + noise
    
    # Create a new point cloud with jittered points
    jittered_pcd = o3d.geometry.PointCloud()
    jittered_pcd.points = o3d.utility.Vector3dVector(points_jittered)
    
     # Retain colors and normals of original pointcloud if availaible    
    if pcd.has_colors(): jittered_pcd.colors = pcd.colors
    if pcd.has_normals(): jittered_pcd.normals = pcd.normals

    return jittered_pcd

# Function to add Gaussian noise to points in the point cloud
def add_noise(pcd, noise_strength=0.01):
    # Add to points a random noise array with values from normal (Gaussian) 
    # distribution with a mean of loc = 0.0 & standard deviation of noise_strength
    points = np.asarray(pcd.points)
    noise = np.random.normal(loc=0.0, scale=noise_strength, size=points.shape)
    points_noisy = points + noise
    
    # Create a new point cloud with points + noise
    noisy_pcd = o3d.geometry.PointCloud()
    noisy_pcd.points = o3d.utility.Vector3dVector(points_noisy)
    
    # Retain colors and normals of original pointcloud if availaible
    if pcd.has_colors(): noisy_pcd.colors = pcd.colors
    if pcd.has_normals(): noisy_pcd.normals = pcd.normals

    return noisy_pcd

# Function to perturb the colors of the point cloud
def perturb_colors(pcd, color_strength=0.2):
    # Add to colors a random noise array with values between -color_strength and +color_strength
    # Resulting color values should stay within the valid range [0.0, 1.0]
    colors = np.asarray(pcd.colors)
    color_noise = np.random.uniform(-color_strength, color_strength, size=colors.shape)
    perturbed_colors = np.clip(colors + color_noise, 0.0, 1.0)
    
    # Create a new point cloud with perturbed colors
    perturbed_pcd = o3d.geometry.PointCloud()
    perturbed_pcd.points = pcd.points
    perturbed_pcd.colors = o3d.utility.Vector3dVector(perturbed_colors)
    
    # Retain normals of original pointcloud if availaible
    if pcd.has_normals(): perturbed_pcd.normals = pcd.normals

    return perturbed_pcd

# Function to apply all augmentations to a point cloud
def apply_augmentations(pcd):
    pcd = drop_points(pcd)
    pcd = add_jitter(pcd)
    pcd = add_noise(pcd)
    pcd = perturb_colors(pcd)
    return pcd

# Function to load, augment, and visualize point clouds
def process_and_visualize_point_clouds(paths):
    for i, path in enumerate(paths):
        # Load the point cloud
        original_pcd = load_point_cloud(path)
        
        # Apply augmentations
        augmented_pcd = apply_augmentations(original_pcd)
        
        # Visualize original point cloud
        visualize(original_pcd, title=f"Original Point Cloud {i+1}")
        
        # Visualize augmented point cloud
        visualize(augmented_pcd, title=f"Augmented Point Cloud {i+1}")

def main():
    # Define the directory containing point cloud files
    directory = './Pointclouds'

    # Get all point cloud files in the directory
    ply_files = get_ply_files(directory)

    # Process and visualize all point clouds
    process_and_visualize_point_clouds(ply_files)
    
if __name__ == "__main__":
    main()