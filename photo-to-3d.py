import numpy as np
import cv2
import torch
from PIL import Image
import trimesh
from rembg import remove
from transformers import pipeline
import os

def extract_object(image_path):
    """Remove background and extract the main object from the image."""
    print("Extracting object from image...")
    # Load image
    input_image = Image.open(image_path)
    
    # Remove background
    output_image = remove(input_image)
    
    # Save and return processed image
    processed_path = "extracted_object.png"
    output_image.save(processed_path)
    print(f"Object extracted and saved to {processed_path}")
    return processed_path, np.array(output_image)

def generate_depth_map(image_path):
    """Generate a depth map from the image using a pre-trained model."""
    print("Generating depth map...")
    # Initialize depth estimation model
    try:
        depth_estimator = pipeline("depth-estimation")
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Downloading depth estimation model for the first time...")
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    
    image = Image.open(image_path)
    
    # Get depth map
    depth_result = depth_estimator(image)
    depth_map = depth_result["depth"]
    depth_map = np.array(depth_map)
    
    # Normalize depth map
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Save depth map for visualization
    depth_image = (depth_map * 255).astype(np.uint8)
    cv2.imwrite("depth_map.png", depth_image)
    print("Depth map generated and saved to depth_map.png")
    
    return depth_map

def create_3d_model(image_array, depth_map, output_path="model.obj"):
    """Create a 3D mesh from the image and depth map."""
    print("Creating 3D model...")
    # Get image dimensions
    height, width = depth_map.shape
    
    # Create vertices grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create vertices - flip the y-axis to correct orientation
    # In image coordinates, y increases downward, but in 3D, y typically increases upward
    vertices = np.stack([x.flatten(), (height - 1 - y).flatten(), depth_map.flatten() * 50], axis=1)
    
    # Create faces (triangles)
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            v0 = i * width + j
            v1 = i * width + (j + 1)
            v2 = (i + 1) * width + j
            v3 = (i + 1) * width + (j + 1)
            
            # Add two triangles for each quad - adjust winding order
            faces.append([v0, v2, v1])  # Changed order to maintain correct face normals
            faces.append([v1, v2, v3])  # Changed order to maintain correct face normals
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Apply colors from the image
    # Ensure we're using RGBA colors for proper color mapping
    if image_array.shape[2] == 4:  # If RGBA
        colors = image_array[:, :, :4].reshape(-1, 4)
    else:  # If RGB
        # Add alpha channel (fully opaque)
        alpha = np.full((image_array.shape[0], image_array.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate((image_array[:, :, :3], alpha), axis=2)
        colors = rgba.reshape(-1, 4)
    
    # Make sure colors match vertices
    if len(colors) == len(vertices):
        mesh.visual.vertex_colors = colors
    else:
        print(f"Warning: Color array length ({len(colors)}) doesn't match vertices length ({len(vertices)})")
        # If image dimensions don't match depth map dimensions, resample
        if image_array.shape[0] != height or image_array.shape[1] != width:
            print("Resampling image to match depth map dimensions...")
            # Resize image to match depth map dimensions
            from PIL import Image
            resized_img = np.array(Image.fromarray(image_array).resize((width, height)))
            
            # Create colors from resized image
            if resized_img.shape[2] == 4:  # If RGBA
                colors = resized_img[:, :, :4].reshape(-1, 4)
            else:  # If RGB
                # Add alpha channel (fully opaque)
                alpha = np.full((resized_img.shape[0], resized_img.shape[1], 1), 255, dtype=np.uint8)
                rgba = np.concatenate((resized_img[:, :, :3], alpha), axis=2)
                colors = rgba.reshape(-1, 4)
            
            mesh.visual.vertex_colors = colors
        else:
            # Resize color array if needed
            if len(colors) > len(vertices):
                colors = colors[:len(vertices)]
            else:
                # Pad with white if needed
                if colors.shape[1] == 4:
                    padding = np.ones((len(vertices) - len(colors), 4), dtype=np.uint8) * 255
                else:
                    padding = np.ones((len(vertices) - len(colors), 3), dtype=np.uint8) * 255
                colors = np.vstack([colors, padding])
            
            mesh.visual.vertex_colors = colors
    
    # Export the mesh
    mesh.export(output_path)
    
    return output_path

def photo_to_3d(image_path, output_format="obj"):
    """Main function to convert a photo to a 3D model."""
    print(f"Processing image: {image_path}")
    
    # Extract object from background
    processed_image_path, image_array = extract_object(image_path)
    
    # Generate depth map
    depth_map = generate_depth_map(processed_image_path)
    
    # Create output path based on format
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_3d.{output_format}"
    
    # Create 3D model
    model_path = create_3d_model(image_array, depth_map, output_path)
    
    print(f"3D model created successfully: {model_path}")
    return model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert a photo to a 3D model")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--format", choices=["obj", "stl"], default="obj", 
                        help="Output format (obj or stl)")
    
    args = parser.parse_args()
    
    try:
        photo_to_3d(args.image_path, args.format)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
    photo_to_3d(args.image_path, args.format)
