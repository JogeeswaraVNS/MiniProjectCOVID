import tensorflow as tf
from flask import Flask,jsonify,request,Response,send_file
from flask_cors import CORS
import os
import io
from PIL import Image



app=Flask(__name__)

CORS(app)





UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Constants
PATCH_SIZE = 32
IMG_SIZE = 128
k = 3  # Number of neighbors to consider for each patch


def image_to_patches(image, patch_size):
    """
    Split an image into patches of given size.
    """
    h, w, c = image.shape
    patches = [
        image[i:i + patch_size, j:j + patch_size]
        for i in range(0, h, patch_size)
        for j in range(0, w, patch_size)
    ]
    return patches

def patches_to_image(patches, image_shape, patch_size):
    """
    Reconstruct an image from patches.
    """
    h, w, c = image_shape
    image_reconstructed = np.zeros((h, w, c), dtype=np.uint8)
    patch_idx = 0

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = patches[patch_idx]
            image_reconstructed[i:i+patch_size, j:j+patch_size] = patch
            patch_idx += 1

    return image_reconstructed

def save_image(image):
    """
    Save the reconstructed image as a PNG file in the 'graphed' folder of each category.
    """

    # Save the reconstructed image in the 'graphed' folder using OpenCV
    save_path = os.path.join(UPLOAD_FOLDER, 'PatchedImg.png')
    cv2.imwrite(save_path, image)

def get_knn_for_patch(patches, patch_index, k):
    """
    Get the k nearest neighbors for a patch using Euclidean distance (using scikit-learn's NearestNeighbors).
    """
    target_patch = patches[patch_index].flatten()  # Flatten the target patch
    patches_flattened = [patch.flatten() for patch in patches]  # Flatten all patches

    # Create a NearestNeighbors object and fit it to the flattened patches
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(patches_flattened)

    # Query the k nearest neighbors for the target patch
    distances, indices = nn.kneighbors([target_patch])

    # Return the indices of the k nearest neighbors
    return indices[0]

def process_patches_with_neighbors(patches, k):
    """
    Process each patch with its neighbors, apply CNN, and aggregate features.
    """
    updated_patches = []
    neighbors = {}

    for i in range(len(patches)):
        # Get K nearest neighbors for patch i
        neighbors_indices = get_knn_for_patch(patches, i, k)
        neighbors[i] = neighbors_indices

    for i in neighbors:
        arr = []
        for j in neighbors[i]:
            arr.append(patches[j])
        updated_patches.append(np.mean(arr, axis=0))  # Average the patches

    return updated_patches

def process_images_in_category(category, datadir, patch_size, k, save_dir):
    """
    Process all images in a given category, convert them into patches, apply KNN,
    reconstruct the images, and save them in the 'graphed' folder of the respective category.
    """
    category_path = os.path.join(datadir, category)
    image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg') or f.endswith('.png')]

    for i, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        # Load image
        image_path = os.path.join(category_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to a consistent size

        # Step 1: Convert the image into patches
        patches = image_to_patches(image, patch_size)

        # Step 2: Process the patches by averaging based on KNN
        updated_patches = process_patches_with_neighbors(patches, k)

        # Step 3: Reconstruct the image from the updated patches
        reconstructed_image = patches_to_image(updated_patches, image.shape, patch_size)

        # Step 4: Save the reconstructed image in the 'graphed' folder using OpenCV
        save_image(reconstructed_image)


@app.route('/get-patch', methods=['POST'])
def get_patch():
    # Expecting 'image_url' in the JSON payload
    data = request.json
    if 'image_url' not in data:
        return jsonify({"error": "Image URL not provided"}), 400

    image_url = data['image_url']

    # Load and resize the image
    image = cv2.imread(image_url)
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Step 1: Convert image into patches
    patches = image_to_patches(image, PATCH_SIZE)
    
    # Step 2: Process patches with neighbors
    updated_patches = process_patches_with_neighbors(patches, k)
    
    # Step 3: Reconstruct image from patches
    reconstructed_image = patches_to_image(updated_patches, image.shape, PATCH_SIZE)
    
    # Step 4: Convert the reconstructed image to a PNG file in memory
    img_io = io.BytesIO()
    Image.fromarray(reconstructed_image).save(img_io, 'PNG')
    img_io.seek(0)
    
    # Optionally remove the original file
    try:
        os.remove(image_url)
    except Exception as e:
        print(f"Error deleting file {image_url}: {e}")

    return send_file(img_io, mimetype='image/png')



@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400

    # Save the image to the upload folder
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)





    # Return the image path for visualization
    return jsonify({"message": "Image uploaded successfully", "image_url": f"/uploads/{image.filename}"})


if __name__ == '__main__':
    app.run(debug=True)