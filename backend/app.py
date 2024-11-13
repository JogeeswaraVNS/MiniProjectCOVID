import tensorflow as tf
from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import os
import io
from PIL import Image
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from keras.models import load_model
from keras.layers import PReLU
from tensorflow.keras.preprocessing.image import img_to_array
import keras.backend as K

app = Flask(__name__)
CORS(app)

def squash(x, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * x

def loss_fn(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.45 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

custom_objects = {
    'loss_fn': loss_fn,
    'squash': squash,
    'PReLU': PReLU
}
model = load_model('C:/Users/PVR SUDHAKAR/Desktop/MiniProjectCOVID/backend/Model/FilterModel.h5', custom_objects=custom_objects)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

PATCH_SIZE = 16
IMG_SIZE = 64
k = 3


def image_to_patches(image, patch_size):
    h, w, c = image.shape
    patches = [
        image[i:i + patch_size, j:j + patch_size]
        for i in range(0, h, patch_size)
        for j in range(0, w, patch_size)
    ]
    return patches

def patches_to_image(patches, image_shape, patch_size):
    h, w, c = image_shape
    image_reconstructed = np.zeros((h, w, c), dtype=np.uint8)
    patch_idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            image_reconstructed[i:i + patch_size, j:j + patch_size] = patches[patch_idx]
            patch_idx += 1
    return image_reconstructed

def save_image(image):
    save_path = os.path.join(UPLOAD_FOLDER, '/PatchedImg.png')
    print(save_path)
    cv2.imwrite(save_path, image)

def get_knn_for_patch(patches, patch_index, k):
    target_patch = patches[patch_index].flatten()
    patches_flattened = [patch.flatten() for patch in patches]
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(patches_flattened)
    distances, indices = nn.kneighbors([target_patch])
    return indices[0]

def process_patches_with_neighbors(patches, k):
    updated_patches = []
    neighbors = {}
    for i in range(len(patches)):
        neighbors[i] = get_knn_for_patch(patches, i, k)
    for i in neighbors:
        arr = [patches[j] for j in neighbors[i]]
        updated_patches.append(np.mean(arr, axis=0))
    return updated_patches

@app.route('/get-patch', methods=['POST'])
def get_patch():
    data = request.json
    if 'image_url' not in data:
        return jsonify({"error": "Image URL not provided"}), 400
    image_url = data['image_url']
    image = cv2.imread(image_url)
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    patches = image_to_patches(image, PATCH_SIZE)
    updated_patches = process_patches_with_neighbors(patches, k)
    reconstructed_image = patches_to_image(updated_patches, image.shape, PATCH_SIZE)
    processed_image = np.expand_dims(reconstructed_image, axis=0)
    result=model.predict(processed_image)
    print(result)
    # save_image(reconstructed_image)
    img_io = io.BytesIO()
    Image.fromarray(reconstructed_image).save(img_io, 'PNG')
    img_io.seek(0)
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
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)
    return jsonify({"message": "Image uploaded successfully", "image_url": f"/uploads/{image.filename}"})

if __name__ == '__main__':
    app.run(debug=True)
