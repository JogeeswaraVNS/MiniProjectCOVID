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
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import keras.backend as K
import keras
import matplotlib.pyplot as plt

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
model = load_model('../Models_GCNN/R4G2P16model.h5', custom_objects=custom_objects)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

PATCH_SIZE = 16
IMG_SIZE = 128
k = 2

def preprocess_image(file, target_size):
    try:
        file.seek(0)
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')
        image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        raise e

# Helper function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save Grad-CAM image and return it as an in-memory file
def save_gradcam_image(img_path, heatmap, alpha=0.4):
    img = load_img(img_path)
    img = img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))

    jet_heatmap = img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    img_io = io.BytesIO()
    Image.fromarray(superimposed_img).save(img_io, 'PNG')
    img_io.seek(0)
    
    return img_io


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
    img_url=f"./uploads/{image.filename}"
    image = cv2.imread(img_url)
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    patches = image_to_patches(image, PATCH_SIZE)
    updated_patches = process_patches_with_neighbors(patches, k)
    reconstructed_image = patches_to_image(updated_patches, image.shape, PATCH_SIZE)
    processed_image = np.expand_dims(reconstructed_image, axis=0)
    result=model.predict(processed_image)
    print(result)
    preclass = {0: "Positive", 1: "Negative"}
    res=preclass[np.argmax(result)]
    return jsonify({"message": res, "image_url": img_url})


# Route to handle file uploads and Grad-CAM generation Layer 1
@app.route('/GradCamLayer1', methods=['POST'])
def gradcam_layer_1():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)
    img_url=f"./uploads/{image.filename}"
    image = cv2.imread(img_url)
    if image is None:
        return jsonify({"error": "Failed to load image"}), 400
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    patches = image_to_patches(image, PATCH_SIZE)
    updated_patches = process_patches_with_neighbors(patches, k)
    file = patches_to_image(updated_patches, image.shape, PATCH_SIZE)
    preclass = {0: "Positive", 1: "Negative"}
    last_conv_layer_name = "cnn1"
    if file:
        processed_image = preprocess_image(file, target_size=(128, 128))
        results = model.predict(processed_image)
        response = preclass[np.argmax(results)]
        
        original_filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.seek(0)
        file.save(save_path)

        img_array = preprocess_image(file, target_size=(128, 128))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        gradcam_img_io = save_gradcam_image(save_path, heatmap)

        try:
            os.remove(save_path)
        except Exception as e:
            print(f"Error deleting file {save_path}: {e}")

        return send_file(gradcam_img_io, mimetype='image/png')
    else:
        return jsonify("Error")
    
    

# Route to handle file uploads and Grad-CAM generation Layer 2
@app.route('/GradCamLayer2', methods=['POST'])
def gradcam_layer_2():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    preclass = {0: "Positive", 1: "Negative"}
    last_conv_layer_name = "cnn2"
    if file:
        processed_image = preprocess_image(file, target_size=(128, 128))
        results = model.predict(processed_image)
        response = preclass[np.argmax(results)]
        
        original_filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.seek(0)
        file.save(save_path)

        img_array = preprocess_image(file, target_size=(128, 128))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        gradcam_img_io = save_gradcam_image(save_path, heatmap)

        try:
            os.remove(save_path)
        except Exception as e:
            print(f"Error deleting file {save_path}: {e}")

        return send_file(gradcam_img_io, mimetype='image/png')
    else:
        return jsonify("Error")
    


# Route to handle file uploads and Grad-CAM generation Layer 3
@app.route('/GradCamLayer3', methods=['POST'])
def gradcam_layer_3():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    preclass = {0: "Positive", 1: "Negative"}
    last_conv_layer_name = "cnn3"
    if file:
        processed_image = preprocess_image(file, target_size=(128, 128))
        results = model.predict(processed_image)
        response = preclass[np.argmax(results)]
        
        original_filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.seek(0)
        file.save(save_path)

        img_array = preprocess_image(file, target_size=(128, 128))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        gradcam_img_io = save_gradcam_image(save_path, heatmap)

        try:
            os.remove(save_path)
        except Exception as e:
            print(f"Error deleting file {save_path}: {e}")

        return send_file(gradcam_img_io, mimetype='image/png')
    else:
        return jsonify("Error")  
    
    
    
# Route to handle file uploads and Grad-CAM generation Layer 4
@app.route('/GradCamLayer4', methods=['POST'])
def gradcam_layer_4():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    preclass = {0: "Positive", 1: "Negative"}
    last_conv_layer_name = "cnn4"
    if file:
        processed_image = preprocess_image(file, target_size=(128, 128))
        results = model.predict(processed_image)
        response = preclass[np.argmax(results)]
        
        original_filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.seek(0)
        file.save(save_path)

        img_array = preprocess_image(file, target_size=(128, 128))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        gradcam_img_io = save_gradcam_image(save_path, heatmap)

        try:
            os.remove(save_path)
        except Exception as e:
            print(f"Error deleting file {save_path}: {e}")

        return send_file(gradcam_img_io, mimetype='image/png')
    else:
        return jsonify("Error")  
    



if __name__ == '__main__':
    app.run(debug=True)
