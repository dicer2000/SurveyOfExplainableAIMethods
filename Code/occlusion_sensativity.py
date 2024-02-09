# Python Explain - Occlusion Sensativity
#(c) by Brett Huffman

#  Need: pip install tf-explain


import zipfile
from io import BytesIO

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
#import tensorflow_hub as hub
from PIL import Image
#from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os,sys
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from skimage.segmentation import mark_boundaries

RESOLUTION = 224
PATCH_SIZE = 16

crop_layer = keras.layers.CenterCrop(RESOLUTION, RESOLUTION)
norm_layer = keras.layers.Normalization(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
)
rescale_layer = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)


def preprocess_image(image, model_type, size=RESOLUTION):
    # Turn the image into a numpy array and add batch dim.
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    # If model type is vit rescale the image to [-1, 1].
    if model_type == "original_vit":
        image = rescale_layer(image)

    # Resize the image using bicubic interpolation.
    resize_size = int((256 / 224) * size)
    image = tf.image.resize(image, (resize_size, resize_size), method="bicubic")

    # Crop the image.
    image = crop_layer(image)

    # If model type is DeiT or DINO normalize the image.
    if model_type != "original_vit":
        image = norm_layer(image)

    return image.numpy()


def load_image_from_url(url, model_type):
    # Credit: Willi Gierke
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    preprocessed_image = preprocess_image(image, model_type)
    return image, preprocessed_image

# ImageNet-1k label mapping file and load it.

mapping_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
)

with open(mapping_file, "r") as f:
    lines = f.readlines()
imagenet_int_to_str = [line.rstrip() for line in lines]

img_url = "https://dl.fbaipublicfiles.com/dino/img.png"
img_url = "https://files.worldwildlife.org/wwfcmsprod/images/African_Elephant_Kenya_112367/story_full_width/qxyqxqjtu_WW187785.jpg"
#img_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=724&q=80"

image, preprocessed_image = load_image_from_url(img_url, model_type="original_vit")

p = np.squeeze(preprocessed_image, axis=0)
#print(p.shape)

#plt.imshow(preprocessed_image[0])
#plt.axis("off")
#plt.show()

# Load models
import os
GITHUB_RELEASE = "https://github.com/sayakpaul/probing-vits/releases/download/v1.0.0/probing_vits.zip"
FNAME = "probing_vits.zip"
MODELS_ZIP = {
    "vit_dino_base16": "Probing_ViTs/vit_dino_base16.zip",
    "vit_b16_patch16_224": "Probing_ViTs/vit_b16_patch16_224.zip",
    "vit_b16_patch16_224-i1k_pretrained": "Probing_ViTs/vit_b16_patch16_224-i1k_pretrained.zip",
}
'''
zip_path = tf.keras.utils.get_file(
    fname=FNAME,
    origin=GITHUB_RELEASE,
)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("./")

os.rename("Probing ViTs", "Probing_ViTs")

'''
def load_model(model_path: str) -> tf.keras.Model:
#    with zipfile.ZipFile(model_path, "r") as zip_ref:
#        zip_ref.extractall("Probing_ViTs/")
    model_name = model_path.split(".")[0]

    inputs = keras.Input((RESOLUTION, RESOLUTION, 3))
    model = keras.models.load_model(model_name, compile=False)
    outputs, attention_weights = model(inputs, training=False)

    return keras.Model(inputs, outputs=[outputs, attention_weights])


vit_base_i21k_patch16_224 = load_model(MODELS_ZIP["vit_b16_patch16_224-i1k_pretrained"])
print("Model loaded.")


# Try to predict from the sample image
predictions, attention_score_dict = vit_base_i21k_patch16_224.predict(
    preprocessed_image
)
predicted_label = imagenet_int_to_str[int(np.argmax(predictions))]
print(predicted_label)
print(np.argmax(predictions))
print(imagenet_int_to_str[386])


def pred_fn(imgs):
    tot_probs = []
    for img in imgs:
        # Add the explanation dimension
        exp_img = np.expand_dims(img, axis=0)
        # Make the prediction
        img_pred, _ = vit_base_i21k_patch16_224.predict(exp_img)
        # Add the predictions to a list to be returned to LIME
        tot_probs.append(img_pred[0])
    return tot_probs

#y= pred_fn([preprocessed_image])

# Occlusion Analysis
data = ([preprocessed_image], None)

cla./ss_index = int(np.argmax(predictions))
explainer = OcclusionSensitivity()
# Compute Occlusion Sensitivity for patch_size 16
grid = explainer.explain(data, vit_base_i21k_patch16_224, class_index, 16)
explainer.save(grid, ".", "16.png")

#temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=3, hide_rest=False)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
# ax1.imshow(mark_boundaries(temp_1, mask_1))
# ax2.imshow(mark_boundaries(temp_2, mask_2))
# ax1.axis('off')
# ax2.axis('off')
# ax1.set_title('')
# ax2.set_title('')
# fig.tight_layout()

# plt.show()


# def explanation_heatmap(exp, exp_class):
#     '''
#     Using heat-map to highlight the importance of each super-pixel for the model prediction
#     '''
#     dict_heatmap = dict(exp.local_exp[exp_class])
#     heatmap = np.vectorize(dict_heatmap.get)(exp.segments) 
#     plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
#     plt.colorbar()
#     plt.show()

# explanation_heatmap(explanation, explanation.top_labels[0])
