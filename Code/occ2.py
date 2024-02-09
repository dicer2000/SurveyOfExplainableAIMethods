# Python Explain - Occlusion Tester
#(c) by Brett Huffman

#  Need: pip install tf-explain


import tensorflow as tf
import numpy as np
#from keras.applications.resnet50 import ResNet50
from keras.applications import resnet
from keras.applications import ResNet50
#from keras_resnet.models import ResNet50
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LogNorm
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

IMAGE_PATH = "./en.jpg"

if __name__ == "__main__":
    model = tf.keras.applications.resnet50.ResNet50(
        weights="imagenet", include_top=True
    )

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()

    image_batch = np.expand_dims(img, axis = 0)
    processed_image = resnet.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)

    print(int(np.argmax(predictions)))

    data = ([img], None)

    tabby_cat_class_index = int(np.argmax(predictions))
    explainer = OcclusionSensitivity()
    # Compute Occlusion Sensitivity for patch_size 20
    grid = explainer.explain(data, model, tabby_cat_class_index, 16, cv2.COLORMAP_TURBO)


#    explainer.save(grid, ".", "occlusion_sensitivity_16.png")
    # Compute Occlusion Sensitivity for patch_size 10
    # grid = explainer.explain(data, model, tabby_cat_class_index, 10)
    # explainer.save(grid, ".", "occlusion_sensitivity_10.png")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def explanation_heatmap(data):
    '''
    Using heat-map to highlight the importance of each super-pixel for the model prediction
    '''
#    dict_heatmap = dict(exp.local_exp[exp_class])
#    heatmap = rgb2gray(data) 
    plt.title('Occlusion Analysis')
#    c = plt.imshow(data, cmap = 'YlOrRd', vmin  = -np.argmax(data), vmax = np.argmax(data))
#    plt.colorbar(c)
    plt.imshow(data)
    plt.show()
#    ax = sns.heatmap(data, cmap="YlOrRd")
    return

explanation_heatmap(grid)
exit