import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import os
import io
import uuid
import sys
import yaml
import traceback
import cv2
from PIL import Image, ImageEnhance
with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
from flask import Flask, render_template, make_response, flash
import flask
from PIL import Image
import numpy as np
import skvideo.io

from cartoonize import WB_Cartoonize

wb_cartoonizer = WB_Cartoonize(os.path.abspath("saved_models/"), opts['gpu'])


def cartoonify(frame):
    cartoon_image = wb_cartoonizer.infer(frame)
    cartoon_image = wb_cartoonizer.infer(cartoon_image)
    # Convert the image to RGB format (PIL expects RGB)
    original_image_rgb = cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2RGB)

    # Convert the original image to a PIL Image
    original_pil_image = Image.fromarray(original_image_rgb)

    # Increase color saturation
    saturation_factor = 1.5  # You can adjust this value as needed
    enhancer = ImageEnhance.Color(original_pil_image)
    saturated_pil_image = enhancer.enhance(saturation_factor)

    # Increase contrast
    contrast_factor = 2  # You can adjust this value as needed
    enhancer = ImageEnhance.Contrast(saturated_pil_image)
    contrasted_pil_image = enhancer.enhance(contrast_factor)

    # Convert the contrasted image to a NumPy array
    contrasted_image = cv2.cvtColor(np.array(contrasted_pil_image), cv2.COLOR_RGB2BGR)

    #cartoon_image = wb_cartoonizer.infer(cartoon_image)
    #cv2.imwrite("out.jpg", cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
    return contrasted_image


#image = cv2.imread("input.png")
#cartoonify(image)

def enhance_image(input_path, output_path):
    # Read the image using OpenCV
    original_image = cv2.imread(input_path)

    # Convert the image to RGB format (PIL expects RGB)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Convert the original image to a PIL Image
    original_pil_image = Image.fromarray(original_image_rgb)

    # Increase color saturation
    saturation_factor = 1.5  # You can adjust this value as needed
    enhancer = ImageEnhance.Color(original_pil_image)
    saturated_pil_image = enhancer.enhance(saturation_factor)

    # Increase contrast
    contrast_factor = 2  # You can adjust this value as needed
    enhancer = ImageEnhance.Contrast(saturated_pil_image)
    contrasted_pil_image = enhancer.enhance(contrast_factor)

    # Convert the contrasted image to a NumPy array
    contrasted_image = cv2.cvtColor(np.array(contrasted_pil_image), cv2.COLOR_RGB2BGR)

    # Enhance edges using the Canny edge detection algorithm
    #edges = cv2.Canny(contrasted_image, 100, 200)

    # Convert the edges image to RGB format
    #edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Combine the original image with enhanced edges
    #enhanced_image = cv2.addWeighted(contrasted_image, 0.8, edges_rgb, 0.2, 0)

    # Convert the enhanced image to a PIL Image
    enhanced_pil_image = Image.fromarray(contrasted_image)

    # Save the enhanced image
    enhanced_pil_image.save(output_path)


if __name__ == "__main__":
    input_image_path = "out.jpg"
    output_image_path = "test7.jpg"

    enhance_image(input_image_path, output_image_path)