import tensorflow as tf
import os
import json
import random
from dotenv import load_dotenv


load_dotenv()
# augment function
def augment(image, seed_left_right, seed_up_down, seed_brightness, seed_saturation, seed_contrast, seed_hue, jpeg_quality):
    image = tf.image.stateless_random_flip_left_right(image, seed=seed_left_right)
    image = tf.image.stateless_random_flip_up_down(image, seed=seed_up_down)
    image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed_brightness)
    image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed_saturation)
    image = tf.image.stateless_random_contrast(image, lower=0.7, upper=1.3, seed=seed_contrast)
    image = tf.image.stateless_random_hue(image, max_delta=0.3, seed=seed_hue)
    image = tf.image.encode_jpeg(image, quality=jpeg_quality)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

# original dataset folder path
folder_path = os.getenv('dataset_path')

# label file path
labels_file = os.getenv("labels_file")
# Load image labels from the JSON file
with open(labels_file, 'r') as f:
    image_labels = json.load(f)

# export path
augmented_folder_path = os.getenv("export_path")
os.makedirs(augmented_folder_path, exist_ok=True)

# Loop through the image files, apply data augmentation with unique seeds for each attribute, and save the augmented images with updated labels in the JSON file
for image_file in os.listdir(folder_path):
    if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Load the image
        image_path = os.path.join(folder_path, image_file)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image)

        # Get the image filename
        image_filename = os.path.basename(image_file)

        # Get the labels for the image from the JSON file
        image_label = image_labels.get(image_filename, {})

        # Generate unique seeds for each attribute for every image
        seed_left_right = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
        seed_up_down = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
        seed_brightness = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
        seed_saturation = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
        seed_contrast = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
        seed_hue = tf.random.uniform(shape=(2,), maxval=10000, dtype=tf.int32)
        jpeg_quality = random.randint(80, 95)  # Generate a random JPEG quality value

        # Apply data augmentation to the image with unique seeds for each attribute
        augmented_image = augment(image, seed_left_right, seed_up_down, seed_brightness, seed_saturation, seed_contrast, seed_hue, jpeg_quality)

        # Define the filename for the augmented image
        augmented_image_filename = f'augmented_images_150_{image_filename}'

        # Save the augmented image as a JPEG file in the augmented folder
        augmented_image_path = os.path.join(augmented_folder_path, augmented_image_filename)
        tf.io.write_file(augmented_image_path, tf.image.encode_jpeg(augmented_image))

        # Update the image labels with the augmented image's filename and labels
        augmented_image_label = {augmented_image_filename: image_label}
        image_labels.update(augmented_image_label)

        # Print the image label and the path where the augmented image is saved
        print(f"Image Label: {image_label}")
        print(f"Augmented image saved as {augmented_image_path}\n")

# Save the updated labels to the JSON file
with open(labels_file, 'w') as f:
    json.dump(image_labels, f, indent=4)

print("Updated JSON file with augmented image labels.")
