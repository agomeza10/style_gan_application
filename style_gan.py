"""
This module applies styleGAN to a video.
Extracting the frames from the video, applying styleGAN to each frame
and then generating a new video with the styled frames.
"""

# libraries
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# video


def extract_frames(video_path: str, destination_path: str) -> None:
    video = cv2.VideoCapture(video_path)
    currentframe = 0
    while True:
        has_frame, frame = video.read()
        if has_frame:
            name = f"{destination_path}/frame{currentframe}.jpg"
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def load_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image[tf.newaxis, :]
    return image


def process_and_write_frames(
    model, style_image_path: str, original_frames_path: str, destination_path: str
) -> None:
    style_image = load_image(style_image_path)
    original_frames = os.listdir(original_frames_path)
    for frame in range(len(original_frames)):
        content_image = load_image(f"{original_frames_path}/frame{frame}.jpg")
        stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
        cv2.imwrite(
            f"{destination_path}/generated_frame{frame}.jpg",
            cv2.cvtColor(np.squeeze(stylized_image) * 255, cv2.COLOR_BGR2RGB),
        )


def generate_video_from_styled_frames(styled_frames_path: str, styled_video_name: str) -> None:
    styled_frames = os.listdir(styled_frames_path)
    image_array = []
    for generated_frame in styled_frames:
        image = cv2.imread(f"{styled_frames_path}/{generated_frame}")
        height, width, _ = image.shape
        size = (width, height)
        image_array.append(image)

    output = cv2.VideoWriter(f"{styled_video_name}.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 15, size)

    for i in range(len(image_array)):
        output.write(image_array[i])
    output.release()


def main():
    original_video_path = "original_video/1_fogata_original_video.mp4"
    original_frames_destination_path = "original_video/1_fogata_original_frames"
    extract_frames(original_video_path, original_frames_destination_path)

    # pre-trained styleGAN
    model = hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

    style_image_path = "style_images/4_mariposa_style.jpeg"
    styled_frames_destination_path = "generated_styled_frames/4_mariposa_styled_frames"

    process_and_write_frames(
        model,
        style_image_path,
        original_frames_destination_path,
        styled_frames_destination_path,
    )

    styled_video_name = "output_videos/4_mariposa_styled_video"
    generate_video_from_styled_frames(styled_frames_destination_path, styled_video_name)

main()
