import os
import imageio
from PIL import Image
import cv2

def split_video_into_images(input_path, output_folder, frame_skip, max_height=720):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the video file
    reader = imageio.get_reader(input_path)

    # Get video metadata
    fps = reader.get_meta_data()['fps']

    # Set maximum height for resizing
    target_height = min(reader.get_meta_data()['size'][1], max_height)

    # Calculate corresponding width to maintain aspect ratio
    aspect_ratio = reader.get_meta_data()['size'][0] / reader.get_meta_data()['size'][1]
    target_width = int(aspect_ratio * target_height)

    # Iterate through frames, skipping every second frame
    for frame_number, frame_np in enumerate(reader):
        if frame_number % frame_skip == 0:
            # Convert the NumPy array to a PIL Image
            frame_pil = Image.fromarray(frame_np)

            # Resize the image to maintain aspect ratio and limit height to 720 pixels
            frame_pil = frame_pil.resize((target_width, target_height), Image.ANTIALIAS)

            # Save the resized image to the output folder
            image_filename = f"frame_{frame_number // frame_skip:04d}.png"
            image_path = os.path.join(output_folder, image_filename)
            frame_pil.save(image_path)

    print(f"Images saved to {output_folder}")
    


import os
from PIL import Image
from photo_cartoon import cartoonify
def copy_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Iterate through each image file
    for image_file in image_files:
        # Construct the full paths for input and output images
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)
        processed_img = cartoonify(cv2.imread(input_image_path))
        # Open the image using PIL
            # Save the processed image to the output folder
        cv2.imwrite(output_image_path,processed_img)

    print(f"Images processed and saved to {output_folder}")

def images_to_video(input_folder, output_video_path, fps=30):
    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the image files to maintain order
    image_files.sort()

    # Create a writer object to save the video
    writer = imageio.get_writer(output_video_path, fps=fps)

    # Iterate through each image file and add it to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = imageio.imread(image_path)
        writer.append_data(image)

    writer.close()
    print(f"Video created at {output_video_path}")

import shutil
def get_video_fps(video_path):
    try:
        # Open the video file
        reader = imageio.get_reader(video_path)

        # Get the frames per second (fps)
        fps = reader.get_meta_data()['fps']

        print(f"Frames per second (fps): {fps}")

        # Close the video file
        reader.close()

        return fps
    except Exception as e:
        print(f"Error getting fps: {e}")
        return None

# Example usage:
from moviepy.editor import VideoFileClip, AudioFileClip
def extract_audio(video_path, output_audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path, codec='aac')
    audio_clip.close()

def combine_audio_with_video(video_path, audio_path, output_video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # Set the video's audio to the extracted audio
    video_clip = video_clip.set_audio(audio_clip)

    # Write the combined video with audio
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    video_clip.close()
    audio_clip.close()

if __name__ == "__main__":
    input_video_path = "input/input.mov"
    output_folder = "output_images"
    cartoon_folder = "cartonized_images"
    output_audio_path = "temp_files/extracted_audio.aac"
    output_combined_video_path = "output/output.mp4"
    output_cartoon_video = "temp_files/cartoonized_video.mp4"
    fps = get_video_fps(input_video_path)
    fps_division = 2
    fps = fps/fps_division
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cartoon_folder, exist_ok=True)
    shutil.rmtree(output_folder)
    shutil.rmtree(cartoon_folder)


    split_video_into_images(input_video_path, output_folder,fps_division)
    copy_images(output_folder,cartoon_folder)
    images_to_video(cartoon_folder,output_cartoon_video,fps)
    extract_audio(input_video_path,output_audio_path)
    combine_audio_with_video(output_cartoon_video, output_audio_path, output_combined_video_path)


    

