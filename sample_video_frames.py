import os
import cv2
import argparse
from tqdm import tqdm

def sample_frames(video_path, output_folder, k):
    """
    Sample k evenly spaced frames from a video and save them to the output folder.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to save the sampled frames
        k (int): Number of frames to sample
    """
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create video-specific output folder
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample
    if k >= total_frames:
        frame_indices = range(total_frames)
    else:
        frame_indices = [int(i * total_frames / k) for i in range(k)]
    
    # Sample and save frames
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(video_output_folder, f"frame_{i+1:03d}.png")
            cv2.imwrite(output_path, frame)
    
    cap.release()

def main():
    parser = argparse.ArgumentParser(description='Sample k frames from videos in a source folder')
    parser.add_argument('--source_folder', required=True,
                        help='Path to the source folder containing videos')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of frames to sample from each video')
    parser.add_argument('--output_folder', default='sampled_frames', 
                        help='Name of the output subfolder (default: sampled_frames)')
    
    args = parser.parse_args()
    
    # Create output folder path
    output_folder = os.path.join(args.source_folder, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all video files in the source folder
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(args.source_folder) 
                  if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {args.source_folder}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(args.source_folder, video_file)
        sample_frames(video_path, output_folder, args.k)
    
    print(f"Finished sampling {args.k} frames from each video")
    print(f"Frames saved to: {output_folder}")

if __name__ == "__main__":
    main() 