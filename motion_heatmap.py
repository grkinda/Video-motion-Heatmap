import cv2
import numpy as np
from collections import deque
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import matplotlib.pyplot as plt
from datetime import timedelta
from queue import Queue
import threading
import json

def process_frame_batch(frames_batch, prev_gray, threshold):
    """
    Process a batch of frames for motion detection
    """
    results = []
    for frame in frames_batch:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        _, motion_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        motion_percentage = (np.sum(motion_mask) / 255) / (motion_mask.shape[0] * motion_mask.shape[1]) * 100
        results.append((motion_mask, curr_gray, motion_percentage))
        prev_gray = curr_gray.copy()
    return results

def create_motion_heatmap(video_path, output_path, buffer_size=64, threshold=25):
    """
    Creates a motion heatmap from a video file with CPU optimizations.
    """
    print("\n🔍 Starting video processing...")
    
    # Check if input video exists
    if not os.path.exists(video_path):
        print(f"Error: Input video file '{video_path}' not found!")
        return False

    print("  - Opening video file...")
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"  - Video properties:")
    print(f"    Resolution: {width}x{height}")
    print(f"    FPS: {fps}")
    print(f"    Total frames: {total_frames}")
    print(f"    Duration: {duration:.1f} seconds")
    
    print("  - Initializing video writer...")
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize buffer for motion frames
    motion_buffer = deque(maxlen=buffer_size)
    
    # Initialize lists for motion activity graph
    motion_percentages = []
    timestamps = []
    
    print("  - Reading first frame...")
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        return False
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frames")
    
    # Initialize thread pool for parallel processing
    num_threads = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    print(f"✨ Using {num_threads} CPU threads for processing")
    
    # Create a queue for processed frames
    frame_queue = Queue(maxsize=num_threads * 2)
    
    # Function to read frames
    def read_frames():
        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((frame_count, frame))
            frame_count += 1
        frame_queue.put(None)  # Signal end of frames
    
    print("  - Starting frame reading thread...")
    # Start frame reading thread
    reader_thread = threading.Thread(target=read_frames)
    reader_thread.start()
    
    start_time = time.time()
    processed_count = 0
    batch_size = 4  # Process 4 frames at a time
    
    print("  - Starting frame processing...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while processed_count < total_frames:
            # Collect a batch of frames
            frames_batch = []
            frame_indices = []
            
            # Try to get frames for the batch
            for _ in range(batch_size):
                if processed_count >= total_frames:
                    break
                try:
                    frame_data = frame_queue.get(timeout=5.0)  # Add timeout to prevent hanging
                    if frame_data is None:
                        break
                    frame_idx, frame = frame_data
                    frames_batch.append(frame)
                    frame_indices.append(frame_idx)
                except Exception as e:
                    print(f"\n⚠️ Error getting frame: {str(e)}")
                    break
            
            if not frames_batch:
                if processed_count < total_frames:
                    print("\n⚠️ No more frames available, but processing not complete")
                    break
                break
            
            # Process the batch
            future = executor.submit(process_frame_batch, frames_batch, prev_gray, threshold)
            results = future.result()
            
            # Process results
            for (motion_mask, curr_gray, motion_percentage), frame_idx in zip(results, frame_indices):
                # Store motion data for graph
                motion_percentages.append(motion_percentage)
                timestamps.append(frame_idx / fps)
                
                # Add motion mask to buffer
                motion_buffer.append(motion_mask)
                
                # Create heatmap
                heatmap = np.zeros((height, width), dtype=np.float32)
                for mask in motion_buffer:
                    heatmap += mask.astype(np.float32) / buffer_size
                
                # Normalize heatmap
                heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap = heatmap.astype(np.uint8)
                
                # Apply colormap to heatmap
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Blend original frame with heatmap
                alpha = 0.7
                output = cv2.addWeighted(frames_batch[frame_indices.index(frame_idx)], 1-alpha, heatmap_colored, alpha, 0)
                
                # Write frame
                out.write(output)
                
                # Update previous frame
                prev_gray = curr_gray.copy()
                
                # Update progress
                processed_count += 1
                pbar.update(1)
                
                # Calculate and display estimated time remaining
                elapsed_time = time.time() - start_time
                fps_processed = processed_count / elapsed_time
                remaining_frames = total_frames - processed_count
                eta = remaining_frames / fps_processed if fps_processed > 0 else 0
                pbar.set_postfix({'ETA': f'{eta:.1f}s', 'FPS': f'{fps_processed:.1f}'})
    
    # Ensure progress bar is complete
    pbar.n = total_frames
    pbar.refresh()
    pbar.close()
    
    print("\n✨ Video processing complete!")
    
    print("  - Saving motion data...")
    # Save motion data to JSON file
    data_path = output_path.rsplit('.', 1)[0] + '_data.json'
    motion_data = {
        'timestamps': timestamps,
        'motion_percentages': motion_percentages,
        'fps': fps,
        'total_frames': total_frames
    }
    
    try:
        with open(data_path, 'w') as f:
            json.dump(motion_data, f)
        print(f"✨ Motion data saved to: {data_path}")
    except Exception as e:
        print(f"⚠️ Could not save motion data: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
    
    print("  - Releasing resources...")
    # Release resources
    cap.release()
    out.release()
    
    print("✅ Video processing and data saving complete!")
    return True

def create_motion_graph(data_path, output_path):
    """
    Creates a motion activity graph from saved data.
    """
    print("\n📊 Starting graph creation...")
    try:
        print("  - Loading motion data...")
        # Load motion data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        timestamps = data['timestamps']
        motion_percentages = data['motion_percentages']
        
        print(f"  - Loaded {len(timestamps)} data points")
        print("  - Creating figure...")
        plt.figure(figsize=(12, 6))
        
        print("  - Setting style...")
        plt.style.use('default')
        
        print("  - Plotting data...")
        print(f"    Data points: {len(timestamps)} timestamps, {len(motion_percentages)} motion values")
        print(f"    Time range: {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s")
        print(f"    Motion range: {min(motion_percentages):.1f}% to {max(motion_percentages):.1f}%")
        
        plt.plot(timestamps, motion_percentages, color='#FF6B6B', linewidth=1.5)
        
        print("  - Adding labels and formatting...")
        plt.title('Motion Activity Over Time', fontsize=14, pad=20)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Motion Activity (%)', fontsize=12)
        
        # Set consistent y-axis scale from 0 to 100%
        plt.ylim(0, 100)
        
        # Add grid with consistent intervals
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        plt.yticks(np.arange(0, 101, 10))  # Show ticks every 10%
        
        def format_time(x, pos):
            return str(timedelta(seconds=int(x)))
        
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
        
        print("  - Adjusting layout...")
        plt.tight_layout()
        
        print("  - Saving graph...")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✨ Motion activity graph saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not create motion activity graph: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run the motion heatmap generator
    """
    print("\n🚀 Starting Motion Heatmap Generator...")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    print("Welcome to the Motion Heatmap Generator! (＾▽＾)")
    print("\nYour video files are in the 'Videos' folder.")
    print("The processed video will be saved in the 'output' folder.")
    
    # Get input video path
    video_path = input("\nEnter the name of your video file (e.g., 'temoin.mp4'): ")
    video_path = os.path.join("Videos", video_path)
    
    # Generate output paths
    output_filename = f"heatmap_{os.path.basename(video_path)}"
    output_path = os.path.join("output", output_filename)
    data_path = output_path.rsplit('.', 1)[0] + '_data.json'
    graph_path = output_path.rsplit('.', 1)[0] + '_activity.png'
    
    print(f"\n📁 Output paths:")
    print(f"   Video: {output_path}")
    print(f"   Data: {data_path}")
    print(f"   Graph: {graph_path}")
    
    # Get user preferences
    try:
        buffer_size = int(input("\nEnter buffer size (default: 64, higher = longer trails): ") or "64")
        threshold = int(input("Enter motion threshold (default: 25, lower = more sensitive): ") or "25")
    except ValueError:
        print("Using default values...")
        buffer_size = 64
        threshold = 25
    
    print("\n🎬 Starting video processing...")
    # Process the video
    success = create_motion_heatmap(
        video_path=video_path,
        output_path=output_path,
        buffer_size=buffer_size,
        threshold=threshold
    )
    
    if success:
        print(f"\n✨ Success! Your heatmap video has been saved to: {output_path}")
        
        print("\n📊 Starting graph creation...")
        # Create the motion activity graph
        if create_motion_graph(data_path, graph_path):
            print(f"✨ Motion activity graph has been saved to: {graph_path}")
        else:
            print("⚠️ Graph creation failed, but the video was processed successfully.")
    else:
        print("\n❌ There was an error processing the video. Please check the input file and try again.")

if __name__ == "__main__":
    main() 