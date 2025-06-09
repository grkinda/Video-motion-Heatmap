# Video Motion Heatmap Generator (＾▽＾)

This Python script creates beautiful motion heatmaps from your videos! It visualizes movement patterns by creating a colorful overlay that shows where motion has occurred.

## Features ✨

- Creates motion heatmaps from any video file
- Customizable motion sensitivity
- Adjustable trail length
- Progress tracking
- User-friendly interface

## Installation 🚀

1. Make sure you have Python 3.7+ installed
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Use 📝

1. Place your video file in the `input` folder
2. Run the script:
```bash
python motion_heatmap.py
```
3. Follow the prompts to:
   - Enter your video filename (make sure to include the extension of the video for example: "Video.mp4")
   - Set the buffer size (how long motion trails last)
   - Set the motion threshold (how sensitive to movement)
4. Find your processed video in the `output` folder

## Tips 💡

- Higher buffer size = longer motion trails
- Lower threshold = more sensitive to movement
- Default values work well for most videos:
  - Buffer size: 64
  - Threshold: 25

## Example Output 🎥

The output video will show:
- Original video with a colorful overlay
- Red/yellow areas indicate recent motion
- Blue/purple areas indicate older motion
- The intensity of the color shows how much motion occurred

## Troubleshooting 🔧

If you encounter any issues:
1. Make sure your video file is in the correct format (MP4 recommended)
2. Check that the video file is in the `input` folder
3. Ensure you have enough disk space for the output
4. Try adjusting the threshold if the motion detection is too sensitive/not sensitive enough

## License 📄

This project is open source and available under the MIT License. 
