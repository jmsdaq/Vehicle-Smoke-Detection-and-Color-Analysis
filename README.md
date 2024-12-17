# YOLOv8 Video Object Detection with K-Means Clustering for Dominant Color Extraction

## Overview
This project implements a Flask-based web application that leverages YOLOv8 for video object detection and K-Means clustering to extract the most dominant color from detected objects. The application is tailored for scenarios such as analyzing smoke emission levels from vehicles, determining the potential risk based on the smoke's color, and providing intuitive visualizations and results in real time.

## Features
- **Object Detection:** Utilizes YOLOv8 for detecting objects in uploaded videos.
- **Color Analysis:** Applies K-Means clustering to extract dominant colors from detected regions.
- **Danger Level Assessment:** Associates detected colors with predefined danger levels and provides descriptive insights.
- **CSV Export:** Saves results (colors, danger levels, and descriptions) into a CSV file.
- **Video Processing:** Generates annotated videos with bounding boxes and labels for detected objects.
- **Web Interface:** User-friendly interface for uploading videos and viewing results.

## Prerequisites
Before setting up the project, ensure the following dependencies and tools are available:

- **Python 3.8+**
- Flask
- Flask-SocketIO 
- OpenCV 
- Scikit-learn 
- Webcolors 
- Ultralytics YOLOv8

Ensure you have YOLOv8 model weights (`best.pt`) in the `model/weights/` directory.

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/yolov8-dominant-color.git
   cd yolov8-dominant-color
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Directories:**
   Ensure the following directories exist:
   - `DominantColors/`
   - `uploaded_video/`
   - `processed_videos/`
   - `model/weights/` (place your YOLOv8 weights here)

## Usage
1. **Run the Flask App:**
   ```bash
   python app.py
   ```

2. **Upload a Video:**
   - Navigate to `http://127.0.0.1:5000` in your web browser.
   - Upload a video file for processing.

3. **View Results:**
   - Processed videos will be available in the `processed_videos/` directory.
   - Download the CSV report for dominant colors and danger level analysis.

## How It Works
1. **Video Upload:**
   The user uploads a video through the Flask web interface.

2. **Object Detection:**
   - YOLOv8 detects objects such as smoke and vehicles.
   - Bounding boxes are drawn, and detected objects are saved as images.

3. **Color Extraction:**
   - For detected smoke regions, K-Means clustering identifies the most dominant color.
   - The closest color name is determined using the Webcolors library.

4. **Danger Level Assessment:**
   - The dominant color is mapped to a predefined danger level (e.g., "None," "Medium," "High").
   - Descriptions provide insights into the cause (e.g., "Coolant being burned").

5. **Visualization and Output:**
   - Processed videos are annotated and saved.
   - Results are compiled into a CSV file, including color names, danger levels, and descriptions.

6. **Result Display:**
   - Results (video, dominant colors, and analysis) are shown on the web interface.

## **Dataset Information**  
This project uses a labeled dataset of vehicles and smoke from [Roboflow](https://roboflow.com). 

### **Contributors**  
This project was a collaborative effort:  

- **James Daquioag (Project Head/Engineer)**: Led the project, designed the architecture, and implemented YOLOv8 integration.  
- **Gerald Serrano (Developer/Tester)**: Assisted with coding and application testing.  
- **Lance Kim Formales (Tester/Documentation)**: Focused on testing and preparing documentation.  

### **License**  
This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.  

### **Contribution**  
Contributions are welcome! Feel free to submit issues or pull requests to improve this project.  
