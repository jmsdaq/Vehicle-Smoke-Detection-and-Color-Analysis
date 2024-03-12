from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import os
import datetime
import cv2
import csv
import webcolors
from ultralytics import YOLO
from sklearn.cluster import KMeans
import time
from flask_socketio import SocketIO

app = Flask(__name__)

BASE_DIR = r'C:\Users\hp\Downloads\FOD\Flask'  # Replace with your base directory
DOMINANT_COLOR_DIR = os.path.join(BASE_DIR, 'DominantColors')
uploaded_video_dir = os.path.join(BASE_DIR, 'uploaded_video')
os.makedirs(DOMINANT_COLOR_DIR, exist_ok=True)

PROCESSED_VID_DIR = os.path.join(BASE_DIR, 'processed_videos')
os.makedirs(PROCESSED_VID_DIR, exist_ok=True)

time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


@app.route('/')
def index():
    return render_template('index.html')

def closest_color(rgb):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb[0]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def determine_danger_level(color_name):
    color_name = color_name.lower()
   
    # Define lists of color shades for each danger level
    wispy_white_shades = ["wispy white", "light gray", "lightgrey", "silver"]  # Add other similar shades here
    black_shades = ["black", "charcoal"]
    blue_or_grey_shades = ["blue", "grey", "gray", "slate gray", "slate grey", "darkgray", "dimgray", "lightslategray"]
    white_shades = ["white", "snow", "ivory"]
   
    # Check and return the danger level and description
    if color_name in wispy_white_shades:
        return ("None", "Water vapor from condensation.")
    elif color_name in black_shades:
        return ("Medium", "Too rich fuel/air mixture.")
    elif color_name in blue_or_grey_shades:
        return ("Elevated", "Engine burning oil.")
    elif color_name in white_shades:            
        return ("High", "Coolant being burned.")
    else:
        return ("Unknown", "Unknown")
   
def store_results_to_csv(current_time, dominant_colors, danger_level, description):
    csv_filename = f"dominant_colors_{current_time}.csv"
    csv_path = os.path.join(DOMINANT_COLOR_DIR, csv_filename)

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["R", "G", "B", "Color Name", "Danger Level", "Description"])
            for color, level, desc in zip(dominant_colors, danger_level, description):
                name = closest_color(tuple(map(int, color)))
                csv_writer.writerow(list(color) + [name, level, desc])
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    # Visualizing dominant colors in 3D plot
    R = [color[0] for color in dominant_colors]
    G = [color[1] for color in dominant_colors]
    B = [color[2] for color in dominant_colors]

    # Visualizing dominant colors using a bar chart
    color_names = [closest_color(tuple(map(int, color))) for color in dominant_colors]

    # Count occurrences of each color
    color_counts = {}
    for name in color_names:
        color_counts[name] = color_counts.get(name, 0) + 1

    # Sorting the color names based on their count, in descending order
    sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x], reverse=True)
    # Determine the most dominant color based on the highest count
    most_dominant_color = sorted_colors[0]
    # Get danger level and description for this color
    danger_level, description = determine_danger_level(most_dominant_color)
    print(f"The most dominant color is {most_dominant_color}. Danger level: {danger_level}. Description: {description}")


    return csv_filename

def prepare_plot_data(dominant_colors):
    color_names = [closest_color(tuple(map(int, color))) for color in dominant_colors]


    color_counts = {}
    for name in color_names:
        color_counts[name] = color_counts.get(name, 0) + 1

    sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x], reverse=True)
    return sorted_colors, [color_counts[color] for color in sorted_colors]

def process_video_logic(video_path):
    ExImg = os.path.join(BASE_DIR, 'ExImg')
    run_dir = os.path.join(ExImg, f'run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)


    smoke_dir = os.path.join(run_dir, 'smoke')  # Define the directory for smoke images
    os.makedirs(smoke_dir, exist_ok=True)  # Create the directory if it doesn't exist

    vehicles_dir = os.path.join(run_dir, 'vehicles')  # Define the directory for vehicles
    os.makedirs(vehicles_dir, exist_ok=True)  # Create the directory if it doesn't exist

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    vid_filename = "processed_vid.mp4"
    video_path_out = os.path.join(PROCESSED_VID_DIR, vid_filename)

    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join(BASE_DIR, r'model\weights\best.pt')
    model = YOLO(model_path)
    threshold = 0.5
    counter = 0

    class_name_mapping = {
        0: "Smoke",
        1: "Vehicle"
    }


    dominant_colors = []
    kmeans = KMeans(n_clusters=1)  # One cluster for dominant color
    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                if class_id == 0:
                    color = (0, 0, 255)
                    save_path = smoke_dir
                elif class_id == 1:
                    color = (0, 255, 0)
                    save_path = vehicles_dir
                else:
                    color = (0, 255, 255)
                    continue  # We don't save images of other classes


                # Map class ID to name, create label with score, draw rectangle around object and add label to frame.
                class_name = class_name_mapping.get(class_id, results.names[int(class_id)].upper())
                label = f"{class_name} {score:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)


                # Extract and save the detected object image.
                detected_object = frame[int(y1):int(y2), int(x1):int(x2)]
                object_path = os.path.join(save_path, f'extracted_object_{counter}.png')
                cv2.imwrite(object_path, detected_object)


                # Extracting dominant color but only for the smoke class
                if class_id == 0:
                    img = detected_object  # Use the already extracted region
                    img = img.reshape((img.shape[0] * img.shape[1], 3))
                    kmeans.fit(img)
                   
                    # Append the center of the first cluster (which is essentially the dominant color
                    # after clustering) to the 'dominant_colors' list.
                    dominant_colors.append(kmeans.cluster_centers_[0])

                counter += 1

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()


    return dominant_colors

def determine_dominant_color(dominant_colors):
    color_names = [closest_color(tuple(map(int, color))) for color in dominant_colors]

    # Count occurrences of each color
    color_counts = {}
    for name in color_names:
        color_counts[name] = color_counts.get(name, 0) + 1

    # Sorting the color names based on their count, in descending order
    sorted_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x], reverse=True)

    # Determine the most dominant color based on the highest count
    most_dominant_color = sorted_colors[0]
    return most_dominant_color

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return "No video found!", 400


    uploaded_video = request.files['video']
    video_path = os.path.join(BASE_DIR, 'uploaded_video', uploaded_video.filename)
    uploaded_video.save(video_path)

    dominant_colors = process_video_logic(video_path)
    video_path_out = os.path.join(PROCESSED_VID_DIR, 'processed_vid.mp4')
    # Get danger level and description from dominant colors
    most_dominant_color = determine_dominant_color(dominant_colors)
   
    # Get danger level and description from the most dominant color
    danger_level, description = determine_danger_level(most_dominant_color)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = store_results_to_csv(current_time, dominant_colors, danger_level, description)
    colors, counts = prepare_plot_data(dominant_colors)


    return render_template('result.html', csv_filename=csv_filename, colors=colors, counts=counts,
                       levels=[danger_level], descriptions=[description], video=video_path_out)

@app.route('/video')
def video_route():
    return send_from_directory(BASE_DIR, 'processed_videos', 'processed_vid.mp4', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)