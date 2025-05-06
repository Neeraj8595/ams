import cv2
import numpy as np
import streamlit as st
import pandas as pd
import face_recognition
import os
from datetime import datetime
import time
from PIL import Image
import glob
import pickle
import shutil
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Advanced Attendance System",
    page_icon="🔍",
    layout="wide"
)

# Create necessary directories if they don't exist
if not os.path.exists('attendence-log'):
    os.makedirs('attendence-log')

if not os.path.exists('face_db'):
    os.makedirs('face_db')

if not os.path.exists('user_images'):
    os.makedirs('user_images')

# Path for storing user data and attendance logs
FACE_DB_PATH = 'face_db/face_encodings.pkl'
ATTENDANCE_LOG_PATH = 'attendence-log'
USER_IMAGES_PATH = 'user_images'

# Initialize or load face database
def load_face_db():
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {'names': [], 'roll_numbers': [], 'encodings': []}

def save_face_db(face_db):
    with open(FACE_DB_PATH, 'wb') as f:
        pickle.dump(face_db, f)

# Initialize the face database
face_db = load_face_db()

def create_attendance_log(name, roll_no):
    """Create or update attendance log for the day"""
    today = datetime.now().strftime('%Y-%m-%d')
    today_log_file = f"{ATTENDANCE_LOG_PATH}/{today}.csv"
    
    current_time = datetime.now().strftime('%H:%M:%S')
    
    if os.path.exists(today_log_file):
        attendance_df = pd.read_csv(today_log_file)
        # Check if the student has already been marked present
        if any(attendance_df['Roll_Number'] == roll_no):
            return False
    else:
        attendance_df = pd.DataFrame(columns=['Username', 'Roll_Number', 'Timestamp', 'Date'])
    
    # Add new attendance record
    new_record = pd.DataFrame({
        'Username': [name],
        'Roll_Number': [roll_no],
        'Timestamp': [current_time],
        'Date': [today]
    })
    
    updated_df = pd.concat([attendance_df, new_record], ignore_index=True)
    updated_df.to_csv(today_log_file, index=False)
    return True

def register_from_webcam():
    """Register a new user by capturing face from webcam"""
    st.subheader("Register User via Webcam")
    
    # Input fields for user information
    username = st.text_input("Username")
    roll_number = st.text_input("Roll Number")
    
    # Check if roll number already exists
    if roll_number in face_db['roll_numbers']:
        st.error("This roll number already exists. Please use a different roll number.")
        return
    
    # Capture face image for encoding
    st.write("Please look at the camera to capture your face.")
    capture_button = st.button("Capture Face")
    
    if capture_button:
        if not username or not roll_number:
            st.error("Please fill in both username and roll number")
            return
            
        cap = cv2.VideoCapture(0)
        st.text("Camera is starting... Please wait.")
        time.sleep(2)  # Give the camera time to initialize
        
        # Display placeholder for camera feed
        camera_placeholder = st.empty()
        
        # Countdown before capturing
        for i in range(3, 0, -1):
            camera_placeholder.text(f"Capturing in {i}...")
            ret, frame = cap.read()
            if ret:
                camera_placeholder.image(frame, channels="BGR", caption="Live Feed")
            time.sleep(1)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            st.error("Failed to capture image from camera. Please try again.")
            return
        
        # Display captured image
        camera_placeholder.image(frame, channels="BGR", caption="Captured Image")
        
        # Process the captured image for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if not face_locations:
            st.error("No face detected. Please try again with proper lighting and positioning.")
            return
        
        # If multiple faces are detected, use the largest one (assumed to be the closest)
        if len(face_locations) > 1:
            st.warning(f"Multiple faces detected. Using the most prominent one.")
            
            # Find the largest face by area
            largest_area = 0
            largest_face_idx = 0
            
            for i, (top, right, bottom, left) in enumerate(face_locations):
                area = (bottom - top) * (right - left)
                if area > largest_area:
                    largest_area = area
                    largest_face_idx = i
                    
            face_location = [face_locations[largest_face_idx]]
        else:
            face_location = face_locations
        
        # Extract face encoding
        face_encoding = face_recognition.face_encodings(rgb_frame, face_location)[0]
        
        # Save the image file
        user_image_path = f"{USER_IMAGES_PATH}/{username}_{roll_number}.jpg"
        cv2.imwrite(user_image_path, frame)
        
        # Save user data to face database
        face_db['names'].append(username)
        face_db['roll_numbers'].append(roll_number)
        face_db['encodings'].append(face_encoding)
        
        save_face_db(face_db)
        
        st.success(f"User {username} (Roll No: {roll_number}) registered successfully!")

def register_from_images():
    """Register users from uploaded images"""
    st.subheader("Register Users from Images")
    
    st.write("Upload images named in format: username_rollnumber.jpg")
    uploaded_files = st.file_uploader("Choose image files", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Images"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            successful_registrations = 0
            failed_registrations = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                progress = int((i / len(uploaded_files)) * 100)
                progress_bar.progress(progress)
                
                try:
                    # Parse filename
                    filename = uploaded_file.name
                    base_name = os.path.splitext(filename)[0]  # Remove extension
                    
                    if '_' not in base_name:
                        failed_registrations.append(f"{filename} (Invalid format - must be username_rollnumber)")
                        continue
                    
                    username, roll_number = base_name.split('_', 1)
                    
                    # Check if roll number already exists
                    if roll_number in face_db['roll_numbers']:
                        failed_registrations.append(f"{filename} (Roll number {roll_number} already exists)")
                        continue
                    
                    # Convert the uploaded file to an image and save it
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)
                    
                    # Save the image to the user_images folder
                    file_path = os.path.join(USER_IMAGES_PATH, filename)
                    image.save(file_path)
                    
                    # Process the image for face recognition
                    if len(image_np.shape) == 2:  # Grayscale
                        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                    elif image_np.shape[2] == 4:  # RGBA
                        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                    else:
                        rgb_image = image_np  # Already RGB
                    
                    face_locations = face_recognition.face_locations(rgb_image, model="hog")
                    
                    if not face_locations:
                        failed_registrations.append(f"{filename} (No face detected)")
                        continue
                    
                    # Use the largest face if multiple are detected
                    if len(face_locations) > 1:
                        # Find the largest face by area
                        largest_area = 0
                        largest_face_idx = 0
                        
                        for j, (top, right, bottom, left) in enumerate(face_locations):
                            area = (bottom - top) * (right - left)
                            if area > largest_area:
                                largest_area = area
                                largest_face_idx = j
                                
                        face_location = [face_locations[largest_face_idx]]
                    else:
                        face_location = face_locations
                    
                    # Extract face encoding
                    face_encoding = face_recognition.face_encodings(rgb_image, face_location)[0]
                    
                    # Save user data to face database
                    face_db['names'].append(username)
                    face_db['roll_numbers'].append(roll_number)
                    face_db['encodings'].append(face_encoding)
                    
                    successful_registrations += 1
                    status_placeholder.text(f"Processing: {filename}")
                
                except Exception as e:
                    failed_registrations.append(f"{uploaded_file.name} (Error: {str(e)})")
            
            # Save the updated face database
            save_face_db(face_db)
            
            progress_bar.progress(100)
            
            # Display results
            st.success(f"Successfully registered {successful_registrations} users!")
            
            if failed_registrations:
                st.error("The following registrations failed:")
                for failure in failed_registrations:
                    st.write(f"- {failure}")

def mark_attendance():
    """Mark attendance by recognizing faces"""
    st.subheader("Mark Attendance")
    
    if not face_db['encodings']:
        st.warning("No registered users found. Please register users first.")
        return
    
    # Extract face encodings from the database
    known_face_encodings = face_db['encodings']
    known_names = face_db['names']
    known_roll_numbers = face_db['roll_numbers']
    
    # Start camera
    start_button = st.button("Start Camera")
    
    if start_button:
        cap = cv2.VideoCapture(0)
        st.text("Camera is starting... Please wait.")
        time.sleep(2)  # Give the camera time to initialize
        
        if not cap.isOpened():
            st.error("Failed to open camera. Please check your camera connection.")
            return
            
        # Display placeholder for camera feed
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            status_placeholder.info("Looking for face... Please position yourself properly.")
            
            # Process frames for up to 20 seconds or until a face is recognized
            start_time = time.time()
            while time.time() - start_time < 20:
                ret, frame = cap.read()
                
                if not ret:
                    status_placeholder.error("Failed to capture frame. Please try again.")
                    break
                
                # Display the frame
                camera_placeholder.image(frame, channels="BGR", caption="Live Feed")
                
                # Process every 5th frame for efficiency
                if int(time.time() * 10) % 5 == 0:
                    # Convert to RGB for face_recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    
                    if face_locations:
                        # Get encodings of detected faces
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        for face_encoding in face_encodings:
                            # Compare with known faces
                            # Using lower tolerance (0.5 instead of 0.6) for higher precision
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                            
                            if True in matches:
                                # Find the matched user
                                match_index = matches.index(True)
                                matched_name = known_names[match_index]
                                matched_roll = known_roll_numbers[match_index]
                                
                                # Mark attendance
                                attendance_success = create_attendance_log(matched_name, matched_roll)
                                
                                if attendance_success:
                                    status_placeholder.success(f"✅ Attendance marked for {matched_name} (Roll No: {matched_roll})")
                                else:
                                    status_placeholder.warning(f"⚠️ {matched_name}, your attendance is already marked for today!")
                                
                                # Display the matched user with a rectangle around the face
                                for (top, right, bottom, left) in face_locations:
                                    # Draw rectangle
                                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                    
                                    # Add name label
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(frame, matched_name, (left + 6, bottom + 20), font, 0.5, (255, 255, 255), 1)
                                
                                camera_placeholder.image(frame, channels="BGR", caption="Recognition Complete")
                                time.sleep(3)  # Show the result for 3 seconds
                                cap.release()
                                return
                            else:
                                status_placeholder.warning("Face not recognized. Please register first.")
                                time.sleep(2)
                    
                # Brief pause to reduce CPU usage
                time.sleep(0.1)
            
            status_placeholder.error("Time limit exceeded. Please try again.")
                
        finally:
            cap.release()

def view_attendance_logs():
    """View attendance logs by date"""
    st.subheader("View Attendance Logs")
    
    # Get all available log files
    log_files = [f for f in os.listdir(ATTENDANCE_LOG_PATH) if f.endswith('.csv')]
    
    if not log_files:
        st.info("No attendance logs found.")
        return
    
    # Sort log files by date (newest first)
    log_files.sort(reverse=True)
    
    # Convert filenames to readable dates for the selectbox
    date_options = [f.split('.')[0] for f in log_files]
    
    selected_date = st.selectbox("Select Date:", date_options)
    
    if selected_date:
        log_file = f"{ATTENDANCE_LOG_PATH}/{selected_date}.csv"
        
        if os.path.exists(log_file):
            attendance_df = pd.read_csv(log_file)
            
            # Display attendance statistics
            total_present = len(attendance_df)
            st.write(f"**Total Present:** {total_present}")
            
            # Add search functionality
            search_query = st.text_input("Search by Username or Roll Number:")
            
            if search_query:
                filtered_df = attendance_df[
                    attendance_df['Username'].str.contains(search_query, case=False) | 
                    attendance_df['Roll_Number'].astype(str).str.contains(search_query, case=False)
                ]
                
                if filtered_df.empty:
                    st.info("No matching records found.")
                else:
                    st.dataframe(filtered_df)
            else:
                # Display full attendance table
                st.dataframe(attendance_df)
            
            # Option to download the log as CSV
            csv = attendance_df.to_csv(index=False)
            st.download_button(
                label="Download Attendance Log",
                data=csv,
                file_name=f"attendance_{selected_date}.csv",
                mime="text/csv"
            )

def view_registered_users():
    """View all registered users"""
    st.subheader("Registered Users")
    
    if not face_db['names']:
        st.info("No users registered yet.")
        return
    
    # Create dataframe of users
    users_df = pd.DataFrame({
        'Username': face_db['names'],
        'Roll_Number': face_db['roll_numbers']
    })
    
    # Add search functionality
    search_query = st.text_input("Search Users:")
    
    if search_query:
        filtered_df = users_df[
            users_df['Username'].str.contains(search_query, case=False) | 
            users_df['Roll_Number'].astype(str).str.contains(search_query, case=False)
        ]
        
        if filtered_df.empty:
            st.info("No matching users found.")
        else:
            st.dataframe(filtered_df)
    else:
        # Display all users
        st.dataframe(users_df)
    
    # Display user images if available
    st.subheader("User Images")
    
    # Get and display user images in a grid
    image_files = glob.glob(f"{USER_IMAGES_PATH}/*.jpg") + glob.glob(f"{USER_IMAGES_PATH}/*.jpeg") + glob.glob(f"{USER_IMAGES_PATH}/*.png")
    
    if not image_files:
        st.info("No user images available.")
        return
    
    # Filter images based on search query if provided
    if search_query:
        image_files = [img for img in image_files if search_query.lower() in os.path.basename(img).lower()]
        
        if not image_files:
            st.info("No matching user images found.")
            return
    
    # Display images in columns
    cols = st.columns(3)
    
    for i, image_file in enumerate(image_files):
        col_idx = i % 3
        with cols[col_idx]:
            image = Image.open(image_file)
            st.image(image, caption=os.path.basename(image_file).split('.')[0], width=200)

def main():
    """Main function to run the Streamlit app"""
    st.title("Advanced Face Recognition Attendance System")
    
    # Navigation menu with emojis
    menu = ["📊 Dashboard", "📸 Mark Attendance", "👤 Register via Webcam", 
           "📁 Register via Image Upload", "📋 View Attendance Logs", "👥 View Registered Users"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Display today's date
    st.sidebar.markdown(f"**Today's Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Display current time (updates when page refreshes)
    st.sidebar.markdown(f"**Current Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Show registered user count
    st.sidebar.markdown(f"**Registered Users:** {len(face_db['names'])}")
    
    # Show today's attendance count
    today = datetime.now().strftime('%Y-%m-%d')
    today_log_file = f"{ATTENDANCE_LOG_PATH}/{today}.csv"
    if os.path.exists(today_log_file):
        today_attendance = len(pd.read_csv(today_log_file))
        st.sidebar.markdown(f"**Today's Attendance:** {today_attendance}")
    else:
        st.sidebar.markdown("**Today's Attendance:** 0")
    
    # Handle menu choices
    if choice == "📊 Dashboard":
        st.subheader("Dashboard")
        
        # Display welcome message
        st.write("""
        # Welcome to the Advanced Face Recognition Attendance System!
        
        This system uses face recognition technology to efficiently track attendance.
        
        ## Key Features:
        - **Fast face recognition** using modern algorithms
        - **Two registration methods**: webcam or image uploads
        - **Daily attendance logs** with timestamps
        - **Easy attendance marking** with just a face scan
        
        ## Getting Started:
        1. First, register users via webcam or image upload
        2. Then, users can mark their attendance by facing the camera
        3. View and download attendance reports by date
        
        ## Need to add multiple users at once?
        - Prepare image files named in the format: username_rollnumber.jpg
        - Upload them through the "Register via Image Upload" section
        """)
        
        # Quick stats in three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Registered Users", value=len(face_db['names']))
        
        with col2:
            # Count log files (excluding today if no attendance)
            log_count = len([f for f in os.listdir(ATTENDANCE_LOG_PATH) if f.endswith('.csv')])
            st.metric(label="Days with Attendance", value=log_count)
        
        with col3:
            # Today's attendance
            if os.path.exists(today_log_file):
                today_attendance = len(pd.read_csv(today_log_file))
                st.metric(label="Today's Attendance", value=today_attendance)
            else:
                st.metric(label="Today's Attendance", value="0")
        
    elif choice == "📸 Mark Attendance":
        mark_attendance()
    elif choice == "👤 Register via Webcam":
        register_from_webcam()
    elif choice == "📁 Register via Image Upload":
        register_from_images()
    elif choice == "📋 View Attendance Logs":
        view_attendance_logs()
    elif choice == "👥 View Registered Users":
        view_registered_users()

if __name__ == "__main__":
    main()