import os

file_list = ['C:/Users/chowd/Desktop/Pose detection/image1.jpg', 'C:/Users/chowd/Desktop/Pose detection/image2.jpg']  # Update with your actual paths

for file in file_list:
    if os.path.exists(file):
        print(f"File exists: {file}")
    else:
        print(f"File does not exist: {file}")
