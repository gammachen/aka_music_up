import kagglehub

# Download latest version
path = kagglehub.dataset_download("iamprateek/wider-face-a-face-detection-dataset")

print("Path to dataset files:", path)

'''
(translate-env) (base) shhaofu@shhaofudeMacBook-Pro aka_music % /opt/anaconda3/envs/translate-env/
bin/python /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/download_wider_face_d
atasets.py
Downloading from https://www.kaggle.com/api/v1/datasets/download/iamprateek/wider-face-a-face-detection-dataset?dataset_version_number=1...
100%|████████████████████████████████████████████████████████| 3.43G/3.43G [10:36<00:00, 5.78MB/s]
Extracting files...
Path to dataset files: /Users/shhaofu/.cache/kagglehub/datasets/iamprateek/wider-face-a-face-detection-dataset/versions/1
'''