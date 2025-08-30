import os

# 数据集路径配置
widerface_root = "/Users/shhaofu/.cache/kagglehub/datasets/iamprateek/wider-face-a-face-detection-dataset/versions/1"
train_annot_file = os.path.join(widerface_root, "wider_face_annotations/wider_face_split/wider_face_train_bbx_gt.txt")

# 读取并打印前20行注释文件内容，以了解其格式
def check_annotation_format(annot_file, num_lines=20):
    try:
        with open(annot_file, 'r') as f:
            lines = f.readlines()
            print(f"Total lines in annotation file: {len(lines)}")
            print("\nFirst {num_lines} lines of the annotation file:")
            for i, line in enumerate(lines[:num_lines]):
                print(f"{i}: {line.strip()}")
    except Exception as e:
        print(f"Error reading annotation file: {e}")

if __name__ == "__main__":
    print(f"Checking annotation file: {train_annot_file}")
    check_annotation_format(train_annot_file)