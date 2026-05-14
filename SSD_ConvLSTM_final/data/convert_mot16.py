import os
import json
import shutil
from pathlib import Path

MOT16_ROOT = '/home/felix/MOT16/train'
VID_ROOT = '/home/felix/VID'  # Output directory

def convert_mot16_to_vid():
    os.makedirs(VID_ROOT, exist_ok=True)
    train_data = {}

    # List MOT16 sequences (e.g., MOT16-02, MOT16-04, etc.)
    sequences = [d for d in os.listdir(MOT16_ROOT) if d.startswith('MOT16-')]

    print("Found sequences:", sequences)
    for seq in sequences:
        seq_path = Path(MOT16_ROOT) / seq
        img_dir = seq_path / 'img1'
        gt_file = seq_path / 'gt' / 'gt.txt'

        if not img_dir.exists() or not gt_file.exists():
            continue

        # Read ground truth
        gt_data = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                obj_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                if conf < 0.5:  # Skip low-confidence detections
                    continue
                if frame_id not in gt_data:
                    gt_data[frame_id] = []
                gt_data[frame_id].append([x, y, x+w, y+h])  # Convert to [xmin, ymin, xmax, ymax]

        # Create video folder in VID/train
        vid_seq_dir = Path(VID_ROOT) / 'train' / seq
        vid_img_dir = vid_seq_dir / 'img'
        vid_img_dir.mkdir(parents=True, exist_ok=True)

        # Copy and rename images
        frame_files = sorted(img_dir.glob('*.jpg'))
        gt_rects = []
        for i, img_file in enumerate(frame_files, start=1):
            new_name = f"{i:06d}.jpg"
            shutil.copy(img_file, vid_img_dir / new_name)
            # Get GT for this frame (assuming frame numbers match)
            frame_gt = gt_data.get(i, [])
            gt_rects.append(frame_gt)

        # Add to train.json
        train_data[seq] = {
            'name': 'person_' + seq,
            'image_files': [f"{i:06d}.jpg" for i in range(1, len(frame_files)+1)],
            'gt_rect': gt_rects
        }

    # Save train.json
    with open(Path(VID_ROOT) / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    print(f"Converted {len(train_data)} sequences to VID format at {VID_ROOT}")

if __name__ == '__main__':
    convert_mot16_to_vid()
