# inference.py
import os
import json
import torch
import numpy as np
import zipfile
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as mask_utils
import skimage.io as sio
from model import get_model
from PIL import Image


# Load image ID mapping
with open('./data/test_image_name_to_ids.json', 'r') as f:
    image_id_mapping = {item['file_name']: item['id'] for item in json.load(f)}

# Load COCO V1 weights
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transform = weights.transforms()

# Dataset
class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_list = sorted([f for f in os.listdir(root_dir) if f.lower().endswith('.tif')])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_id)
        image = sio.imread(img_path)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        image = Image.fromarray(image)
        image = transform(image)

        return image, img_id

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = './data/test_release'
MODEL_PATH = './checkpoints/resnet50_new_ep80_maskrcnn.pth'

test_dataset = TestDataset(TEST_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = get_model(num_classes=5, model_type='resnet50')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

results = []
threshold = 0.0

with torch.no_grad():
    for images, img_ids in tqdm(test_loader):
        image = images[0].to(device)
        filename = os.path.splitext(os.path.basename(img_ids[0]))[0] + ".tif"
        coco_image_id = image_id_mapping.get(filename)
        outputs = model([image])[0]
        scores = outputs['scores']
        masks = outputs['masks']
        boxes = outputs['boxes']
        labels = outputs['labels']  # <-- NEW

        for i in range(len(scores)):
            if scores[i] >= threshold:
                mask = masks[i, 0].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                x1, y1, x2, y2 = boxes[i].cpu().tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]  # ← FIXED FORMAT

                results.append({
                    'image_id': coco_image_id,
                    'bbox': bbox,
                    'score': scores[i].item(),
                    'category_id': labels[i].item(),  # <-- FIXED
                    'segmentation': rle
                })


# 5. Save pred.json
json_path = os.path.join('./result', 'pred.json')
with open(json_path, 'w') as f:
    json.dump(results, f)
print(f"Saved prediction results to {json_path}")

# 6. Save pred.zip
zip_path = os.path.join('./result', 'pred.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(json_path, arcname='test-results.json')
print(f"Saved zipped prediction results to {zip_path}")

print("✅ test-results.json has been saved.")
