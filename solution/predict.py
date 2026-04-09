import torch
import os
from PIL import Image
from utils import CoordMapNet, transform


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ORIG_W = 3840
ORIG_H = 2160


def load_model(weights_path):
    model = CoordMapNet().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def predict_points(model, src_path, dst_path, pts_src, source_id):
    src_t = transform(Image.open(src_path).convert('RGB')).unsqueeze(0).to(DEVICE)
    dst_t = transform(Image.open(dst_path).convert('RGB')).unsqueeze(0).to(DEVICE)

    xy_src = torch.tensor(pts_src, dtype=torch.float32)

    xy_src_norm = xy_src.clone()
    xy_src_norm[:, 0] /= ORIG_W
    xy_src_norm[:, 1] /= ORIG_H

    source_id = torch.tensor([source_id], device=DEVICE)
    batch_idx = torch.zeros(len(xy_src), dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pred_norm = model(
            src_t,
            dst_t,
            source_id,
            xy_src_norm.to(DEVICE),
            batch_idx
        )

    pred_px = pred_norm.cpu() * torch.tensor([ORIG_W, ORIG_H])
    return pred_px.numpy()


if __name__ == "__main__":
    model = load_model('runs/09_04_26/best.pt')

    src_path = 'data/test-task/val/camera_door2_2025-11-27_15-08-08/top/frame_000091.jpg'
    dst_path = 'data/test-task/val/camera_door2_2025-11-27_15-08-08/door2/frame_000091.jpg'

    pts_src = [
        [2500, 376.5],
        [3131.6, 703.3],
    ]

    source_id = 0  # 0 - top, 1 - bottom

    pred = predict_points(model, src_path, dst_path, pts_src, source_id)

    print(pred)