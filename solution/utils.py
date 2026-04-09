import torch.nn as nn
import torchvision
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class CoordMapNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.source_emb = nn.Embedding(2, 16)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512 + 512 + 16 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def sample_features(self, feat_map, xy_norm):
        grid = xy_norm * 2 - 1
        grid = grid.view(1, 1, -1, 2)
        return grid

    def forward(self, src_img, door2_img, source_id, xy_src, batch_idx):
        B = src_img.shape[0]

        src_fmap  = self.encoder(src_img)
        dst_fmap  = self.encoder(door2_img)

        f_dst = self.pool(dst_fmap).flatten(1)

        N = xy_src.shape[0]
        grid = xy_src * 2 - 1
        grid = grid.view(N, 1, 1, 2)

        src_fmap_per_pt = src_fmap[batch_idx]
        sampled = torch.nn.functional.grid_sample(src_fmap_per_pt, grid, mode='bilinear', align_corners=True)
        f_src_local = sampled.flatten(1)

        f_dst_per_pt = f_dst[batch_idx]

        s_emb = self.source_emb(source_id)[batch_idx]
        xy_feat = self.coord_mlp(xy_src)

        x = torch.cat([f_src_local, f_dst_per_pt, s_emb, xy_feat], dim=1)
        return self.head(x)