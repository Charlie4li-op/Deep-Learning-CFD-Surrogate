import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
DATA_PATH = 'D:/AI_CFD_Project/processed_data_norm'
BATCH_SIZE = 16
LEARNING_RATE = 0.0005 
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 Training on device: {DEVICE}")

# ==========================================
# 1. PHYSICS-INFORMED LOSS (The Upgrade)
# ==========================================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, weight_mse=1.0, weight_cont=0.2, weight_smooth=0.1):
        super().__init__()
        self.w_mse = weight_mse
        self.w_cont = weight_cont       # Continuity (Conservation of Mass)
        self.w_smooth = weight_smooth   # Smoothness
        
        # Sobel Filters for Gradients (Keep on same device as model)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3).to(DEVICE)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3).to(DEVICE)

    def gradient(self, img):
        """ Calculates dx and dy of an image using convolution """
        # Padding=1 ensures output size matches input size
        dx = F.conv2d(img, self.sobel_x, padding=1)
        dy = F.conv2d(img, self.sobel_y, padding=1)
        return dx, dy

    def forward(self, pred, target, input_mask):
        """
        pred: (Batch, 3, H, W) -> [u, v, p]
        target: (Batch, 3, H, W)
        input_mask: (Batch, 1, H, W) -> 1=Wing, 0=Air
        """
        
        # 1. DATA LOSS (MSE) - Focus on FLUID only
        fluid_mask = 1.0 - input_mask
        mse_loss = F.mse_loss(pred * fluid_mask, target * fluid_mask)
        
        # 2. CONTINUITY LOSS (Physics)
        # Equation: du/dx + dv/dy = 0
        u_pred = pred[:, 0:1, :, :] # Channel 0 (Velocity X)
        v_pred = pred[:, 1:2, :, :] # Channel 1 (Velocity Y)
        
        du_dx, _ = self.gradient(u_pred)
        _, dv_dy = self.gradient(v_pred)
        
        # Divergence should be 0. Any value != 0 is a physics violation.
        divergence = du_dx + dv_dy
        continuity_loss = torch.mean((divergence * fluid_mask) ** 2)
        
        # 3. SMOOTHNESS LOSS (Regularization)
        # Penalize jagged noise in the velocity field
        u_dx, u_dy = self.gradient(u_pred)
        v_dx, v_dy = self.gradient(v_pred)
        smoothness_loss = torch.mean(u_dx**2 + u_dy**2 + v_dx**2 + v_dy**2)

        # TOTAL LOSS
        total_loss = (self.w_mse * mse_loss) + \
                     (self.w_cont * continuity_loss) + \
                     (self.w_smooth * smoothness_loss)
                     
        return total_loss

# ==========================================
# 2. DATASET (Standard)
# ==========================================
class CFDDataset(Dataset):
    def __init__(self, folder_path):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.from_numpy(data['x']).float() # Geometry Mask
        y = torch.from_numpy(data['y']).float() # Target Fields
        return x, y

# ==========================================
# 3. MODEL (U-Net)
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(1, 64); self.pool1 = nn.MaxPool2d(2) 
        self.enc2 = self.conv_block(64, 128); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(256, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1); e2 = self.enc2(p1); p2 = self.pool2(e2); e3 = self.enc3(p2); p3 = self.pool3(e3); b = self.bottleneck(p3)
        d3 = self.up3(b); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.final(d1)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
dataset = CFDDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimpleUNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# USE NEW PHYSICS LOSS
criterion = PhysicsInformedLoss(weight_mse=1.0, weight_cont=0.2, weight_smooth=0.1).to(DEVICE)

print(f"🔥 Starting Physics-Informed Training...")

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(x)
        
        # Calculate Physics Loss (Needs x, pred, and y)
        loss = criterion(pred, y, x)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Physics Loss: {avg_loss:.6f}")

# Save
torch.save(model.state_dict(), "cfd_unet_physics_v2.pth")
print("💾 Saved physics-aware model as 'cfd_unet_physics_v2.pth'")

plt.plot(loss_history)
plt.title("Physics-Informed Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()