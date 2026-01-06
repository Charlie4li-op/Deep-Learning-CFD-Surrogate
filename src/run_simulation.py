import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
# Make sure this points to your NEW model
MODEL_PATH = "D:\AI_CFD_Project\cfd_unet_physics_v2.pth" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. MODEL ARCHITECTURE
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
# 2. HELPER FUNCTIONS
# ==========================================
def create_airfoil_mask(coords, resolution=128):
    x, y = np.meshgrid(np.linspace(-0.5, 1.5, resolution), np.linspace(-0.5, 0.5, resolution))
    grid_points = np.vstack((x.flatten(), y.flatten())).T
    path = Path(coords)
    mask = path.contains_points(grid_points).reshape(resolution, resolution)
    return mask.astype(float)

# NACA 4412 Coordinates (Cambered)
naca_4412 = [
    [1.0000, 0.0013], [0.9500, 0.0147], [0.9000, 0.0271], [0.8000, 0.0489],
    [0.7000, 0.0669], [0.6000, 0.0814], [0.5000, 0.0919], [0.4000, 0.0980],
    [0.3000, 0.0976], [0.2000, 0.0880], [0.1500, 0.0789], [0.1000, 0.0659],
    [0.0500, 0.0473], [0.0250, 0.0339], [0.0125, 0.0244], [0.0000, 0.0000],
    [0.0125, -0.0143], [0.0250, -0.0195], [0.0500, -0.0249], [0.1000, -0.0286],
    [0.1500, -0.0288], [0.2000, -0.0274], [0.3000, -0.0226], [0.4000, -0.0180],
    [0.5000, -0.0140], [0.6000, -0.0100], [0.7000, -0.0065], [0.8000, -0.0039],
    [0.9000, -0.0022], [0.9500, -0.0016], [1.0000, -0.0013]
]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
print("🚀 Loading Physics-Informed Model...")
model = SimpleUNet().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    print(f"❌ Error: {MODEL_PATH} not found.")
    exit()

# Generate Input
input_mask = create_airfoil_mask(naca_4412)
x_tensor = torch.from_numpy(input_mask).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    prediction = model(x_tensor).cpu().numpy()[0] 

# Apply Physics Mask (Hard Constraint)
physics_mask = 1.0 - input_mask
physics_mask_3d = np.repeat(physics_mask[np.newaxis, :, :], 3, axis=0)
final_output = prediction * physics_mask_3d

# Extract Velocity Components for Streamlines
u = final_output[0] # X-Velocity
v = final_output[1] # Y-Velocity
speed = np.sqrt(u**2 + v**2)

# ==========================================
# 4. VISUALIZE (The "Engineer's View")
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Velocity Magnitude (Heatmap)
# We use 'jet' colormap to highlight acceleration (Red)
im = ax[0].imshow(speed, origin='lower', cmap='jet', extent=[-0.5, 1.5, -0.5, 0.5])
ax[0].set_title("Velocity Magnitude (Physics-Informed)")
ax[0].set_xlabel("X Position")
ax[0].set_ylabel("Y Position")
plt.colorbar(im, ax=ax[0], label='Normalized Speed')

# Plot 2: Streamlines (The Flow Lines)
# We need a grid for streamplot
Y, X = np.mgrid[-0.5:0.5:128j, -0.5:1.5:128j]
# Note: Streamplot expects X and Y relative to array indices, but let's map it
x_grid = np.linspace(-0.5, 1.5, 128)
y_grid = np.linspace(-0.5, 0.5, 128)

# Streamplot requires 1D arrays for x/y grids or 2D meshes
# We overlay the wing mask first (Gray)
ax[1].imshow(input_mask, origin='lower', cmap='gray_r', extent=[-0.5, 1.5, -0.5, 0.5], alpha=0.3)

# Draw Streamlines
# density=1.5 makes it look detailed but not cluttered
strm = ax[1].streamplot(x_grid, y_grid, u, v, color=speed, cmap='autumn', density=1.5, linewidth=1)

ax[1].set_title("Flow Streamlines (Visualization of Continuity)")
ax[1].set_xlabel("X Position")
ax[1].set_xlim([-0.5, 1.5])
ax[1].set_ylim([-0.5, 0.5])

plt.tight_layout()
plt.show()

print("✅ Visualization Generated.")
print("Look for: Smooth lines curving around the wing (Continuity) and Red color over the top (Bernoulli Acceleration).")