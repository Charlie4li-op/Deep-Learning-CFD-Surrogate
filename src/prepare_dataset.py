import airfrans as af
import numpy as np
import os
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
# Input Folder (Raw Data)
SOURCE_PATH = 'D:/AI_CFD_Project/airfrans_data/Dataset' 

# Output Folder (New Normalized Data)
SAVE_PATH = 'D:/AI_CFD_Project/processed_data_norm' 

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# ==========================================
# 1. SETUP
# ==========================================
print("Loading dataset metadata...")
dataset, names = af.dataset.load(root=SOURCE_PATH, task='scarce', train=True)

# Define Grid (128x128)
resolution = 128
grid_x, grid_y = np.mgrid[-0.5:1.5:complex(resolution), -0.5:0.5:complex(resolution)]
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

print(f"🚀 Processing {len(dataset)} simulations with NORMALIZATION...")

# ==========================================
# 2. PROCESSING LOOP
# ==========================================
for i in tqdm(range(len(dataset))):
    data = dataset[i]
    
    # Extract Raw Columns (x, y, u, v, p)
    x, y = data[:, 0], data[:, 1]
    u, v, p = data[:, 2], data[:, 3], data[:, 4]
    
    # --- ⚡ KEY CHANGE: NORMALIZATION ⚡ ---
    # Find the maximum speed in this specific simulation
    max_speed = np.max(np.sqrt(u**2 + v**2))
    
    # Avoid division by zero (just in case)
    if max_speed == 0: max_speed = 1.0
        
    # Divide everything by max_speed to get range 0.0 to 1.0
    u_norm = u / max_speed
    v_norm = v / max_speed
    p_norm = p / (max_speed**2) # Pressure scales with velocity squared
    
    points = np.c_[x, y]
    
    # --- A. Hole Punching (Distance Check) ---
    # We build the tree once per sim
    tree = cKDTree(points)
    dist, _ = tree.query(grid_points)
    hole_mask = (dist.reshape(grid_x.shape) > 0.005)
    
    # --- B. Interpolate Channels ---
    channels = []
    # We now loop through the NORMALIZED fields
    for field in [u_norm, v_norm, p_norm]:
        grid_field = griddata(points, field, (grid_x, grid_y), method='linear', fill_value=0)
        
        # Physics Logic: 
        # Inside the wing (hole) -> 0
        # Outside (background) -> 0 (or 1 if you prefer, but 0 is safer for now)
        grid_field[hole_mask] = 0 
        grid_field = np.nan_to_num(grid_field)
        
        channels.append(grid_field)
    
    # Stack them: (3, 128, 128)
    target_image = np.stack(channels, axis=0)
    
    # --- C. Create Input Mask ---
    input_mask = hole_mask.astype(float)
    input_mask = np.expand_dims(input_mask, axis=0)
    
    # --- D. Save to Disk ---
    save_name = os.path.join(SAVE_PATH, f"sim_{i}.npz")
    # We also save 'max_speed' so we can reverse the normalization later if needed!
    np.savez_compressed(save_name, x=input_mask, y=target_image, max_speed=max_speed)

print("\n✅ DONE! Normalized data saved to:", SAVE_PATH)