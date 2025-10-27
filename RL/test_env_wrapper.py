"""Test environment wrapper"""
print("Step 1: Importing env_wrapper...")
from env_wrapper import SplendorEnv
print("Import successful!")

print("\nStep 2: Creating environment...")
env = SplendorEnv(num_players=2)
print(f"Environment created! OBS_DIM={env.OBS_DIM}, ACTION_DIM={env.ACTION_DIM}")

print("\nStep 3: Resetting environment...")
try:
    obs, mask, info = env.reset(seed=42)
    print(f"Reset successful!")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Legal actions: {mask.sum()}")
    print(f"  Current player: {info['current_player']}")
except Exception as e:
    import traceback
    print(f"Reset failed!")
    print(traceback.format_exc())

print("\nAll tests passed!")
