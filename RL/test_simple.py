"""
Minimal test to isolate the segfault
"""

from env_wrapper import SplendorEnv

print("Test 1: Create one environment")
env1 = SplendorEnv(num_players=2)
print("  Created env1 successfully")

print("\nTest 2: Reset env1")
obs, mask, info = env1.reset(seed=10000)
print(f"  Reset successful. Obs shape: {obs.shape}, legal actions: {mask.sum()}")

print("\nTest 3: Create second environment")
env2 = SplendorEnv(num_players=2)
print("  Created env2 successfully")

print("\nTest 4: Reset env2")
obs2, mask2, info2 = env2.reset(seed=20000)
print(f"  Reset successful. Obs shape: {obs2.shape}, legal actions: {mask2.sum()}")

print("\nTest 5: Reset env1 again")
obs3, mask3, info3 = env1.reset(seed=30000)
print(f"  Reset successful. Obs shape: {obs3.shape}, legal actions: {mask3.sum()}")

print("\nAll tests passed!")
