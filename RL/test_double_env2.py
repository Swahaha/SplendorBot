"""
Test creating two environments WITHOUT resetting the first
"""

from env_wrapper import SplendorEnv

print("Test 1: Create first environment")
env1 = SplendorEnv(num_players=2)
print("  Created env1")

print("\nTest 2: Create second environment")
env2 = SplendorEnv(num_players=2)
print("  Created env2")

print("\nTest 3: Reset first environment")
obs1, mask1, info1 = env1.reset(seed=0)
print(f"  Reset env1 successful. Legal actions: {mask1.sum()}")

print("\nTest 4: Reset second environment")
obs2, mask2, info2 = env2.reset(seed=10000)
print(f"  Reset env2 successful. Legal actions: {mask2.sum()}")

print("\nAll tests passed!")
