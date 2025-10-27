"""Minimal test - just import"""
import sys

print("Starting...")
sys.stdout.flush()

print("Step A: Basic imports...")
sys.stdout.flush()
import os
import numpy as np
print("  numpy OK")
sys.stdout.flush()

print("Step B: Set up paths...")
sys.stdout.flush()
rl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, rl_dir)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)

python_path = os.path.join(project_root, 'python')
build_path = os.path.join(project_root, 'build', 'Release')
sys.path.insert(0, python_path)
sys.path.insert(0, build_path)
print(f"  Working dir: {os.getcwd()}")
sys.stdout.flush()

print("Step C: Import splendor_game...")
sys.stdout.flush()
import splendor_game as sg
print("  splendor_game OK")
sys.stdout.flush()

print("Step D: Create game...")
sys.stdout.flush()
game = sg.SplendorGame(2, 42)
print("  Game created")
sys.stdout.flush()

print("Step E: Get state...")
sys.stdout.flush()
state = game.state_summary()
print("  State retrieved")
sys.stdout.flush()

print("\nDone!")
