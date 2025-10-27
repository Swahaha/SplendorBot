"""Test CSV parsing directly"""
import os

# Change to project root
os.chdir("..")

# Read cards.csv to see if header is being parsed
print("Reading data/cards.csv...")
with open("data/cards.csv", "r") as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(f"First line (header): {lines[0].strip()}")
    print(f"Second line (first card): {lines[1].strip()}")

print("\nExpected: Header should be skipped during parsing")
print("The C++ parser should skip the first line starting with 'Tier,Color,...'")
