# Building Splendor Game Module on Windows

You're on Windows with Python 3.12. Here are your options for compiling the C++ game module:

## Option 1: Use Pre-compiled Module (Easiest - Try This First!)

I'll create a simple Python-only wrapper that you can test with. Skip to Option 4 below.

## Option 2: Install Build Tools (Recommended for Long-term)

### Step 1: Install Visual Studio Build Tools

Download and install **Visual Studio Build Tools 2022** (free):
- Go to: https://visualstudio.microsoft.com/downloads/
- Scroll down to "All Downloads" → "Tools for Visual Studio"
- Download "Build Tools for Visual Studio 2022"
- Run installer and select:
  - ✓ Desktop development with C++
  - ✓ C++ CMake tools for Windows

This is ~7 GB download but gives you everything needed.

### Step 2: Install CMake

Option A - Via Python (easiest):
```bash
pip install cmake
```

Option B - Standalone installer:
- Download from: https://cmake.org/download/
- Get "Windows x64 Installer"
- During install, select "Add CMake to system PATH"

### Step 3: Install pybind11

```bash
pip install pybind11
```

### Step 4: Build the Module

Open **Developer Command Prompt for VS 2022** (or PowerShell):

```bash
cd C:\Users\swara\Desktop\Splendor\SplendorBot

# Clean old build
rmdir /s /q build
mkdir build
cd build

# Configure for Visual Studio 2022
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build Release version
cmake --build . --config Release

# The output will be in build\Release\splendor_game.*.pyd
```

The `.pyd` file is the compiled Python module!

### Step 5: Copy Module

```bash
# Copy to python directory
copy Release\splendor_game.*.pyd ..\python\

# Or use in place
cd ..\RL
# The env_wrapper.py will find it automatically
```

## Option 3: Use MinGW (Alternative Compiler)

If you don't want Visual Studio:

### Install MinGW
```bash
# Using Chocolatey (install chocolatey first from chocolatey.org)
choco install mingw

# Or download from: https://www.mingw-w64.org/
```

### Build with MinGW
```bash
cd C:\Users\swara\Desktop\Splendor\SplendorBot
rmdir /s /q build
mkdir build
cd build

# Configure with MinGW
cmake -G "MinGW Makefiles" ..

# Build
mingw32-make

# Output: splendor_game.*.pyd
```

## Option 4: Python-only Alternative (No Compilation Needed!)

Since compilation might be challenging, I can create a **pure Python implementation** of the Splendor game that works with your RL code. This will be slower but requires no compilation.

**Pros:**
- No compilation needed
- Works immediately
- Same interface as C++ version
- Good for development and testing

**Cons:**
- 5-10x slower than C++
- Still fast enough for RL training (just takes longer)

Would you like me to create this? It would take ~10 minutes.

## Option 5: Use WSL (Windows Subsystem for Linux)

If you have WSL2 installed:

```bash
# In WSL Ubuntu terminal
sudo apt update
sudo apt install build-essential cmake python3-dev python3-pip

pip3 install pybind11

cd /mnt/c/Users/swara/Desktop/Splendor/SplendorBot
mkdir -p build && cd build
cmake ..
make

# Copy .so file to Windows Python path
cp splendor_game.*.so ../python/
```

## Testing After Build

Once you have the module (either `.pyd` or `.so`):

```bash
# Test import
python -c "import sys; sys.path.append('python'); import splendor_game; print('Success!')"

# Run RL pipeline
cd RL
python main.py --mode test
```

## Troubleshooting

### Error: "pybind11 not found"
```bash
pip install pybind11
```

### Error: "CMake not found"
```bash
pip install cmake
# Or download installer from cmake.org
```

### Error: "No C++ compiler"
Install Visual Studio Build Tools (Option 2 above)

### Error: "Python.h not found"
Your Python installation is missing dev headers. Reinstall Python with "pip" and "development" options checked.

### Module loads but crashes
Version mismatch. Make sure you're using the same Python version (3.12) that you built with.

## My Recommendation

For you right now, I recommend **Option 4** (Python-only implementation) because:

1. No installation required ✓
2. Works immediately ✓
3. Same interface ✓
4. You can switch to C++ later for speed

Once you have everything working, you can install Visual Studio Build Tools and compile for 10x speedup.

**Would you like me to create the Python-only version?** Just say "yes" and I'll build it for you in a few minutes.
