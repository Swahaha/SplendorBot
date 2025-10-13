PYBIND11_DIR=/opt/homebrew/opt/pybind11/share/cmake/pybind11
BUILD_DIR=build
PYTHON_DIR=python
MODULE_NAME = splendor_game

# Default target
all: build

# Build the Python module
build:
	rm -rf $(BUILD_DIR)
	mkdir $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -Dpybind11_DIR=$(PYBIND11_DIR) ..
	cd $(BUILD_DIR) && make
	mv $(BUILD_DIR)/$(MODULE_NAME)*.so $(PYTHON_DIR)/

# Clean the build directory
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PYTHON_DIR)/$(MODULE_NAME)*.so