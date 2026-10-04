.PHONY: all build clean

# The directory where CMake will generate build files and binaries
BUILD_DIR = build

all: build

build:
	@echo "--- Configuring and building the project ---"
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

clean:
	@echo "--- Cleaning build directory ---"
	rm -rf $(BUILD_DIR)