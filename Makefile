# Define the binary name as a variable
TARGET = snakebyte_v5.2
SRC_FILE = ./src/$(TARGET).cpp
BIN_FILE = ./bin/$(TARGET)

# Compiler and flags
CXX = g++-11
CXXFLAGS = -g -Wall -Wextra -std=c++20 -lm -lpthread

# Main rule
all: $(BIN_FILE)

# Rule to build the target
$(BIN_FILE): $(SRC_FILE)
	$(CXX) $(CXXFLAGS) -o $(BIN_FILE) $(SRC_FILE)

# Create bin directory
bin/:
	mkdir -p bin/

# Clean rule
clean:
	rm -f $(BIN_FILE)

.PHONY: all clean
