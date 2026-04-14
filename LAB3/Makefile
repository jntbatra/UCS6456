# Compiler
CXX = g++

# Compiler flags
CXXFLAGS_SEQ = -std=c++17 -Wall -O2
CXXFLAGS_PAR = -std=c++17 -Wall -O2 -fopenmp
CXXFLAGS_OPT = -std=c++17 -Wall -O3 -fopenmp -mavx2 -mfma -march=native -funroll-loops

# Target executable
TARGET = correlate_bench

# Source files
SOURCES = main.cpp correlate.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Header files
HEADERS =

# Default target — optimized build
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS_OPT) -o $(TARGET) $(OBJECTS)

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_OPT) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run with custom size
run-large: $(TARGET)
	./$(TARGET) 2000 1000

# Phony targets
.PHONY: all clean run run-large
