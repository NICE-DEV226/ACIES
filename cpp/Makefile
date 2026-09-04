CXX = g++
CXXFLAGS = -std=c++17 -O3 -fPIC -Wall
LDFLAGS = -lm

SRCS = belief.cpp clarity_learner.cpp acies.cpp
OBJS = $(SRCS:.cpp=.o)
LIB = libacies.so

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)
	@echo "Built $(LIB)"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
