CFLAGS:= -Wall -I. -Wno-unknown-pragmas

all:
	@g++ $(CFLAGS) src/main.cpp -o run
