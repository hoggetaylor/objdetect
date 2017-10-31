CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

% : %.cpp
	g++ -O2 -g $(CFLAGS) $(LIBS) -o $@ $<
