CC = g++
CFLAGS = -c -O2 -std=c++11
BIN = train_neuralnet

OBJS = \
train_neuralnet.o two_layer_net.o myfunc.o layer.o layer_affine.o layer_relu.o last_layer.o last_layer_softmaxwithloss.o

$(BIN): $(OBJS)
	$(CC) -g -o $(BIN) $(OBJS) 

.cpp.o:
	$(CC) $(CFLAGS) $*.cpp
clean:
	$(RM) $(BIN) $(OBJS)
