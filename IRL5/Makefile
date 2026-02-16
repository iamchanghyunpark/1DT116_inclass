TARGETS = grayscale_image pi matmult grayscale_image_serial grayscale_image_pipelined hello_world
SOURCES = $(shell echo *.cu) $(shell echo *.c)
OBJECTS = $(SOURCES:.cu=.o)
CC = nvcc
CFLAGS = -O2 -g -lcudart

.PHONY: all clean

all: $(TARGETS)

grayscale_image_pipelined : grayscale_image_pipelined.cu common.c ppm.c
	$(CC) $(CFLAGS) $^  -o $@

grayscale_image_serial : grayscale_image_serial.cu common.c ppm.c
	$(CC) $(CFLAGS) $^  -o $@

grayscale_image : grayscale_image.cu common.c ppm.c
	$(CC) $(CFLAGS) $^  -o $@

pi: pi.cu common.c
	$(CC) $(CFLAGS) $^  -o $@

matmult: matmult.cu
	$(CC) $(CFLAGS) $^  -o $@

hello_world: hello_world.cu
	$(CC) $(CFLAGS) $^  -o $@



clean:
	-rm $(TARGETS)

