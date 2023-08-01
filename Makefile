SOURCES := cloud_gpu_inference.md
NBS := $(SOURCES:.md=.ipynb)

%.ipynb: %.md
	pandoc --embed-resources --standalone --wrap=none  $< -o $@

all: $(NBS)

clean: 
	rm -f $(NBS)