nvcc_options= -gencode arch=compute_30,code=sm_30 --compiler-options -Wall

all = ant_colony

ant_colony: ant_colony.cu
	nvcc ant_colony.cu $(nvcc_option) -o ant_colony

clean:
	rm ant_colony
