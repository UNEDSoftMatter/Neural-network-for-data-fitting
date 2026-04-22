NVCC = nvcc
CFLAGS = -O3 --std=c++14 -arch=sm_75 --relocatable-device-code=true # sm_86 depends on your GPU
# -g -G
#beta: c-1-6 -> sm_86
#beta: c-1-2 -> sm_61
#dseta: c-4-0 -> sm_61
#dseta: c-4-2 -> sm_86
#portatil: sm_75

# Source files
HEADERS = config.h class_system.h kernel_functions.h atomicAdd_double.cuh
CUDA_SOURCES = main.cu kernel_declare_constants.cu kernel_print_constants.cu kernel_forward_propagation.cu kernel_activation_function.cu kernel_der_activation_function.cu kernel_back_propagation.cu kernel_average_gradients.cu kernel_update_ADAM_hidden.cu kernel_update_ADAM_out.cu kernel_loss_function_to_zero.cu kernel_loss_function.cu system_print_loss_function.cu system_print_approximation_function.cu kernel_average_grad_W_hidden.cu kernel_average_grad_W_hidden_layer0.cu kernel_average_grad_W_out.cu kernel_average_grad_b_hidden.cu kernel_average_grad_b_out.cu
CPP_SYSTEM_SOURCES  = system_read_input.cpp system_initialize.cpp system_print_info.cpp system_initialize_pointers.cpp system_constructor.cpp system_read_data_file.cpp system_destructor.cpp system_initialize_weights.cpp system_initialize_batches.cpp system_copy_pointers_to_device.cpp system_free_device_double_pointer.cpp system_pick_batch.cpp 
CPP_CONFIG_SOURCES  = config.h

# Objects are created depparately for CUDA and C++
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_SYSTEM_OBJECTS  = $(CPP_SYSTEM_SOURCES:.cpp=.o)

EXECUTABLE = neural_network_program

all: $(EXECUTABLE)

$(EXECUTABLE): $(CUDA_OBJECTS) $(CPP_SYSTEM_OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^

%.o: %.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -c $< -o $@

%.o: %.cpp $(HEADERS)
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(EXECUTABLE)
