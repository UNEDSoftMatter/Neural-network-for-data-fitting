/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <iostream>
#include <time.h>
#include "class_system.h"
#include "kernel_functions.h"
#include "config.h"

//***************************
//   main program
//***************************
int main() {

  //-----------------------------------
  //----- Some declarations -----------
  //-----------------------------------  
  int error = 0;
  cudaError_t cuda_err;  
  const char* data_file_name = "data.dat";
  int batch;

  //--------------------------------------------------------------------  
  //--- Declaration of the input variables (and their default values)---
  //--------------------------------------------------------------------    
  int  Nneurons_per_layer    = 10;
  int  Nhidden_layers        = 3;
  int  Nfiles                = 1;
  int  Niterations           = 100000;
  int  initialization        = 2;
  real epsilon               = 1.0;
  int  Number_per_batch      = 100;
  real eta_sys               = 0.0001;
  real beta1_sys             = 0.9;
  real beta2_sys             = 0.999;
  real eps_adam              = 1.0e-8;
  int  freq_loss_function    = 100;
  int  freq_gnu_file         = 1000;  

  //--------------------------------------------------------------------  
  //---------- initialization ------------------------------------------
  //--------------------------------------------------------------------    
  // The neural_network system is initialized in the host 
  class_system sys;

  //-------------------------------
  //-----  Input file is read  ----
  //-------------------------------  
  error = sys.read_input(Nneurons_per_layer, Nhidden_layers, Nfiles,
			 Niterations, initialization, epsilon,
  			 Number_per_batch, eta_sys, beta1_sys, beta2_sys, eps_adam,
			 freq_loss_function, freq_gnu_file);
  if (error != 0)
    return; // End of program

  //----------------------------------------------  
  //---- Initializing all pointers as nullptr ----
  //----------------------------------------------    
  sys.initialize_pointers();

  //---------------------------------------
  //---- Reading data file ----------------
  //---------------------------------------
  sys.read_data_file(data_file_name);

  //------------------------------------  
  //----  The system is constructed ----
  //------------------------------------    
  sys.constructor(Nneurons_per_layer, Nhidden_layers, Number_per_batch);

  //------------------------------------    
  //---- The system is initialized -----
  //------------------------------------    
  error = sys.initialize(Nfiles, Niterations, initialization, epsilon,
			 eta_sys, beta1_sys, beta2_sys, eps_adam);
  if (error != 0)
    return;  

  // ----------------------------------  
  // ---- System info is displayed ----
  // ----------------------------------    
  sys.print_info();

  //---------------------------------------------------
  //---- Pointers in the device (GPU) are declared ----
  //---------------------------------------------------
  // If we need memory, we could use only k_a, and calculate sigma(k_a) when we need k_z
  real *k_data;           // Data
  real **k_W_hidden;      // Weights of the hidden layers
  real *k_W_out;          // Weight of the exit layer
  real **k_b_hidden;      // Biases of the hidden layers
  real *k_b_out;          // Bias of the exit layer
  real **k_z;             // Values of the neurons. As many as Nneurons*N_per_batches
  real **k_a;             // Values of the neurons before activation. z = sigma(a)
  int  *k_batch_index;    // Indices of the batches (initial position on k_data
  real *k_exit_value;     // Exit values of the neural network (one por data of the batch)
  real **k_delta;         // delta_i^(k) = dE/da_i^k, where a_i^k is the value of the 
                          //  neuron i of the layer k before activation.
  real *k_delta_out;      // delta of the exit layer
  real **k_grad_W_hidden; // gradient of error respect to the weights of the hidden layers
  real *k_grad_W_out;     // gradient of error respect to the weights of the output layer
  real **k_grad_b_hidden; // gradient of error respect to the biases of the hidden layers
  real *k_grad_b_out;     // gradient of error respect to the bias of the output layer
  real **k_m_W_hidden;    // Parameter of the ADAM method (first moment m of weights W)
  real *k_m_W_out;        // Parameter of the ADAM method (first moment m of weights W)
  real **k_v_W_hidden;    // Parameter of the ADAM method (second moment v of weights W)
  real *k_v_W_out;        // Parameter of the ADAM method (second moment v of weights W)
  real **k_m_b_hidden;    // Parameter of the ADAM method (first moment m of biases b)
  real *k_m_b_out;        // Parameter of the ADAM method (first moment m of biases b)
  real **k_v_b_hidden;    // Parameter of the ADAM method (second moment v of biases b)
  real *k_v_b_out;        // Parameter of the ADAM method (second moment v of biases b)
  real *k_loss_function;  // Loss function

  //-------------------------------------------------------------      
  //----- Some variables (constants) are passed to the GPU ------
  //-------------------------------------------------------------        
  // The constants are declared at kernel_declare_constants.cu, and must be also 
  // specified in kernel_functions.h
  cudaMemcpyToSymbol(Ndata        , &sys.Ndata        , sizeof(int));
  cudaMemcpyToSymbol(Nbatches     , &sys.Nbatches     , sizeof(int));
  cudaMemcpyToSymbol(N_per_batch  , &sys.N_per_batch  , sizeof(int));
  cudaMemcpyToSymbol(Nneurons     , &sys.Nneurons     , sizeof(int));
  cudaMemcpyToSymbol(Nhidden      , &sys.Nhidden      , sizeof(int));
  cudaMemcpyToSymbol(beta1        , &sys.beta1        , sizeof(real));
  cudaMemcpyToSymbol(beta2        , &sys.beta2        , sizeof(real));
  cudaMemcpyToSymbol(eta          , &sys.eta          , sizeof(real));
  cudaMemcpyToSymbol(epsilon_adam , &sys.epsilon_adam , sizeof(real));        

  //--------------------------------------------------------  
  //---- Memory for pointers is allocated in the device ----
  //--------------------------------------------------------
  sys.copy_pointers_to_device(&k_data, &k_W_hidden, &k_W_out, &k_b_hidden,
			      &k_b_out, &k_z, &k_a, &k_batch_index, &k_exit_value,
			      &k_delta, &k_delta_out, &k_grad_W_hidden,
			      &k_grad_W_out, &k_grad_b_hidden, &k_grad_b_out,
			      &k_m_W_hidden, &k_m_W_out, &k_v_W_hidden, &k_v_W_out,
			      &k_m_b_hidden, &k_m_b_out, &k_v_b_hidden, &k_v_b_out,
			      &k_loss_function);

  //------------------------------------  
  //--- Configure blocks and threads ---
  //------------------------------------
  //--- Changes in this value can modified performance of the code ---
  dim3 threadsPerBlock(128);
  dim3 numBlocks((sys.N_per_batch + threadsPerBlock.x - 1) / threadsPerBlock.x);

  //-------------------------------------------------    
  //--- Constants stored in the GPU are displayed ---
  //-------------------------------------------------      
  kernel_print_constants<<<1, 1>>>();
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf("Error en kernel_print_constants: %s\n", cudaGetErrorString(cuda_err));
    return;
  }
  
  //-------------------------------------------------------------------------------------- 
  //----------------------------------- Main loop ----------------------------------------
  //--------------------------------------------------------------------------------------
  clock_t start = clock();
  for (int step = 0; step < Niterations; ++step) {

    //--- The batch is selected ---
    sys.pick_batch(batch);
							       
    //-- data is forward propagated through the neural network --
    kernel_forward_propagation<<<numBlocks, threadsPerBlock>>>(batch,
							       k_batch_index,
							       k_data,
							       k_W_hidden,
							       k_b_hidden,
							       k_W_out,
							       k_b_out,
							       k_z,
							       k_a,
							       k_exit_value);

    //-- delta is calculated by back propagation through the neural network --    
    kernel_back_propagation<<<numBlocks, threadsPerBlock>>>(batch,
							    k_batch_index,
							    k_data,
							    k_W_hidden,
							    k_W_out,
							    k_a,
							    k_delta,
							    k_delta_out,
							    k_exit_value);
    
    //-- Average gradients of weights and biases are calculated --
    //--- gradients of weights
    // First layer
    kernel_average_grad_W_hidden_layer0<<<numBlocks, threadsPerBlock>>>(batch,
									k_batch_index,
									k_grad_W_hidden,
									k_delta,
									k_data);
    // Interior layers
    for (int k = 1; k < sys.Nhidden; k++)
      kernel_average_grad_W_hidden<<<numBlocks, threadsPerBlock>>>(batch,
								   k,
								   k_batch_index,
								   k_grad_W_hidden,
								   k_delta,
								   k_z);
    // Output layer
    kernel_average_grad_W_out<<<numBlocks, threadsPerBlock>>>(batch,
							      k_batch_index,
							      k_grad_W_out,
							      k_delta_out,
							      k_z);
    //--- Biases ---
    // Hidden layers (including the first one)
    for (int k = 0; k < sys.Nhidden; k++)
      kernel_average_grad_b_hidden<<<numBlocks, threadsPerBlock>>>(batch,
								   k, 
								   k_batch_index,
								   k_grad_b_hidden,
								   k_delta);
    // Output layer
    kernel_average_grad_b_out<<<1, 1>>>(batch,
					k_batch_index,
					k_grad_b_out,
					k_delta_out);	

    //-- Gradient descent is done in hidden layers --    
    //-- Weights, biases and parameters of the ADAM method are updated --
    int size;
    for (int k_layer = 0; k_layer < sys.Nhidden; ++k_layer) {
      //-- Updating weights --
      if (k_layer == 0) 
	size = sys.Nneurons * 2;
      else
	size = sys.Nneurons * sys.Nneurons;
      kernel_update_ADAM_hidden<<<numBlocks, threadsPerBlock>>>(k_layer,
								size,
								step + 1,
								k_grad_W_hidden,
								k_W_hidden,
								k_m_W_hidden,
								k_v_W_hidden);
      //-- Updating biases --
      size = sys.Nneurons;
      kernel_update_ADAM_hidden<<<numBlocks, threadsPerBlock>>>(k_layer,
								size,
								step + 1,
								k_grad_b_hidden,
								k_b_hidden,
								k_m_b_hidden,
								k_v_b_hidden);
    }
    //-- Gradient descent is done in out layer --    
    //-- Weights, biases and parameters of the ADAM method are updated --
    size = sys.Nneurons;
    kernel_update_ADAM_out<<<numBlocks, threadsPerBlock>>>(size,
							      step + 1,
							      k_grad_W_out,
							      k_W_out,
							      k_m_W_out,
							      k_v_W_out);
    size = 1;
    kernel_update_ADAM_out<<<numBlocks, threadsPerBlock>>>(size,
							   step + 1,
							   k_grad_b_out,
							   k_b_out,
							   k_m_b_out,
							   k_v_b_out);

    //--- Loss function is calculated and written ---
    if (step % freq_loss_function == 0 || step == 0) 
      sys.print_loss_function(numBlocks, threadsPerBlock, batch, k_loss_function,
			      k_exit_value, k_batch_index, k_data, step);
    if (step % freq_gnu_file == 0)
      sys.print_approximation_function(step, k_W_hidden, k_W_out, k_b_hidden,
				       k_b_out);
    
  }  //--------- End of main loop ------------

  //---------------------------------------
  //---- Computation time is displayed ----
  //---------------------------------------  
  clock_t end = clock();
  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Total time: %f seconds\n", time_spent);

  //---------------------------------
  //---- Resources are released -----
  //---------------------------------  
  cudaFree(k_data);
  cudaFree(k_W_out);
  cudaFree(k_b_out);
  cudaFree(k_batch_index);
  cudaFree(k_exit_value);
  cudaFree(k_delta_out);
  cudaFree(k_grad_W_out);
  cudaFree(k_grad_b_out);
  cudaFree(k_m_W_out);
  cudaFree(k_v_W_out);
  cudaFree(k_m_b_out);
  cudaFree(k_v_b_out);
  cudaFree(k_loss_function);  
  sys.free_device_double_pointer(k_W_hidden);
  sys.free_device_double_pointer(k_b_hidden);
  sys.free_device_double_pointer(k_z);
  sys.free_device_double_pointer(k_a);  
  sys.free_device_double_pointer(k_delta);
  sys.free_device_double_pointer(k_grad_W_hidden);
  sys.free_device_double_pointer(k_grad_b_hidden);
  sys.free_device_double_pointer(k_m_W_hidden);
  sys.free_device_double_pointer(k_v_W_hidden);
  sys.free_device_double_pointer(k_m_b_hidden);
  sys.free_device_double_pointer(k_v_b_hidden);              
  cudaDeviceReset();      

  //-------------------------------------  
  //---- The sys object is destroyed ----
  //-------------------------------------    
  sys.destructor(); 
  
  return 0;
}  //-------------- End of main ------------------
