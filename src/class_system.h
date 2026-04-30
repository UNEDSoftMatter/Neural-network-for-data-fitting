/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "config.h"
#include <cuda_runtime.h>
#include <random>

struct class_system {
  int Nneurons;       // Number of neurons per layer
  int Nhidden;        // Number of hidden layers
  int Nfiles;         // Number of files
  int Niterations;    // Number of iterations
  int initialization; // Initialization option.
                      //  1. From a uniform random distribution in the
                      //     interval (-epsilon, epsilon)
                      //  2. Xavier initialization
  real epsilon;       // epsilon for initialization = 1
  int  N_per_batch;   // Number of data per batch
  int  Nbatches;      // Number of data batches
  int* batch_index;   // indices pointing the initial index of data of each batch
  real eta;           // initial learning rate
  real beta1;         // Parameter of the Adams model
  real beta2;         // Parameter of the Adams model
  real epsilon_adam;  // Parameter of the Adams model
  real* data;         // Data to fit
  size_t Ndata;       // Number of data read from the data file
  real** W_hidden;    // Weights of the hidden layers
  real*  W_out;       // Weights of the last hidden layers to output
  real** b_hidden;    // Biases of the hidden layers.
  real*  b_out;       // Bias of the last hidden layer to output
  real loss_function; // Loss function in the host
  std::mt19937 gen;   // random number generator


  /********* Subroutines **********/
  int read_input(int&   Nneurons_per_layer,
		 int&   Nhidden_layers,
		 int&   Nfiles,
		 int&   Niterations,
		 int&   initialization,
		 real&  epsilon,
		 int&   Number_per_batch,
		 real&  eta_sys,
		 real&  beta1_sys,
		 real&  beta2_sys,
		 real&  eps_adam,
		 int&   freq_loss_function,
		 int&   freq_gnu_file);
  int initialize(int  Nfiles,
		 int  Niterations,
		 int  initialization,
		 real epsilon,
		 real eta_sys,
		 real beta1_sys,
		 real beta2_sys,
		 real eps_adam);
  void print_info();
  void initialize_pointers();
  void constructor(int Nneurons_per_layer,
		   int Nhidden_layers,
		   int Number_per_batch);
  void destructor();  
  int  read_data_file(const char* file_name);
  int  initialize_weights();
  void initialize_batches();
  void copy_pointers_to_device(real**  k_data,
			       real*** k_W_hidden,
			       real**  k_W_out,
			       real*** k_b_hidden,
			       real**  k_b_out,
			       real*** k_z,
			       real*** k_a,			       
			       int**   k_batch_index,
			       real**  k_exit_value,
			       real*** k_delta,
			       real**  k_delta_out,
			       real*** k_grad_W_hidden,
			       real**  k_grad_W_out,
			       real*** k_grad_b_hidden,
			       real**  k_grad_b_out,
			       real*** k_m_W_hidden,
			       real**  k_m_W_out,
			       real*** k_v_W_hidden,
			       real**  k_v_W_out,
			       real*** k_m_b_hidden,
			       real**  k_m_b_out,
			       real*** k_v_b_hidden,
			       real**  k_v_b_out,
			       real**  k_loss_function);
  void print_loss_function(dim3  numBlocks,
			   dim3  threadsPerBlock,
			   int   batch,
			   real* k_loss_function,
			   real* k_exit_value,
			   int*  k_batch_index,
			   real* k_data,
			   int   step);  
  void free_device_double_pointer(real** k_pointer);
  void pick_batch(int &batch);
  void print_approximation_function(int    step,
				    real** k_W_hidden,
				    real*  k_W_out,
				    real** k_b_hidden,
				    real*  k_b_out);
  
};
