/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <iostream>
#include "cuda_runtime.h"
#include "class_system.h"
#include "kernel_functions.h"

// function to write a gnuplot script to output the approximation function
void class_system::print_approximation_function(int    step,
						real** k_W_hidden,
						real*  k_W_out,
						real** k_b_hidden,
						real*  k_b_out) {

  //-- The file is opened --
  char filename[50];
  sprintf(filename, "approx_function-%d.gnu", step);
  FILE* file = fopen(filename, "w");
  if (!file) {
    printf("System print particle error: Error opening the file %s\n", filename);
    return;
  }

  //-- First, we write the activation function --
  fprintf(file, "sigma(x) = 1.0/ (1.0 + exp(-x))\n");

  //---------------------
  //--- HIDDEN WEIGHTS --
  //---------------------
  // The GPU values are transferred from the GPU to the CPU
  real** temp = new real*[Nhidden];
  cudaMemcpy(temp, k_W_hidden, Nhidden * sizeof(real*), cudaMemcpyDeviceToHost);
  for (int k = 0; k < Nhidden; k++) {
    int size_k;
    if (k == 0)
        size_k = Nneurons * 2;
    else
        size_k = Nneurons * Nneurons;
    cudaMemcpy(W_hidden[k],      // destino CPU
               temp[k],           // origen GPU
               size_k * sizeof(real),
               cudaMemcpyDeviceToHost);
  }
  delete[] temp;
  // Weights are written
  // indices ranges:
  // w_j0: 0 -> Nneurons - 1
  // w_j1: Nneurons -> Nneurons + (Nneurons - 1)
  // etc
  // In general, w_ji: i*Nneurons -> i*Nneurons + (Nneurons - 1)
  // In general w_ji^k = W_hidden[k][j + i*Nneurons]
  // -- First hidden layer --
  int n;
  int k_aux = 0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < Nneurons; j++) {
      n = i*Nneurons + j;
      fprintf(file, "W_%d_%d_%d = " REAL_FMT "\n", j, i, k_aux, W_hidden[k_aux][n]);
    }
  //-- Rest of hidden layers --
  for (int k = 1; k < Nhidden; k++)
    for (int i = 0; i < Nneurons; i++)
      for (int j = 0; j < Nneurons; j++) {
	n = i*Nneurons + j;
	fprintf(file, "W_%d_%d_%d = " REAL_FMT "\n", j, i, k, W_hidden[k][n]);
      }
    
  //---------------------
  //--- OUTPUT WEIGHTS --
  //---------------------
  // Weights from the last hidden layer to the output layer
  cudaMemcpy(this->W_out,  k_W_out,  Nneurons * sizeof(real), cudaMemcpyDeviceToHost);
  k_aux = Nhidden;
  int j_aux = 0;
  for (int i = 0; i < Nneurons; i++)
    fprintf(file, "W_%d_%d_%d = " REAL_FMT "\n", j_aux, i, k_aux, W_out[i]);

  //----------------------
  //--- HIDDEN BIASES ----
  //----------------------
  // Biases of the hidden layers
  // The GPU values are transferred from the GPU to the CPU
  temp = new real*[Nhidden];
  cudaMemcpy(temp, k_b_hidden, Nhidden * sizeof(real*), cudaMemcpyDeviceToHost);
  for (int k = 0; k < Nhidden; k++) {
    int size_k;
    size_k = Nneurons;
    cudaMemcpy(b_hidden[k],      // destino CPU
               temp[k],           // origen GPU
               size_k * sizeof(real),
               cudaMemcpyDeviceToHost);
  }
  delete[] temp;
  // Biases are written
  for (int k = 0; k < Nhidden; k++)
    for (int j = 0; j < Nneurons; j++) 
	fprintf(file, "b_%d_%d = " REAL_FMT "\n", j, k, b_hidden[k][j]);

  //---------------------
  //--- OUTPUT BIASES --
  //---------------------
  //-- Output biases --
  k_aux = Nhidden;
  j_aux = 0;
  cudaMemcpy(this->b_out,  k_b_out, 1 * sizeof(real), cudaMemcpyDeviceToHost);
  fprintf(file, "b_%d_%d = " REAL_FMT "\n", j_aux, k_aux, b_out[0]);

  //------------------
  //--- FUNCTIONS ----
  //------------------
  //-- Input layer --
  for (int i = 0; i < 2; i++)  
    fprintf(file, "z_in_%d(x_0, x_1) = x_%d\n", i, i);
  //-- First hidden layer --
  k_aux = 0;
  for (int j = 0; j < Nneurons; j++) {
    fprintf(file, "a_%d_%d(x_0, x_1) = \\\n", j, k_aux);    
    for (int i = 0; i < 2; i++)
      if (i < 1)
	fprintf(file, "+ W_%d_%d_%d * z_in_%d(x_0, x_1) \\\n", j, i, k_aux, i);
      else
	fprintf(file, "+ W_%d_%d_%d * z_in_%d(x_0, x_1) + b_%d_%d \n", j, i, k_aux, i, j, k_aux);
    fprintf(file, "z_%d_%d(x_0, x_1) = sigma(a_%d_%d(x_0, x_1)) \n\n", j, k_aux, j, k_aux);
    }
  //-- Rest of hidden layers --
  for (int k = 1; k < Nhidden; k++)  
    for (int j = 0; j < Nneurons; j++) {
      fprintf(file, "a_%d_%d(x_0, x_1) = \\\n", j, k);    
      for (int i = 0; i < Nneurons; i++)
	if (i < Nneurons - 1)
	  fprintf(file, "+ W_%d_%d_%d * z_%d_%d(x_0, x_1) \\\n", j, i, k, i, k-1);
	else
	  fprintf(file, "+ W_%d_%d_%d * z_%d_%d(x_0, x_1) + b_%d_%d \n", j, i, k, i, k-1, j, k);
      fprintf(file, "z_%d_%d(x_0, x_1) = sigma(a_%d_%d(x_0, x_1)) \n\n", j, k, j, k);
}
  //-- Output layer (approximation function) --
  fprintf(file, "f(x_0, x_1) = \\\n");  
  for (int i = 0; i < Nneurons; i++) 
    if (i < Nneurons - 1)
      fprintf(file, "+ W_%d_%d_%d * z_%d_%d(x_0, x_1) \\\n", 0, i, Nhidden, i, Nhidden-1);
    else
      fprintf(file, "+ W_%d_%d_%d * z_%d_%d(x_0, x_1) + b_%d_%d \n", 0, i, Nhidden, i, Nhidden-1, 0, Nhidden);      
  

  //-- The order to plot is written --
  fprintf(file, "splot f(x,y)\n");
  fprintf(file, "pause -1\n");
    
  //-- The file is closed --
  fclose(file);

  printf("Approx function gnu file written. Time step %d\n", step);  
}
