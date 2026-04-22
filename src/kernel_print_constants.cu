/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"
#include <stdio.h>

//--- Constant variables storaged in the GPU are written ---
__global__ void kernel_print_constants() {

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i == 0)    {
    printf("----------------- Info storaged in the GPU -------------------------\n");
    printf("Number of data, Ndata                        = %d\n", Ndata);    
    printf("Number of batches, Nbatches                  = %d\n", Nbatches);
    printf("Number of data per batch, N_per_batch        = %d\n", N_per_batch);
    printf("Number of neurons per hidden layer, Nneurons = %d\n", Nneurons);
    printf("Number of hidden layers, Nhidden             = %d\n", Nhidden);
    printf("----- Adam parameters -----\n");
    printf("beta1                                        = %f\n", beta1);
    printf("beta2                                        = %f\n", beta2);
    printf("eta                                          = %f\n", eta);
    printf("epsilon_ADAM                                 = %f\n", epsilon_adam);
    printf("--------------------------------------------------------------------\n"); 
  }

}
