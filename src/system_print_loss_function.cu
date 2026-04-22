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

// function to print loss function
void class_system::print_loss_function(dim3  numBlocks,
				       dim3  threadsPerBlock,
				       int   batch,
				       real* k_loss_function,
				       real* k_exit_value,
				       int*  k_batch_index,
				       real* k_data,
				       int   step) {

  //--  Loss function is set to zero --
  kernel_loss_function_to_zero<<<1, 1>>>(k_loss_function);  

  //-- Loss function is calculated --
  kernel_loss_function<<<numBlocks, threadsPerBlock>>>(batch,
						       k_loss_function,
						       k_exit_value,
						       k_batch_index,
						       k_data);  

  //-- Loss function is copied into the host --
  cudaMemcpy(&this->loss_function, k_loss_function, sizeof(real), cudaMemcpyDeviceToHost);  

  //-- The file is opened --
  char filename[50];
  sprintf(filename, "loss_function.dat");
  FILE* file;
  if (step == 0)
    file = fopen(filename, "w");
  else
    file = fopen(filename, "a");
  if (!file)
    {
      printf("System print loss function error: Error opening the file %s\n", filename);
      return;
    }
  
  fprintf(file, "%d %f\n", step, this->loss_function);

  fclose(file);

  //  printf("Loss function file written. Time step %d\n", step);
}
