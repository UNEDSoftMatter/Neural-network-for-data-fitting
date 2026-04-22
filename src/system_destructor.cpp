/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include <stdio.h>

// Destructor of the class system
void class_system::destructor() {
  if (data)
    delete[] data;

  for (int k = 0; k < Nhidden; k++) {
    if (W_hidden[k])
      delete[] W_hidden[k];
    if (b_hidden[k])
      delete[] b_hidden[k];
  }
  if (W_hidden)
    delete[] W_hidden;
  if (b_hidden)
    delete[] b_hidden;
  
  // output ewights and biases are destroyed
  if (W_out)
    delete[] W_out;  
  if (b_out)
    delete[] b_out;

  // Batch indices
  if (batch_index)
    delete[] batch_index;

  printf("System destroyed\n");
}
