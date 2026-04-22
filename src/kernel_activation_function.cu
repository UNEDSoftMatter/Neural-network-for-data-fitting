/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "kernel_functions.h"
#include "config.h"

// Activation function of the neural network
__device__ real kernel_activation_function(real x) {
  
  real f;

  //------ ReLU function -----
  // if (x > 0)
  //   f = x;
  // else
  //   f = 0;

  //--- Sigmoid function ---
  f = 1.0/(1.0 + exp(-x));
    
  return f;
}
