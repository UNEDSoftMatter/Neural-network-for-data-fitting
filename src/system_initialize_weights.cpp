/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <stdio.h>
#include <random>

// Initialization of the weights
// initialization = 1: uniform random distribution in (-epsilon, epsilon)
// initialization = 2: Xavier initialization
int class_system::initialize_weights() {
  
  if (initialization < 1 || initialization > 2) {
    printf("system initialize weights error: initialization = %d is not an available option. \n", initialization);
    return 1;
  }

  //---- Epsilon0 is defined (size of the uniform random distribution)
  real epsilon0;
  if (initialization == 1) // Uniform random distribution in (-epsilon, epsilon)
    epsilon0 = epsilon;
  else // initialization == 2 (2 inputs, 1 output)
    epsilon0 = sqrt(6) / sqrt(2 + 1);

  // Distribución uniforme entre 0.0 y 1.0
  std::uniform_real_distribution<real> dist(-epsilon0, epsilon);

  //-- Weights between the input and the first hidden layer (2 entries)  
  for (int i = 0; i < 2 * Nneurons; i++) 
    W_hidden[0][i] = dist(gen);
  for (int i = 0; i < Nneurons; i++) 
    b_hidden[0][i] = dist(gen);
  //-- Rest of hidden layers --
  for (int k = 1; k < Nhidden; k++) 
    for (int i = 0; i < Nneurons * Nneurons; i++)    
      W_hidden[k][i] = dist(gen);
  for (int k = 1; k < Nhidden; k++) 
    for (int i = 0; i < Nneurons; i++)    
      b_hidden[k][i] = dist(gen);
  //-- Weights from last hidden layer to output (1 output) --
  for (int i = 0; i < Nneurons; i++)
    W_out[i] = dist(gen);
  b_out[0] = dist(gen);  // One output

  return 0;
    
}
