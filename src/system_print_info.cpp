/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <iostream>
#include "class_system.h"

// function to print system info
void class_system::print_info() {
  std::cout << " \n";
  std::cout << "------------------ System info -----------------------------"     << " \n";
  std::cout << "Number of neurons per layer, Nneurons       = " << Nneurons       << " \n";
  std::cout << "Number of hidden layers, Nhidden            = " << Nhidden        << " \n";
  std::cout << "Number of iterations, Niterations           = " << Niterations    << " \n";
  std::cout << "Number of data per batch, N_per_batch       = " << N_per_batch    << " \n";
  std::cout << "initialization option                       = " << initialization << " \n";
  std::cout << "epsilon                                     = " << epsilon        << " \n";
  std::cout << "Initial learning rate, eta                  = " << eta            << " \n";
  std::cout << "Parameter of the Adams method, beta1        = " << beta1          << " \n";
  std::cout << "Parameter of the Adams method, beta2        = " << beta2          << " \n";
  std::cout << "Parameter of the Adams method, epsilon_adam = " << epsilon_adam   << " \n";
  std::cout << "Number of data files, Nfiles                = " << Nfiles         << " \n";
  std::cout << "Number of data read, Ndata                  = " << Ndata          << " \n";
  std::cout << "Number of batches, Nbatches                 = " << Nbatches       << " \n";
  std::cout << "Initial indices of each batch\n";
  for (int k = 0; k < Nbatches; k++) 
    std::cout << "Batch "<< k << " index " << batch_index[k] << " \n";
  
  std::cout << "------------------------------------------------------------"     << " \n";
  std::cout << " \n";  
}
