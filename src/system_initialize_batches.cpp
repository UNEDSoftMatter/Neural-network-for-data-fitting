/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <stdio.h>

// Initialization of data batches
void class_system::initialize_batches() {

  //*** Nbatches is defined in system_constructor ****

  // batch_index[k] is the first index of the bathc k
  for (int k = 0; k < Nbatches; k++) 
    batch_index[k] = k * N_per_batch;
    
}
