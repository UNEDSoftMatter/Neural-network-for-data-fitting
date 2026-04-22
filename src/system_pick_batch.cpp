/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"

// Function to pick a batch (between 0 and Nbatches - 1
void class_system::pick_batch(int& batch) {

  std::uniform_int_distribution<int> dist(0, Nbatches - 1);
  batch = dist(gen);
  
}
