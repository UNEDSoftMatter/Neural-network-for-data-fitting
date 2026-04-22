/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <stdio.h>

// Constructor of the class system
void class_system::initialize_pointers() {

  data          = nullptr;
  W_hidden      = nullptr;
  W_out         = nullptr;
  b_hidden      = nullptr;
  b_out         = nullptr;
  batch_index   = nullptr;

  printf("Pointers initialized\n");
}
