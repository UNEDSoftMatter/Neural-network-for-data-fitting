/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "config.h"
#include <math.h>

#include <stdio.h>

// Initialization of the system
int class_system::initialize(int  Nfiles,
			     int  Niterations,
			     int  initialization,
			     real epsilon,
			     real eta_sys,
			     real beta1_sys,
			     real beta2_sys,
			     real eps_adam) {

  int error = 0;

  //*** Some variables have been already defined in system_constructor ***//

  this->Nfiles         = Nfiles;
  this->Niterations    = Niterations;
  this->initialization = initialization;
  this->epsilon        = epsilon;
  this->eta            = eta_sys;
  this->beta1          = beta1_sys;
  this->beta2          = beta2_sys;
  this->epsilon_adam   = eps_adam;

  //--- The random number generator is initialized --
  std::random_device rd;   // This generates the seed
  this->gen.seed(rd());    // Mersenne Twister 19937 (random generator algorithm)  
  // this->gen.seed(12345);   // Fixed seed

  //---- Weights are initialized ----
  error = initialize_weights();
  if (error != 0)
    return error;

  //--- Batches are initialized ---
  initialize_batches();

  printf("System initialized\n");

  return 0;
  
}
