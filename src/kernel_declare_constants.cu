/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "config.h"

__constant__ int Ndata;
__constant__ int Nbatches;
__constant__ int N_per_batch;
__constant__ int Nneurons;
__constant__ int Nhidden;
__constant__ real beta1;
__constant__ real beta2;
__constant__ real eta;
__constant__ real epsilon_adam;
