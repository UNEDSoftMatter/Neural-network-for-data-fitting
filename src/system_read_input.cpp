/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "class_system.h"
#include "config.h"
#include <math.h>

#define MAX_LINE 256

int class_system::read_input(int&   Nneurons_per_layer,
			     int&   Nhidden_layers,
			     int&   Nfiles,
			     int&   Niterations,
			     int&   initialization,
			     real&  epsilon,
			     int&   Number_per_batch,
			     real&  eta_sys,
			     real&  beta1_sys,
			     real&  beta2_sys,
			     real&  eps_adam,
			     int&   freq_loss_function,
			     int&   freq_gnu_file) {

  FILE *file = fopen("input", "r");
  if (!file) {
    perror("Opening file error");                
    return 0;
  }

  char line[MAX_LINE];
  int Nneurons_per_layer_read    = 1;
  int Nhidden_layers_read        = 1;
  int Nfiles_read                = 1;
  int Niterations_read           = 1;
  int initialization_read        = 1;
  int epsilon_read               = 1;
  int Number_per_batch_read      = 1;
  int eta_sys_read               = 1;
  int beta1_sys_read             = 1;
  int beta2_sys_read             = 1;
  int eps_adam_read              = 1;
  int freq_loss_function_read    = 1;
  int freq_gnu_file_read         = 1;  
    
  while (fgets(line, sizeof(line), file)) {
    char var[50];
    double val1, val2, val3;

    // Elimina espacios al principio y al final
    char *start = line;
    while (*start == ' ' || *start == '\t') start++;
    char *end = start + strlen(start) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\n')) *end-- = '\0';

    // Extrae la variable y hasta 3 valores
    int count = sscanf(start, "%s %lf %lf %lf", var, &val1, &val2, &val3);
	
    if (strcmp(var, "Nneurons") == 0) {
      Nneurons_per_layer                   = (int)val1;
      Nneurons_per_layer_read              = 0;
    }
    else if (strcmp(var, "Nhidden") == 0) {
      Nhidden_layers                    = (int)val1;
      Nhidden_layers_read               = 0;      
    }
    else if (strcmp(var, "Nfiles") == 0) {
      Nfiles                     = (int)val1;
      Nfiles_read                = 0;      
    }
    else if (strcmp(var, "Niterations") == 0) {
      Niterations                = (int)val1;
      Niterations_read           = 0;      
    }
    else if (strcmp(var, "initialization") == 0) {
      initialization             = (int)val1;
      initialization_read        = 0;      
    }
    else if (strcmp(var, "epsilon") == 0) {
      epsilon                    = val1;
      epsilon_read               = 0;      
    }
    else if (strcmp(var, "N_per_batch") == 0) {
      Number_per_batch                = (int)val1;
      Number_per_batch_read           = 0;      
    }
    else if (strcmp(var, "eta") == 0) {
      eta_sys                        = val1;
      eta_sys_read                   = 0;      
    }
    else if (strcmp(var, "beta1") == 0) {
      beta1_sys                      = val1;
      beta1_sys_read                 = 0;      
    }
    else if (strcmp(var, "beta2") == 0) {
      beta2_sys                      = val1;
      beta2_sys_read                 = 0;      
    }
    else if (strcmp(var, "epsilon_adam") == 0) {
      eps_adam                       = val1;
      eps_adam_read                  = 0;      
    }
    else if (strcmp(var, "freq_loss_function") == 0) {
      freq_loss_function             = val1;
      freq_loss_function_read        = 0;      
    }
    else if (strcmp(var, "freq_gnu_file") == 0) {
      freq_gnu_file                  = val1;
      freq_gnu_file_read             = 0;      
    }                                    
  }

  //-- Checking that every variable was read --
  if (Nneurons_per_layer_read == 1)  {
    printf("system read input error: Nneurons was not read\n");
    return 1;
  }
  if (Nhidden_layers_read == 1)  {
    printf("system read input error: Nhidden was not read\n");
    return 1;
  }
  if (Nfiles_read == 1)  {
    printf("system read input error: Nfiles was not read\n");
    return 1;
  }
  if (Niterations_read == 1)  {
    printf("system read input error: Niterations was not read\n");
    return 1;
  }
  if (initialization_read == 1)  {
    printf("system read input error: initialization was not read\n");
    return 1;
  }
  if (epsilon_read == 1)  {
    printf("system read input error: epsilon was not read\n");
    return 1;
  }
  if (Number_per_batch_read == 1)  {
    printf("system read input error: N_per_batch was not read\n");
    return 1;
  }
  if (eta_sys_read == 1)  {
    printf("system read input error: eta was not read\n");
    return 1;
  }  
  if (beta1_sys_read == 1)  {
    printf("system read input error: beta1 was not read\n");
    return 1;
  }
  if (beta2_sys_read == 1)  {
    printf("system read input error: beta2 was not read\n");
    return 1;
  }
  if (eps_adam_read == 1)  {
    printf("system read input error: epsilon_adam was not read\n");
    return 1;
  }
  if (freq_loss_function_read == 1)  {
    printf("system read input error: freq_loss_function was not read\n");
    return 1;
  }                  
  if (freq_gnu_file_read == 1)  {
    printf("system read input error: freq_gnu_file was not read\n");
    return 1;
  }                
  
  //-- Other checkings --
  if (Nneurons_per_layer < 1) {
    printf("system read input error: Nneurons minimum value is 1\n");
    return 1;
  }
  if (Nhidden_layers < 1) {
    printf("system read input error: Nhidden minimum value is 1\n");
    return 1;
  }
  if (Nfiles < 1) {
    printf("system read input error: Nfiles minimum value is 1\n");
    return 1;
  }
  if (Niterations < 1) {
    printf("system read input error: Niterations minimum value is 1\n");
    return 1;
  }
  if (initialization < 1 || initialization > 2) {
    printf("system read input error: option initialization = %d is not available\n",
	   initialization);
    return 1;
  }
  if (epsilon < 0) {
    printf("system read input error: epsilon should be a positive number\n");
    return 1;
  }
  if (Number_per_batch < 1) {
    printf("system read input error: Number_per_batch minimum value is 1\n");
    return 1;
  }
  

  fclose(file);

  printf("Input file read\n");

  return 0;
  }
