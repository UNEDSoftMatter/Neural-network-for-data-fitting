/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/

#include "class_system.h"
#include "cuda_runtime.h"

#include <stdio.h>

// Function to copy the system data to the device
void class_system::copy_pointers_to_device(real**  k_data,
					   real*** k_W_hidden,
					   real**  k_W_out,
					   real*** k_b_hidden,
					   real**  k_b_out,
					   real*** k_z,
					   real*** k_a,
					   int**   k_batch_index,
					   real**  k_exit_value,
					   real*** k_delta,
					   real**  k_delta_out,
					   real*** k_grad_W_hidden,
					   real**  k_grad_W_out,
					   real*** k_grad_b_hidden,
					   real**  k_grad_b_out,
					   real*** k_m_W_hidden,
					   real**  k_m_W_out,
					   real*** k_v_W_hidden,
					   real**  k_v_W_out,
					   real*** k_m_b_hidden,
					   real**  k_m_b_out,
					   real*** k_v_b_hidden,
					   real**  k_v_b_out,
					   real**  k_loss_function) {
  
  //------- Variables to copy -------------
  //--- data ---
  cudaMalloc(k_data, Ndata * 3 * sizeof(real)); // 2 inputs, 1 output
  cudaMemcpy(*k_data, data, Ndata * 3 * sizeof(real), cudaMemcpyHostToDevice);  

  //--- W_hidden ---
  cudaMalloc(k_W_hidden, Nhidden * sizeof(real*));
  real** temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k;
    if (k == 0)
      size_k = Nneurons * 2;
    else
      size_k = Nneurons * Nneurons;
    // We reserve memory in GPU for the k layer
    real* dWk;
    cudaMalloc(&dWk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemcpy(dWk, W_hidden[k], size_k * sizeof(real), cudaMemcpyHostToDevice);    
    // The pointer of device of this layer is saved
    temp[k] = dWk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_W_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- W_out ---
  cudaMalloc(k_W_out, Nneurons * 1 * sizeof(real)); // 1 output
  cudaMemcpy(*k_W_out, W_out, Nneurons * 1 * sizeof(real), cudaMemcpyHostToDevice);  
  
  //--- b_hidden ---
  cudaMalloc(k_b_hidden, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k = Nneurons;
    // We reserve memory in GPU for the k layer
    real* dbk;
    cudaMalloc(&dbk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemcpy(dbk, b_hidden[k], size_k * sizeof(real), cudaMemcpyHostToDevice);    
    // The pointer of device of this layer is saved
    temp[k] = dbk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_b_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- b_out ---
  cudaMalloc(k_b_out, 1 * sizeof(real)); // 1 output
  cudaMemcpy(*k_b_out, b_out, 1 * sizeof(real), cudaMemcpyHostToDevice);  

  //--- z ---
  cudaMalloc(k_z, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k = Nneurons * N_per_batch;
    // We reserve memory in GPU for the k layer
    real* d_z;
    cudaMalloc(&d_z, size_k * sizeof(real));
    // The layer is initialized
    cudaMemset(d_z, 0, size_k * sizeof(real));    
    // The pointer of device of this layer is saved
    temp[k] = d_z;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_z, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- a ---
  cudaMalloc(k_a, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k = Nneurons * N_per_batch;
    // We reserve memory in GPU for the k layer
    real* d_a;
    cudaMalloc(&d_a, size_k * sizeof(real));
    // The layer is initialized
    cudaMemset(d_a, 0, size_k * sizeof(real));    
    // The pointer of device of this layer is saved
    temp[k] = d_a;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_a, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;      

  //--- batch_index ----
  cudaMalloc(k_batch_index, Nbatches * sizeof(int));
  cudaMemcpy(*k_batch_index, batch_index, Nbatches * sizeof(int), cudaMemcpyHostToDevice);

  //--- exit_value (one value per each data in the batch) ---
  cudaMalloc(k_exit_value, N_per_batch * sizeof(real));
  cudaMemset(*k_exit_value, 0, N_per_batch * sizeof(real));

  //--- delta ---
  cudaMalloc(k_delta, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k = Nneurons * N_per_batch;
    // We reserve memory in GPU for the k layer
    real* d_delta;
    cudaMalloc(&d_delta, size_k * sizeof(real));
    // The layer is initialized
    cudaMemset(d_delta, 0, size_k * sizeof(real));    
    // The pointer of device of this layer is saved
    temp[k] = d_delta;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_delta, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- k_delta_out ---
  cudaMalloc(k_delta_out, N_per_batch * sizeof(real)); // 1 output, N_per_batch batches
  cudaMemset(*k_delta_out, 0, N_per_batch * sizeof(real));

  //--- grad_W_hidden ---
  cudaMalloc(k_grad_W_hidden, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k;
    if (k == 0)
      size_k = Nneurons * 2;
    else
      size_k = Nneurons * Nneurons;
    // We reserve memory in GPU for the k layer
    real* d_grad_Wk;
    cudaMalloc(&d_grad_Wk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemset(d_grad_Wk, 0, size_k * sizeof(real));        
    // The pointer of device of this layer is saved
    temp[k] = d_grad_Wk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_grad_W_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- grad_b_hidden ---
  cudaMalloc(k_grad_b_hidden, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k = Nneurons;
    // We reserve memory in GPU for the k layer
    real* dbk;
    cudaMalloc(&dbk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemset(dbk, 0, size_k * sizeof(real));            
    // The pointer of device of this layer is saved
    temp[k] = dbk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_grad_b_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- grad_W_out ---
  cudaMalloc(k_grad_W_out, Nneurons * 1 * sizeof(real)); // 1 output
  cudaMemset(*k_grad_W_out, 0, Nneurons * 1 * sizeof(real));

  //--- grad_b_out ---
  cudaMalloc(k_grad_b_out, 1 * sizeof(real)); // 1 output
  cudaMemset(*k_grad_b_out, 0, 1 * sizeof(real));  

  //--- m_W_hidden ---
  cudaMalloc(k_m_W_hidden, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k;
    if (k == 0)
      size_k = Nneurons * 2;
    else
      size_k = Nneurons * Nneurons;
    // We reserve memory in GPU for the k layer
    real* d_m_Wk;
    cudaMalloc(&d_m_Wk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemset(d_m_Wk, 0, size_k * sizeof(real));        
    // The pointer of device of this layer is saved
    temp[k] = d_m_Wk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_m_W_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- m_W_out ---
  cudaMalloc(k_m_W_out, Nneurons * 1 * sizeof(real)); // 1 output
  cudaMemset(*k_m_W_out, 0, Nneurons * 1 * sizeof(real));
  
  //--- v_W_hidden ---
  cudaMalloc(k_v_W_hidden, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k;
    if (k == 0)
      size_k = Nneurons * 2;
    else
      size_k = Nneurons * Nneurons;
    // We reserve memory in GPU for the k layer
    real* d_v_Wk;
    cudaMalloc(&d_v_Wk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemset(d_v_Wk, 0, size_k * sizeof(real));        
    // The pointer of device of this layer is saved
    temp[k] = d_v_Wk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_v_W_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- v_W_out ---
  cudaMalloc(k_v_W_out, 1 * sizeof(real)); // 1 output
  cudaMemset(*k_v_W_out, 0, 1 * sizeof(real));

  //--- m_b_hidden ---
  cudaMalloc(k_m_b_hidden, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k = Nneurons;    
    // We reserve memory in GPU for the k layer
    real* d_m_bk;
    cudaMalloc(&d_m_bk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemset(d_m_bk, 0, size_k * sizeof(real));        
    // The pointer of device of this layer is saved
    temp[k] = d_m_bk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_m_b_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- m_b_out ---
  cudaMalloc(k_m_b_out, 1 * sizeof(real)); // 1 output
  cudaMemset(*k_m_b_out, 0, 1 * sizeof(real));
  
  //--- v_b_hidden ---
  cudaMalloc(k_v_b_hidden, Nhidden * sizeof(real*));
  temp = new real*[Nhidden];  // Temporal array in CPU containing pointers of device
  for (int k = 0; k < Nhidden; k++) {
    int size_k = Nneurons;    
    // We reserve memory in GPU for the k layer
    real* d_v_bk;
    cudaMalloc(&d_v_bk, size_k * sizeof(real));
    // The layer is copied from the CPU to the GPU
    cudaMemset(d_v_bk, 0, size_k * sizeof(real));        
    // The pointer of device of this layer is saved
    temp[k] = d_v_bk;
  }
  // The array of pointers is copied to the device
  cudaMemcpy(*k_v_b_hidden, temp, Nhidden * sizeof(real*), cudaMemcpyHostToDevice);
  // The temporal array is freed
  delete[] temp;

  //--- v_b_out ---
  cudaMalloc(k_v_b_out, 1 * sizeof(real)); // 1 output
  cudaMemset(*k_v_b_out, 0, 1 * sizeof(real));

  //--- loss_function ---
  cudaMalloc(k_loss_function, 1 * sizeof(real)); // 1 output
  cudaMemset(*k_loss_function, 0, 1 * sizeof(real));  

}
