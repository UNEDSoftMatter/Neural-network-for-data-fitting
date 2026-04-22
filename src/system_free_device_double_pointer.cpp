/******************************************************
This code has been developed by Adolfo Vazquez-Quesada,
from the Department of Fundamental Physics at UNED, in
Madrid, Spain.
email: a.vazquez-quesada@fisfun.uned.es
********************************************************/
#include "class_system.h"
#include "config.h"

//-- Function to free double pointers of the GPU ---
void class_system::free_device_double_pointer(real** k_pointer)
{
    // 1. Copy the array of internal pointers from the GPU to the CPU
    real** temp = new real*[Nhidden];
    cudaMemcpy(temp, k_pointer, Nhidden * sizeof(real*), cudaMemcpyDeviceToHost);

    // 2. Free each device matrix
    for (int k = 0; k < Nhidden; k++)
        cudaFree(temp[k]);

    // 3. Free the array of pointers in the device
    cudaFree(k_pointer);

    // 4. Free the temporal array
    delete[] temp;
}
