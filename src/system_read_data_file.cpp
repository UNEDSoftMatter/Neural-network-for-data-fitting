#include "class_system.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "config.h"

// Function to read data from the file
// To access the file of i row and j column:
//     datos[i*3 + j]
// row 0 → data[0], data[1], data[2]
// row 1 → data[3], data[4], data[5]
// row 2 → data[6], data[7], data[8]
// ...
// row i → data[i*3 + 0], data[i*3 + 1], data[i*3 + 2]
int class_system::read_data_file(const char* file_name)
{
    std::ifstream file(file_name);
    if (!file.is_open()) {
      printf("System read data file error: the data file is already open.\n");
        return 1;
    }

    std::vector<real> temp;
    real a, b, c;

    while (file >> a >> b >> c) {
        temp.push_back(a);
        temp.push_back(b);
        temp.push_back(c);
    }

    Ndata = temp.size() / 3;

    // Dynamic memory is reserved
    data = new real[temp.size()];

    // Data is copied to the pointer
    for (size_t i = 0; i < temp.size(); ++i)
        data[i] = temp[i];

    printf("Data file read\n");

    return 0;
}
