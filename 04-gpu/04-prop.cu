#include <iostream>


int main(void)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dclock = deviceProp.memoryClockRate * 2;
    int controller = deviceProp.memoryPoolsSupported;
    int width = (deviceProp.memoryBusWidth / 8);

    printf("%d * %d * %d = %d\n", dclock, controller, width, dclock * controller * width);

    int sm = deviceProp.multiProcessorCount;
    printf("%d SM", sm);
}
