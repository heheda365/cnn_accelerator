#include "loader.h"

int main(int argc, char const *argv[])
{
    loader load = loader();

    load.load_libsvm_data("MNIST/mnist.t", 1, 784, 10);
	// load.x_normalize(0, 'r');
    int cnt = 0;
    for(int i=0; i<28; i ++) {
        for(int j=0; j < 28; j ++) {
            // std::cout << load.x[cnt++];
            printf("%4d", (int)(load.x[cnt++]));
        }
        std::cout << endl;
    }

    for(int i=0; i<10; i ++) {
        std::cout << (int)load.y[i];
    }

    // std::cout << load.y[0];
    // printf("")
    return 0;
}
