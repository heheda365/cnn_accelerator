#include <iostream>
#include "conv2d.h"

int main(int argc, char const *argv[])
{
    float in[2][4][4] = {
                        {{1, 1, 1, 1 },
                        {1, 1, 1, 1},
                        {1, 1, 1 ,1},
                        {1, 1, 1 ,1}},

                        {{1, 1, 1, 1 },
                        {1, 1, 1, 1},
                        {1, 1, 1 ,1},
                        {1, 1, 1 ,1}},
                        
                        };
    float out[1][2][2] = {0};
    float w[1][2][3][3] = {{
                    {{1, 1, 1}, 
                    {1, 1, 1},
                    {1, 1, 1}},
                    
                    {{1, 1, 1}, 
                    {1, 1, 1},
                    {1, 1, 1}}
                    }};
    conv2d_nop<2, 4, 4, 1, 2, 2, 3, 1, 0>(in, out, w);
    for(int i=0; i < 2; i ++) {
        for(int j=0; j < 2; j ++) {
            std::cout << out[0][i][j] << " ";
        }
        std::cout << std::endl;
    }

    // float in1[2][4][4] = {
    //                     {{1, 1, 1, 1 },
    //                     {1, 1, 1, 1},
    //                     {1, 1, 1 ,1},
    //                     {1, 1, 1 ,1}},

    //                     {{1, 1, 1, 1 },
    //                     {1, 1, 1, 1},
    //                     {1, 1, 1 ,1},
    //                     {1, 1, 1 ,1}},
    //                     };
    

    // float out1[2][8][8];
    // padding<2, 4, 4, 2>(in1, out1);
    // std::cout << "\n\n";

    // for(int i=0; i < 8; i ++) {
    //     for(int j=0; j < 8; j ++) {
    //         std::cout << out1[0][i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // for(int i=0; i < 8; i ++) {
    //     for(int j=0; j < 8; j ++) {
    //         std::cout << out1[1][i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << add<7>(3, 4);
    return 0;
}
