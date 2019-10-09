#pragma once

#include <math.h>

template<int IN_CH, int IN_ROW, int IN_COL>
void conv_relu(float in[IN_CH][IN_ROW][IN_COL], float out[IN_CH][IN_ROW][IN_COL]) {
    for(int ch=0; ch < IN_CH; ch ++) {
        for(int row=0; row < IN_ROW; row ++) {
            for(int col=0; col < IN_COL; col ++) {
                if( in[ch][row][col] >= 0) {
                    out[ch][row][col] = in[ch][row][col];
                } else {
                     out[ch][row][col] = 0;
                }
                
            }
        }
    }
}
template<int LEN>
void linear_relu(float in[LEN], float out[LEN]) {
    for(int i=0; i < LEN; i ++) {
        if(in[i] >= 0) {
            out[i] = in[i];
        } else {
            out[i] = 0;
        }
    }
}
template<int IN_CH, int IN_ROW, int IN_COL, int PO>
void max_pool2d(float in[IN_CH][IN_ROW][IN_COL], float out[IN_CH][IN_ROW / PO][IN_COL / PO]) {
    for(int row=0; row <= IN_ROW - PO; row += PO) {
        for(int col=0; col <= IN_COL - PO; col += PO) {
            for(int ch=0; ch < IN_CH; ch ++) {
                float max =in[ch][row][col]; 
                for(int pi=0; pi < PO; pi ++) {
                    for(int pj=0; pj < PO; pj ++) {
                        if(in[ch][row + pi][col + pj] > max){
                            max = in[ch][row + pi][col + pj];
                        }
                    }
                }
                out[ch][row/PO][col/PO] = max;
            }
        }
    }
}
template<int LEN>
void softmax(float in[LEN], float out[LEN]) {
    float sum = 0;
    for(int i=0; i < LEN; i ++) {
        sum += in[i];
    }
    for(int i=0; i < LEN; i ++) {
        out[i] = in[i] / sum;
    }
}

template<int LEN>
void log_softmax(float in[LEN], float out[LEN]) {
    float sum_exp = 0;
    for(int i=0; i < LEN; i ++) {
        sum_exp += exp(in[i]);
    }
    for(int i=0; i < LEN; i ++) {
        out[i] = exp(in[i]) / sum_exp;
    }
}
template<int CH, int ROW, int COL>
void view(float in[CH][ROW][COL], float out[CH*ROW*COL]) {
    int cnt = 0;
    for(int ch=0; ch < CH; ch ++) {
        for(int row=0; row < ROW; row ++) {
            for(int col=0; col < COL; col ++) {
                out[cnt ++] = in[ch][row][col];
            }
        }
    }
}

// int main(int argc, char const *argv[])
// {
//     float in[2][4][4] = {{{1, 2, 1, 3},
//                         {1, 1, 1, 4},
//                         {2, 3, 4, 5},
//                         {2, 3, 4, 5}},
                        
//                         {{1, 2, 1, 3},
//                         {1, 1, 1, 4},
//                         {2, 3, 4, 5},
//                         {2, 3, 4, 5}}
//                         };
//     float out[2][2][2] = {0};

//     max_pool2d<2, 4, 4, 2>(in, out);
    
//     // 
//     for(int i=0; i < 2; i ++) {
//         for(int j=0; j < 2; j ++) {
//             for(int k=0; k < 2; k ++) {
//                 std::cout << out[i][j][k] << " ";
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }
//     return 0;
// }



