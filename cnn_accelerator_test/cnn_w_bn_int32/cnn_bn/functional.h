#pragma once

#include <math.h>

template<int IN_CH, int IN_ROW, int IN_COL, int IN_BIT>
void conv_relu(int in[IN_CH][IN_ROW][IN_COL], int out[IN_CH][IN_ROW][IN_COL]) {
    for(int ch=0; ch < IN_CH; ch ++) {
        for(int row=0; row < IN_ROW; row ++) {
            for(int col=0; col < IN_COL; col ++) {
                if(in[ch][row][col] > ((1 << IN_BIT) - 1) * 3) {
                    out[ch][row][col] = 15;
                }else if( in[ch][row][col] >= 0) {
                    out[ch][row][col] = (int)(in[ch][row][col] * 1.0 * ((1 << 4) - 1) / ((1 << IN_BIT) - 1) / 3 + 0.5);
                } else {
                     out[ch][row][col] = 0;
                }
                
            }
        }
    }
}
template<int LEN>
void linear_relu(int in[LEN], int out[LEN]) {
    for(int i=0; i < LEN; i ++) {
        if(in[i] > 3  * 15) {
            out[i] = 15;
        }else if(in[i] >= 0) {
            out[i] = int(in[i] * 1.0 / 3 + 0.5);
        } else {
            out[i] = 0;
        }
    }
}
template<int IN_CH, int IN_ROW, int IN_COL, int PO>
void max_pool2d(int in[IN_CH][IN_ROW][IN_COL], int out[IN_CH][IN_ROW / PO][IN_COL / PO]) {
    for(int row=0; row <= IN_ROW - PO; row += PO) {
        for(int col=0; col <= IN_COL - PO; col += PO) {
            for(int ch=0; ch < IN_CH; ch ++) {
                int max =in[ch][row][col]; 
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
void softmax(int in[LEN], int out[LEN]) {
    int sum = 0;
    for(int i=0; i < LEN; i ++) {
        sum += in[i];
    }
    for(int i=0; i < LEN; i ++) {
        out[i] = in[i] / sum;
    }
}

template<int LEN>
void log_softmax(int in[LEN], float out[LEN]) {
    int sum_exp = 0;
    for(int i=0; i < LEN; i ++) {
        sum_exp += exp(in[i] * 1.0 / 3 / 15);
    }
    for(int i=0; i < LEN; i ++) {
        out[i] = log(exp(in[i] * 1.0 / 3 / 15) / sum_exp);
    }
}
template<int CH, int ROW, int COL>
void view(int in[CH][ROW][COL], int out[CH*ROW*COL]) {
    int cnt = 0;
    for(int ch=0; ch < CH; ch ++) {
        for(int row=0; row < ROW; row ++) {
            for(int col=0; col < COL; col ++) {
                out[cnt ++] = in[ch][row][col];
            }
        }
    }
}
template<int CH, int ROW, int COL, int IN_BIT>
void conv_bn(int in[CH][ROW][COL], int out[CH][ROW][COL], int w[CH], int b[CH]) {
    for(int ch=0; ch < CH; ch ++) {
        for(int row=0; row < ROW; row ++) {
            for(int col=0; col < COL; col ++) {
                out[ch][row][col] = (int)(in[ch][row][col] * w[ch] + b[ch] * 3 * ((1 << IN_BIT) - 1));
            }
        }
    }
}
template<int LEN>
void linear_bn(int in[LEN], int out[LEN], int w[LEN], int b[LEN]) {
    for(int i=0; i < LEN; i ++) {
        out[i] = (int)(in[i] * w[i] + b[i] * 3 * 15);
    }
}

template<int BIT>
int qrelu_search(int target, int inc, int bias) {
    target = target + bias;
    int index = 1 << (BIT - 1);
    int mid = inc << (BIT - 1);
    for(int i=BIT-2; i >= 0; i --) {
        int inc_shift = inc << i;
        int one_shift = 1 << i;
        if(target < mid) {
            mid -= inc_shift;
            index -= one_shift; 
        } else if(mid < target){
            mid += inc_shift;
            index += one_shift;
        }
    }
    if(target < mid) {
        index --;
    }
    return index;
}

template<int IN_CH, int IN_ROW, int IN_COL, int IN_BIT, int OUT_BIT>
void conv_bn_qrelu(int in[IN_CH][IN_ROW][IN_COL], int out[IN_CH][IN_ROW][IN_COL], int w[IN_CH], int b[IN_CH]) {
    
    for(int ch=0; ch < IN_CH; ch ++) {
        for(int row=0; row < IN_ROW; row ++) {
            for(int col=0; col < IN_COL; col ++) {
                out[ch][row][col] = qrelu_search<OUT_BIT>(in[ch][row][col], w[ch], b[ch]);
            }
        }
    }
}

template<int LEN, int IN_BIT, int OUT_BIT>
void linear_bn_qrelu(int in[LEN], int out[LEN], int w[LEN], int b[LEN]) {
    for(int i=0; i < LEN; i ++) {
       
        out[i] = qrelu_search<OUT_BIT>(in[i], w[i], b[i]);
    }
}
// int main(int argc, char const *argv[])
// {
//     int in[2][4][4] = {{{1, 2, 1, 3},
//                         {1, 1, 1, 4},
//                         {2, 3, 4, 5},
//                         {2, 3, 4, 5}},
                        
//                         {{1, 2, 1, 3},
//                         {1, 1, 1, 4},
//                         {2, 3, 4, 5},
//                         {2, 3, 4, 5}}
//                         };
//     int out[2][2][2] = {0};

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



