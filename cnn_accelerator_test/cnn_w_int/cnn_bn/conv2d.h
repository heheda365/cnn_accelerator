
#pragma once
#include <iostream>

template<int IN_CH, int IN_ROW, int IN_COL, int P>
void padding(int in[IN_CH][IN_ROW][IN_COL], int out[IN_CH][IN_ROW + 2*P][IN_COL+2*P]) {
    for(int ch=0; ch < IN_CH; ch ++) {
        for(int row=0; row < P; row ++) {
            for(int col=0; col < IN_COL + 2*P; col ++) {
                out[ch][row][col] = 0;
            }
        }
        for(int row=P; row < IN_ROW + P; row ++) {
            for(int col = 0; col < P; col ++) {
                out[ch][row][col] = 0;
            }
            for(int col=P; col < IN_COL + P; col ++) {
                out[ch][row][col] = in[ch][row-P][col-P];
            }
            for(int col=IN_COL + P; col < IN_COL + 2*P; col ++) {
                out[ch][row][col] = 0;
            }
        }
        for(int row=IN_ROW + P; row < IN_ROW + 2*P; row ++) {
            for(int col=0; col < IN_COL + 2*P; col ++) {
                out[ch][row][col] = 0;
            }
        }
    }
}

/**
 *  不带padding的卷积计算
 * 
 */
template<int IN_CH, int IN_ROW, int IN_COL, int OUT_CH, int OUT_ROW, int OUT_COL, int K, int S, int B>
void conv2d_nop(int in[IN_CH][IN_ROW][IN_COL], int out[OUT_CH][OUT_ROW][OUT_COL], const uint8_t w[OUT_CH][IN_CH][K][K], const uint8_t b[OUT_CH]) {
    // IN ROW
    for(int in_row=0; in_row < IN_ROW - K + 1; in_row += S) {
        // IN COL
        for(int in_col=0; in_col < IN_COL - K + 1; in_col += S) {
            // OUT CH
            for(int out_ch=0; out_ch < OUT_CH; out_ch ++) {
                // K ROW
                for(int k_row=0; k_row < K; k_row ++) {
                    // K COL
                    for(int k_col=0; k_col < K; k_col ++) {
                        // IN CH  K IN CH
                        for(int in_ch=0; in_ch < IN_CH; in_ch ++){
                            out[out_ch][in_row][in_col] += ((in[in_ch][in_row + k_row][in_col + k_col] << 1) * w[out_ch][in_ch][k_row][k_col]
                                                            - (in[in_ch][in_row + k_row][in_col + k_col] << 2) + in[in_ch][in_row + k_row][in_col + k_col]
                                                            ); 
                        }
                    }
                }
            }
        }
    }
    if(B != 0) {
        for(int ch = 0; ch < OUT_CH; ch ++) {
            for(int row = 0; row < OUT_ROW; row ++) {
                for(int col = 0; col < OUT_COL; col ++) {
                    out[ch][row][col] += b[ch];
                }
            }
        }
    }
}

template<int IN_CH, int IN_ROW, int IN_COL, int OUT_CH, int OUT_ROW, int OUT_COL, int K, int S, int P, int B>
void conv2d(int in[IN_CH][IN_ROW][IN_COL], int out[OUT_CH][OUT_ROW][OUT_COL], const uint8_t w[OUT_CH][IN_CH][K][K], const uint8_t b[OUT_CH]){
    // if(P != 0) {
    int out_padding[IN_CH][IN_ROW + 2*P][IN_COL + 2*P];
    padding<IN_CH, IN_ROW, IN_COL, P>(in, out_padding);
    // for(int i=0; i < IN_ROW + 2*P; i ++) {
    //     for(int j=0; j < IN_COL + 2*P; j ++) {
    //         std::cout << out_padding[0][i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    conv2d_nop<IN_CH, IN_ROW + 2*P, IN_COL + 2*P, OUT_CH, OUT_ROW, OUT_COL, K, S, B>(out_padding, out, w, b);
}



// template<int A>
// int add(int a, int b) {
//     return a + b + A;
// }

// int main(int argc, char const *argv[])
// {
//     int in[2][4][4] = {
//                         {{1, 1, 1, 1 },
//                         {1, 1, 1, 1},
//                         {1, 1, 1 ,1},
//                         {1, 1, 1 ,1}},

//                         {{1, 1, 1, 1 },
//                         {1, 1, 1, 1},
//                         {1, 1, 1 ,1},
//                         {1, 1, 1 ,1}},
                        
//                         };
//     int out[1][2][2] = {0};
//     uint8_t[1][2][3][3] = {{
//                     {{1, 1, 1}, 
//                     {1, 1, 1},
//                     {1, 1, 1}},
                    
//                     {{1, 1, 1}, 
//                     {1, 1, 1},
//                     {1, 1, 1}}
//                     }};
//     conv2d_nop<2, 4, 4, 1, 2, 2, 3, 1, 0>(in, out, w);
//     for(int i=0; i < 2; i ++) {
//         for(int j=0; j < 2; j ++) {
//             std::cout << out[0][i][j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     int in1[2][4][4] = {
//                         {{1, 1, 1, 1 },
//                         {1, 1, 1, 1},
//                         {1, 1, 1 ,1},
//                         {1, 1, 1 ,1}},

//                         {{1, 1, 1, 1 },
//                         {1, 1, 1, 1},
//                         {1, 1, 1 ,1},
//                         {1, 1, 1 ,1}},
//                         };
    

//     int out1[2][8][8];
//     padding<2, 4, 4, 2>(in1, out1);
//     std::cout << "\n\n";

//     for(int i=0; i < 8; i ++) {
//         for(int j=0; j < 8; j ++) {
//             std::cout << out1[0][i][j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     for(int i=0; i < 8; i ++) {
//         for(int j=0; j < 8; j ++) {
//             std::cout << out1[1][i][j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     return 0;
// }

