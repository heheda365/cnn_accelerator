#pragma once
#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;


template<int IN_CH, int IN_ROW, int IN_COL, int P>
void padding(float in[IN_CH][IN_ROW][IN_COL], float out[IN_CH][IN_ROW + 2*P][IN_COL+2*P]) {
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
void conv2d_nop(float in[IN_CH][IN_ROW][IN_COL], float out[OUT_CH][OUT_ROW][OUT_COL], const float w[OUT_CH][IN_CH][K][K], const float b[OUT_CH]) {
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
                            out[out_ch][in_row][in_col] += in[in_ch][in_row + k_row][in_col + k_col] * w[out_ch][in_ch][k_row][k_col]; 
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
void conv2d(float in[IN_CH][IN_ROW][IN_COL], float out[OUT_CH][OUT_ROW][OUT_COL], const float w[OUT_CH][IN_CH][K][K], const float b[OUT_CH]){
    float out_padding[IN_CH][IN_ROW + 2*P][IN_COL + 2*P];
    padding<IN_CH, IN_ROW, IN_COL, P>(in, out_padding);
    conv2d_nop<IN_CH, IN_ROW + 2*P, IN_COL + 2*P, OUT_CH, OUT_ROW, OUT_COL, K, S, B>(out_padding, out, w, b);
}


