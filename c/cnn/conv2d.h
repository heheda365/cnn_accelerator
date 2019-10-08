#pragma once
// #include "hls_stream.h"

template<int IN_CH, int IN_ROW, int IN_COL, int OUT_CH, int OUT_ROW, int OUT_COL, int K, int S, int P, int B>
void conv2d(float in[IN_CH][IN_ROW][IN_COL], float out[OUT_CH][OUT_ROW][OUT_COL], float w[OUT_CH][IN_CH][K][K]);

template<int IN_CH, int IN_ROW, int IN_COL, int OUT_CH, int OUT_ROW, int OUT_COL, int K, int S, int B>
void conv2d_nop(float in[IN_CH][IN_ROW][IN_COL], float out[OUT_CH][OUT_ROW][OUT_COL], float w[OUT_CH][IN_CH][K][K]);

template<int IN_CH, int IN_ROW, int IN_COL, int P>
void padding(float in[IN_CH][IN_ROW][IN_COL], float out[IN_CH][IN_ROW + 2*P][IN_COL+2*P]);

template<int A>
int add(int a, int b);



