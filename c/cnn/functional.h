#pragma once

template<int IN_CH, int IN_ROW, int IN_COL>
void conv_relu(float in[IN_CH][IN_ROW][IN_COL], float out[IN_CH][IN_ROW][IN_COL]);

template<int LEN>
void linear_relu(float in[LEN], float out[LEN]);

template<int IN_CH, int IN_ROW, int IN_COL, int PO>
void max_pool2d(float in[IN_CH][IN_ROW][IN_COL], float out[IN_CH][IN_ROW / PO][IN_COL / PO]);

template<int LEN>
void softmax(float in[LEN], float out[LEN]);

template<int LEN>
void log_softmax(float in[LEN], float out[LEN]);