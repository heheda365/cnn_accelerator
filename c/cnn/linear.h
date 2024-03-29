#pragma once

template<int IN_N, int OUT_N>
void linear(float in[IN_N], float out[OUT_N], const float w[OUT_N][IN_N], const float b[OUT_N]){
    for(int i=0; i < OUT_N; i ++) {
        out[i] = 0;
        for(int j=0; j < IN_N; j ++) {
            out[i] += in[j] * w[i][j];
        }
        out[i] += b[i];
    }
}


// int main(int argc, char const *argv[])
// {
//     float in[4] = {1, 1, 1, 1};
//     float out[4];
//     float w[4][4] = {
//         {1, 1, 1, 1},
//         {1, 2, 1, 1},
//         {1, 1, 1, 1},
//         {1, 1, 1, 1}
//     };
//     float b[4] = {1, 1, 1, 1};

//     linear<4, 4>(in, out, w, b);
//     for(int i=0; i < 4; i ++) {
//         std::cout << out[i] << " ";
//     }

//     return 0;
// }
