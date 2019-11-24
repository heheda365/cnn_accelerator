#include <iostream>
#include <fstream>
#include "conv2d.h"
#include "linear.h"
#include "functional.h"
#include "config.h"
#include "params.h"
#include "loader.h"

void load_data(const char * path, char * ptr, unsigned int size) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    f.read(ptr, size);
    f.close();
}
void load_params() {
    load_data("params/conv_0_w.bin", (char *) conv_0_w, sizeof(conv_0_w));

    load_data("params/conv_1_w.bin", (char *) conv_1_w, sizeof(conv_1_w));
    load_data("params/conv_2_w.bin", (char *) conv_2_w, sizeof(conv_2_w));
    load_data("params/conv_3_w.bin", (char *) conv_3_w, sizeof(conv_3_w));
    load_data("params/linear_0_w.bin", (char *) linear_0_w, sizeof(linear_0_w));
    load_data("params/linear_1_w.bin", (char *) linear_1_w, sizeof(linear_1_w));

    load_data("params/bn_0_w.bin", (char *) bn_0_w, sizeof(bn_0_w));
    load_data("params/bn_1_w.bin", (char *) bn_1_w, sizeof(bn_1_w));
    load_data("params/bn_2_w.bin", (char *) bn_2_w, sizeof(bn_2_w));
    load_data("params/bn_3_w.bin", (char *) bn_3_w, sizeof(bn_3_w));
    load_data("params/bn_4_w.bin", (char *) bn_4_w, sizeof(bn_4_w));

    load_data("params/bn_0_b.bin", (char *) bn_0_b, sizeof(bn_0_b));
    load_data("params/bn_1_b.bin", (char *) bn_1_b, sizeof(bn_1_b));
    load_data("params/bn_2_b.bin", (char *) bn_2_b, sizeof(bn_2_b));
    load_data("params/bn_3_b.bin", (char *) bn_3_b, sizeof(bn_3_b));
    load_data("params/bn_4_b.bin", (char *) bn_4_b, sizeof(bn_4_b));

}

int main(int argc, char const *argv[])
{
    std::cout << "load params start \n";
    load_params();
    std::cout << "load params finish \n";
    
    float in0[1][28][28];
    load_data("data/test_data.bin", (char *) in0, sizeof(in0));
    std::cout << "load test data finish \n";

    std::cout << std::endl << "in0 = " << std::endl;
    for(int i=0; i < 28; i ++) {
        for(int j=0; j < 28; j ++) {
            std::cout << in0[0][i][j] << "  ";

        }
        std::cout << "\n";
    }
    // 神绝网络
    float out0[COV_0_OUT_CH][COV_0_OUT_ROW][COV_0_OUT_COL] = {0};
    conv2d<
            COV_0_IN_CH, 
            COV_0_IN_ROW,
            COV_0_IN_COL,
            COV_0_OUT_CH,
            COV_0_OUT_ROW,
            COV_0_OUT_COL,
            COV_0_K,
            COV_0_S,
            COV_0_P,
            COV_0_B>(
                in0,
                out0,
                conv_0_w,
                (float *) NULL
            );

    std::cout << std::endl << "out0 = " << std::endl;
    for(int i=0; i < COV_0_OUT_ROW; i ++) {
        for(int j=0; j < COV_0_OUT_COL; j ++) {
            std::cout << out0[0][i][j] << "  ";

        }
        std::cout << "\n";
    }
    conv_bn<
            COV_0_OUT_CH,
            COV_0_OUT_ROW,
            COV_0_OUT_COL> (
                out0,
                out0,
                bn_0_w,
                bn_0_b
            );
    std::cout << std::endl << "bn_0_b = " << std::endl;
    for(int i=0; i < COV_0_OUT_CH; i ++) {
        std::cout << bn_0_b[i] << " ";     
    }
    std::cout << "\n";

    std::cout << std::endl << "bn0 = " << std::endl;
    for(int i=0; i < COV_0_OUT_ROW; i ++) {
        for(int j=0; j < COV_0_OUT_COL; j ++) {
            std::cout << out0[0][i][j] << "  ";

        }
        std::cout << "\n";
    }

    conv_relu<COV_0_OUT_CH, COV_0_OUT_ROW, COV_0_OUT_COL>(out0, out0);

    float pool0_out[POOL_0_IN_CH][POOL_0_IN_ROW/POOL_0_IN_PO][POOL_0_IN_COL/POOL_0_IN_PO] = {0};
    max_pool2d< 
            POOL_0_IN_CH,
            POOL_0_IN_ROW,
            POOL_0_IN_COL, 
            POOL_0_IN_PO
            > (out0, pool0_out);

    float out1[COV_1_OUT_CH][COV_1_OUT_ROW][COV_1_OUT_COL] = {0};
    conv2d<
            COV_1_IN_CH, 
            COV_1_IN_ROW,
            COV_1_IN_COL,
            COV_1_OUT_CH,
            COV_1_OUT_ROW,
            COV_1_OUT_COL,
            COV_1_K,
            COV_1_S,
            COV_1_P,
            COV_1_B>(
                pool0_out,
                out1,
                conv_1_w,
                (float *) NULL
            );

    std::cout << std::endl << "out1 = " << std::endl;
    for(int i=0; i < COV_1_OUT_ROW; i ++) {
        for(int j=0; j < COV_1_OUT_ROW; j ++) {
            std::cout << out1[2][i][j] << "  ";

        }
        std::cout << "\n";
    }
    
    conv_bn<
            COV_1_OUT_CH,
            COV_1_OUT_ROW,
            COV_1_OUT_COL> (
                out1,
                out1,
                bn_1_w,
                bn_1_b
            );
    conv_relu<COV_1_OUT_CH, COV_1_OUT_ROW, COV_1_OUT_COL>(out1, out1);
    
    float out2[COV_2_OUT_CH][COV_2_OUT_ROW][COV_2_OUT_COL] = {0};
    conv2d<
            COV_2_IN_CH, 
            COV_2_IN_ROW,
            COV_2_IN_COL,
            COV_2_OUT_CH,
            COV_2_OUT_ROW,
            COV_2_OUT_COL,
            COV_2_K,
            COV_2_S,
            COV_2_P,
            COV_2_B>(
                out1,
                out2,
                conv_2_w,
                (float *) NULL
            );
    conv_bn<
            COV_2_OUT_CH,
            COV_2_OUT_ROW,
            COV_2_OUT_COL> (
                out2,
                out2,
                bn_2_w,
                bn_2_b
            );
    conv_relu<COV_2_OUT_CH, COV_2_OUT_ROW, COV_2_OUT_COL>(out2, out2);
    float pool2_out[POOL_1_IN_CH][POOL_1_IN_ROW/POOL_1_IN_PO][POOL_1_IN_COL/POOL_1_IN_PO] = {0};
    max_pool2d<
                POOL_1_IN_CH, 
                POOL_1_IN_ROW,
                POOL_1_IN_COL,
                POOL_1_IN_PO> (
                    out2,
                    pool2_out
                );
    
    float out3[COV_3_OUT_CH][COV_3_OUT_ROW][COV_3_OUT_COL] = {0};
    conv2d<
            COV_3_IN_CH, 
            COV_3_IN_ROW,
            COV_3_IN_COL,
            COV_3_OUT_CH,
            COV_3_OUT_ROW,
            COV_3_OUT_COL,
            COV_3_K,
            COV_3_S,
            COV_3_P,
            COV_3_B>(
                pool2_out,
                out3,
                conv_3_w,
                (float *) NULL
            );
    conv_bn<
            COV_3_OUT_CH,
            COV_3_OUT_ROW,
            COV_3_OUT_COL> (
                out3,
                out3,
                bn_3_w,
                bn_3_b
            );
    conv_relu<COV_3_OUT_CH, COV_3_OUT_ROW, COV_3_OUT_COL>(out3, out3);

    std::cout << std::endl << "out3 = " << std::endl;
    for(int i=0; i < COV_3_OUT_ROW; i ++) {
        for(int j=0; j < COV_3_OUT_ROW; j ++) {
            std::cout << out3[0][i][j] << "  ";

        }
        std::cout << "\n";
    }

    float linear_in[LINEAR_0_IN_N] = {0};
    view<COV_3_OUT_CH, COV_3_OUT_ROW, COV_3_OUT_COL>(out3, linear_in);



    float linear_0_out[LINEAR_0_OUT_N] = {0};
    linear<LINEAR_0_IN_N, LINEAR_0_OUT_N>(linear_in, linear_0_out, linear_0_w, (float *) NULL);
    linear_bn<LINEAR_0_OUT_N>(linear_0_out, linear_0_out, bn_4_w, bn_4_b);
    linear_relu<LINEAR_0_OUT_N>(linear_0_out, linear_0_out);

    std::cout << std::endl << "linear_0_out = " << std::endl;
    for(int i=0; i < LINEAR_0_OUT_N; i ++) {
        std::cout << linear_0_out[i] << "  ";
    }


    float linear_1_out[LINEAR_1_OUT_N] = {0};
    linear<LINEAR_1_IN_N, LINEAR_1_OUT_N>(linear_0_out, linear_1_out, linear_1_w, (float *) NULL);

    std::cout << std::endl << "linear_1_out = " << std::endl;
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        std::cout << linear_1_out[i] << "  ";
    }

    float res[LINEAR_1_OUT_N] = {0};
    log_softmax<LINEAR_1_OUT_N>(linear_1_out, res);

    std::cout << std::endl << "res = " << std::endl;
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        std::cout << res[i] << "  ";
    }

    int res_num = 0;
    float max = res[0];
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        if(max < res[i]) {
            max = res[i];
            res_num = i;
        }
    }
    std::cout << std::endl << "the numble is " << res_num << std::endl;

    return 0;
}
