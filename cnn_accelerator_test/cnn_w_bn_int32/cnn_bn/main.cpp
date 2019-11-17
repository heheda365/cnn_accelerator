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

    load_data("params/conv_0_bn_inc.bin", (char *) bn_0_w, sizeof(bn_0_w));
    load_data("params/conv_1_bn_inc.bin", (char *) bn_1_w, sizeof(bn_1_w));
    load_data("params/conv_2_bn_inc.bin", (char *) bn_2_w, sizeof(bn_2_w));
    load_data("params/conv_3_bn_inc.bin", (char *) bn_3_w, sizeof(bn_3_w));
    load_data("params/linear_0_bn_inc.bin", (char *) bn_4_w, sizeof(bn_4_w));

    load_data("params/conv_0_bn_bias.bin", (char *) bn_0_b, sizeof(bn_0_b));
    load_data("params/conv_1_bn_bias.bin", (char *) bn_1_b, sizeof(bn_1_b));
    load_data("params/conv_2_bn_bias.bin", (char *) bn_2_b, sizeof(bn_2_b));
    load_data("params/conv_3_bn_bias.bin", (char *) bn_3_b, sizeof(bn_3_b));
    load_data("params/linear_0_bn_bias.bin", (char *) bn_4_b, sizeof(bn_4_b));

}

int main(int argc, char const *argv[])
{
    std::cout << "load params start \n";
    load_params();
    std::cout << "load params finish \n";

    for(int i = 0; i < 3; i ++) {
        for(int j=0; j < 3; j ++) {
            std::cout << (int)conv_0_w[0][0][i][j] << "  ";
        }
        std::cout << std::endl;
    }


    
    int32_t in0[1][28][28];
    load_data("data/test_data.bin", (char *) in0, sizeof(in0));
    std::cout << "load test data finish \n";

    int in0_int[1][28][28];
    for(int i=0; i < 28; i ++) {
        for(int j=0; j < 28; j ++) {
            in0_int[0][i][j] = (int) (in0[0][i][j] * 255);
        }
    }

    std::cout << std::endl << "in0 = " << std::endl;
    for(int i=0; i < 28; i ++) {
        for(int j=0; j < 28; j ++) {
            std::cout << in0_int[0][i][j] << "  ";

        }
        std::cout << "\n";
    }
    // 神绝网络
    int out0[COV_0_OUT_CH][COV_0_OUT_ROW][COV_0_OUT_COL] = {0};
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
                in0_int,
                out0,
                conv_0_w,
                (int32_t *) NULL
            );

    std::cout << std::endl << "out0 = " << std::endl;
    for(int i=0; i < COV_0_OUT_ROW; i ++) {
        for(int j=0; j < COV_0_OUT_COL; j ++) {
            std::cout << out0[0][i][j] << "  ";

        }
        std::cout << "\n";
    }
    conv_bn_qrelu<
            COV_0_OUT_CH,
            COV_0_OUT_ROW,
            COV_0_OUT_COL,
            COV_0_IN_BIT> (
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

    conv_relu<COV_0_OUT_CH, COV_0_OUT_ROW, COV_0_OUT_COL, COV_0_IN_BIT>(out0, out0);

    std::cout << std::endl << "out0_relu = " << std::endl;
    for(int i=0; i < COV_0_OUT_ROW; i ++) {
        for(int j=0; j < COV_0_OUT_COL; j ++) {
            std::cout << out0[0][i][j] << "  ";

        }
        std::cout << "\n";
    }

    int pool0_out[POOL_0_IN_CH][POOL_0_IN_ROW/POOL_0_IN_PO][POOL_0_IN_COL/POOL_0_IN_PO] = {0};
    max_pool2d< 
            POOL_0_IN_CH,
            POOL_0_IN_ROW,
            POOL_0_IN_COL, 
            POOL_0_IN_PO
            > (out0, pool0_out);
    
    std::cout << std::endl << "pool0 = " << std::endl;
    for(int i=0; i < POOL_0_IN_ROW/POOL_0_IN_PO; i ++) {
        for(int j=0; j < POOL_0_IN_COL/POOL_0_IN_PO; j ++) {
            std::cout << pool0_out[0][i][j] << "  ";

        }
        std::cout << "\n";
    }

    int out1[COV_1_OUT_CH][COV_1_OUT_ROW][COV_1_OUT_COL] = {0};
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
                (int32_t *) NULL
            );

    std::cout << std::endl << "out1 = " << std::endl;
    for(int i=0; i < COV_1_OUT_ROW; i ++) {
        for(int j=0; j < COV_1_OUT_ROW; j ++) {
            std::cout << out1[0][i][j] << "  ";

        }
        std::cout << "\n";
    }
    
    conv_bn<
            COV_1_OUT_CH,
            COV_1_OUT_ROW,
            COV_1_OUT_COL,
            COV_1_IN_BIT> (
                out1,
                out1,
                bn_1_w,
                bn_1_b
            );
    std::cout << std::endl << "out1_bn = " << std::endl;
    for(int i=0; i < COV_1_OUT_ROW; i ++) {
        for(int j=0; j < COV_1_OUT_ROW; j ++) {
            std::cout << out1[0][i][j] << "  ";

        }
        std::cout << "\n";
    }
    conv_relu<COV_1_OUT_CH, COV_1_OUT_ROW, COV_1_OUT_COL, COV_1_IN_BIT>(out1, out1);
    
    int out2[COV_2_OUT_CH][COV_2_OUT_ROW][COV_2_OUT_COL] = {0};
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
                (int32_t *) NULL
            );
    conv_bn<
            COV_2_OUT_CH,
            COV_2_OUT_ROW,
            COV_2_OUT_COL,
            COV_2_IN_BIT> (
                out2,
                out2,
                bn_2_w,
                bn_2_b
            );
    conv_relu<COV_2_OUT_CH, COV_2_OUT_ROW, COV_2_OUT_COL, COV_2_IN_BIT>(out2, out2);
    int pool2_out[POOL_1_IN_CH][POOL_1_IN_ROW/POOL_1_IN_PO][POOL_1_IN_COL/POOL_1_IN_PO] = {0};
    max_pool2d<
                POOL_1_IN_CH, 
                POOL_1_IN_ROW,
                POOL_1_IN_COL,
                POOL_1_IN_PO> (
                    out2,
                    pool2_out
                );
    
    int out3[COV_3_OUT_CH][COV_3_OUT_ROW][COV_3_OUT_COL] = {0};
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
                (int32_t *) NULL
            );
    conv_bn<
            COV_3_OUT_CH,
            COV_3_OUT_ROW,
            COV_3_OUT_COL,
            COV_3_IN_BIT> (
                out3,
                out3,
                bn_3_w,
                bn_3_b
            );
    conv_relu<COV_3_OUT_CH, COV_3_OUT_ROW, COV_3_OUT_COL, COV_3_IN_BIT>(out3, out3);

    std::cout << std::endl << "out3 = " << std::endl;
    for(int i=0; i < COV_3_OUT_ROW; i ++) {
        for(int j=0; j < COV_3_OUT_ROW; j ++) {
            std::cout << out3[0][i][j] << "  ";

        }
        std::cout << "\n";
    }

    int linear_in[LINEAR_0_IN_N] = {0};
    view<COV_3_OUT_CH, COV_3_OUT_ROW, COV_3_OUT_COL>(out3, linear_in);



    int linear_0_out[LINEAR_0_OUT_N] = {0};
    linear<LINEAR_0_IN_N, LINEAR_0_OUT_N>(linear_in, linear_0_out, linear_0_w, (int32_t *) NULL);
    linear_bn<LINEAR_0_OUT_N>(linear_0_out, linear_0_out, bn_4_w, bn_4_b);
    linear_relu<LINEAR_0_OUT_N>(linear_0_out, linear_0_out);

    std::cout << std::endl << "linear0_relu = " << std::endl;
    for(int i=0; i < LINEAR_0_OUT_N; i ++) {
        std::cout << linear_0_out[i] << "  ";
    }


    int linear_1_out[LINEAR_1_OUT_N] = {0};
    linear<LINEAR_1_IN_N, LINEAR_1_OUT_N>(linear_0_out, linear_1_out, linear_1_w, (int32_t *) NULL);

    std::cout << std::endl << "linear_1_out = " << std::endl;
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        std::cout << linear_1_out[i] << "  ";
    }

    int32_t res[LINEAR_1_OUT_N] = {0};
    log_softmax<LINEAR_1_OUT_N>(linear_1_out, res);

    std::cout << std::endl << "res = " << std::endl;
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        std::cout << res[i] << "  ";
    }

    int res_num = 0;
    int32_t max = res[0];
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        if(max < res[i]) {
            max = res[i];
            res_num = i;
        }
    }
    std::cout << std::endl << "the numble is " << res_num << std::endl;

    return 0;
}
