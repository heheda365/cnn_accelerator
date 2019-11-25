#include <iostream>
#include <fstream>
#include "conv2d.h"
#include "linear.h"
#include "functional.h"
#include "config.h"
#include "params.h"
// #include "loader.h"
#include <stdint.h>
#include <cstdlib>

void load_data(const char * path, char * ptr, unsigned int size) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if(!f) {
        std::cout <<"no such file,please check the file name!/n";
        exit(0);
    }
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
/**
 * mnist conv net
 * ???????? ????
 */
int mnist_conv_net(int in[1][28][28]) {
    int out0[COV_0_OUT_CH][COV_0_OUT_ROW][COV_0_OUT_COL] = {0};
    
    // conv 0 
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
                in,
                out0,
                conv_0_w,
                (int *) NULL
            );

    std::cout << "conv1 out \n";
    for(int i=0; i < 28; i ++) {
        for (int j=0; j < 28; j ++) {
            std::cout << out0[2][i][j] << "  ";
        }
        std::cout << "\n";
    }

    conv_bn_qrelu<
            COV_0_OUT_CH,
            COV_0_OUT_ROW,
            COV_0_OUT_COL,
            COV_0_IN_BIT,
            COV_0_A_BIT> (
                out0,
                out0,
                bn_0_w,
                bn_0_b
            );
    int pool0_out[POOL_0_IN_CH][POOL_0_IN_ROW/POOL_0_IN_PO][POOL_0_IN_COL/POOL_0_IN_PO] = {0};
    max_pool2d< 
            POOL_0_IN_CH,
            POOL_0_IN_ROW,
            POOL_0_IN_COL, 
            POOL_0_IN_PO
            > (out0, pool0_out);
    
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
                (int *) NULL
            );

    conv_bn_qrelu<
            COV_1_OUT_CH,
            COV_1_OUT_ROW,
            COV_1_OUT_COL,
            COV_1_IN_BIT,
            COV_1_A_BIT> (
                out1,
                out1,
                bn_1_w,
                bn_1_b
            );
    
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
                (int *) NULL
            );
    conv_bn_qrelu<
            COV_2_OUT_CH,
            COV_2_OUT_ROW,
            COV_2_OUT_COL,
            COV_2_IN_BIT,
            COV_2_A_BIT> (
                out2,
                out2,
                bn_2_w,
                bn_2_b
            );
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
                (int *) NULL
            );
    conv_bn_qrelu<
            COV_3_OUT_CH,
            COV_3_OUT_ROW,
            COV_3_OUT_COL,
            COV_3_IN_BIT,
            COV_3_A_BIT> (
                out3,
                out3,
                bn_3_w,
                bn_3_b
            );
    int linear_in[LINEAR_0_IN_N] = {0};
    view<COV_3_OUT_CH, COV_3_OUT_ROW, COV_3_OUT_COL>(out3, linear_in);

    int linear_0_out[LINEAR_0_OUT_N] = {0};
    linear<LINEAR_0_IN_N, LINEAR_0_OUT_N>(linear_in, linear_0_out, linear_0_w, (int *) NULL);
    linear_bn_qrelu<LINEAR_0_OUT_N, LINEAR_0_IN_BIT, LINEAR_0_A_BIT>(linear_0_out, linear_0_out, bn_4_w, bn_4_b);

    int linear_1_out[LINEAR_1_OUT_N] = {0};
    linear<LINEAR_1_IN_N, LINEAR_1_OUT_N>(linear_0_out, linear_1_out, linear_1_w, (int *) NULL);


    float res[LINEAR_1_OUT_N] = {0};
    log_softmax<LINEAR_1_OUT_N>(linear_1_out, res);


    int res_num = 0;
    float max = res[0];
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        if(max < res[i]) {
            max = res[i];
            res_num = i;
        }
    }
    return res_num;
}

int main(int argc, char const *argv[])
{
    std::cout << "load params start \n";
    load_params();
    std::cout << "load params finish \n";




    // loader load = loader();
    // load.load_libsvm_data("../../../c/load_dataset/MNIST/mnist.t", 10000, 784, 10);
    // std::cout << "load test data finish \n";

    // int (* in)[1][28][28] = new int[10000][1][28][28];
    // int * y = new int[10000];
    // int cnt_x = 0;

    // for(int i=0; i<10000; i ++) {
    //     for(int row=0; row < 28; row ++) {
    //         for(int col=0; col < 28; col ++) {
    //             in[i][0][row][col] = load.x[cnt_x ++];
    //         }
    //     }
    // }
    // int cnt_y = 0;
    // for(int i=0; i < 10000; i ++) {
    //     int temp_y = 0;
    //     for(int j=0; j < 10; j ++) {
    //         if(load.y[cnt_y ++] > 0.5) {
    //             temp_y = j;
    //         }
    //     }
    //     y[i] = temp_y;
    // }
    // std::cout << "data preproccess finish\n";

    // int accu = 0;
    // for(int i=0; i < 10000; i ++) {
    //     int predict_num = mnist_conv_net(in[i]);
    //     if(predict_num == y[i]) {
    //         accu ++;
    //     }
    //     std::cout << "count : " << i << "  accuracy : " << accu << "\n";
    // }
    // std::cout << std::endl << "the accuracy is " << accu << std::endl;


    float data_raw[28][28];
    load_data("data/test_data.bin", (char *)data_raw, sizeof(data_raw));

    // 输入数据c仿真时两种写法都可以
    int data[1][28][28];
    // hls::stream<ap_uint<8>> in_stream("in_stream");
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            data[0][i][j] = (int)(data_raw[i][j] * 255);
            // in_stream.write(data[i][j]);
            std::cout << (int)data[0][i][j] << "  ";
        }
        std::cout << "\n";
    }
    mnist_conv_net(data);

    return 0;
}
