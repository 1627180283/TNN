#include "base.inc"
#include "io.inc"
#include "activation.inc"

__kernel void TransformToMatrixV(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,                     
                                 __write_only image2d_t matrix_v,
                                 __private const int in_height,
                                 __private const int in_width,
                                 __private const int in_channel,
                                 __private const int round_h,
                                 __private const int round_w,
                                 __private const int2 padding_wh){
    const int output_cw_idx = get_global_id(0); //c/4 w/2
    const int output_bh_idx  = get_global_id(1); //b h/2

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int c_block_idx = output_cw_idx / round_w;
    const int w_block_idx = output_cw_idx - mul24(c_block_idx, round_w);
    const int batch = output_bh_idx / round_h;
    const int h_block_idx = output_bh_idx - mul24(batch, round_h);
    
    const int width_start_idx = (w_block_idx << 1) - padding_wh.x;
    const int height_start_idx = (h_block_idx << 1) - padding_wh.y;

    const int4 width_idx = (int4)(width_start_idx) + (int4)(0,1,2,3);
    const int4 height_idx  = (int4)(height_start_idx) + (int4)(0,1,2,3);

    int4 in_wc_idx = mad24((int4)(c_block_idx), (int4)(in_width), width_idx);
    int4 in_bh_idx = mad24((int4)(batch), (int4)(in_height), height_idx);

    in_wc_idx = select(in_wc_idx, (int4)(-1), width_idx < (int4)(0) || width_idx >= (int4)(in_width));
    in_bh_idx = select(in_bh_idx, (int4)(-1), height_idx < (int4)(0) || height_idx >= (int4)(in_height));

    FLOAT4 in00 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s0));
    FLOAT4 in10 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s0));
    FLOAT4 in20 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s0));
    FLOAT4 in30 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s0));

    FLOAT4 in01 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s1));
    FLOAT4 in11 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s1));
    FLOAT4 in21 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s1));
    FLOAT4 in31 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s1));

    FLOAT4 in02 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s2));
    FLOAT4 in12 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s2));
    FLOAT4 in22 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s2));
    FLOAT4 in32 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s2));

    FLOAT4 in03 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s3));
    FLOAT4 in13 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s3));
    FLOAT4 in23 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s3));
    FLOAT4 in33 = RI_F(input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s3));

    FLOAT4 v00 = in00 - in02;
    FLOAT4 v10 = in10 - in12;
    FLOAT4 v20 = in20 - in22;
    FLOAT4 v30 = in30 - in32;

    FLOAT4 v01 = (FLOAT)0.5f * in01 + (FLOAT)0.5f * in02;
    FLOAT4 v11 = (FLOAT)0.5f * in11 + (FLOAT)0.5f * in12;
    FLOAT4 v21 = (FLOAT)0.5f * in21 + (FLOAT)0.5f * in22;
    FLOAT4 v31 = (FLOAT)0.5f * in31 + (FLOAT)0.5f * in32;

    FLOAT4 v02 = -(FLOAT)0.5f * in01 + (FLOAT)0.5f * in02;
    FLOAT4 v12 = -(FLOAT)0.5f * in11 + (FLOAT)0.5f * in12;
    FLOAT4 v22 = -(FLOAT)0.5f * in21 + (FLOAT)0.5f * in22;
    FLOAT4 v32 = -(FLOAT)0.5f * in31 + (FLOAT)0.5f * in32;

    FLOAT4 v03 = -in01 + in03;
    FLOAT4 v13 = -in11 + in13;
    FLOAT4 v23 = -in21 + in23;
    FLOAT4 v33 = -in31 + in33;

    WI_F(matrix_v, (int2)(output_cw_idx, mul24(16, output_bh_idx)), v00 - v20);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 1)), (FLOAT)0.5f * v10 + (FLOAT)0.5f * v20);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 2)), -(FLOAT)0.5f * v10 + (FLOAT)0.5f * v20);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 3)), -v10 + v30);

    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 4)), v01 - v21);    
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 5)), (FLOAT)0.5f * v11 + (FLOAT)0.5f * v21);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 6)), -(FLOAT)0.5f * v11 + (FLOAT)0.5f * v21);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 7)), -v11 + v31);

    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 8)), v02 - v22);    
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 9)), (FLOAT)0.5f * v12 + (FLOAT)0.5f * v22);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 10)), -(FLOAT)0.5f * v12 + (FLOAT)0.5f * v22);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 11)), -v12 + v32);

    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 12)), v03 - v23);    
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 13)), (FLOAT)0.5f * v13 + (FLOAT)0.5f * v23);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 14)), -(FLOAT)0.5f * v13 + (FLOAT)0.5f * v23);
    WI_F(matrix_v, (int2)(output_cw_idx, mad24(16, output_bh_idx, 15)), -v13 + v33);
}

__kernel void MatrixInnerProduct(GLOBAL_SIZE_2_DIMS __read_only image2d_t matrix_v,
                                 __read_only image2d_t matrix_u,
                                 __write_only image2d_t matrix_m,
                                 __private const int round_w,
                                 __private const int batch_round_h,
                                 __private const int out_channel_block, 
                                 __private const int in_channel_block){
    const int output_cw_idx = get_global_id(0); //c/4 w/2
    const int output_16_bh_idx  = get_global_id(1); //16 b h/2

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_16_bh_idx);

    const int c_block_idx = output_cw_idx / round_w;
    const int w_block_idx = output_cw_idx - mul24(c_block_idx, round_w);

    const int alpha = output_16_bh_idx % 16;
    //const int u_bh_idx = mul24(alpha, out_channel_block) + c_block_idx;
    const int u_bh_idx = mad24(c_block_idx, 16, alpha);

    FLOAT4 m = (FLOAT4)(0);

    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block; ++input_c_block_idx) {
        const int4 input_c_idx = (int4)(input_c_block_idx << 2) + (int4)(0,1,2,3);
        int v_cw_idx = mad24(input_c_block_idx, round_w, w_block_idx);

        FLOAT4 v_in = RI_F(matrix_v, SAMPLER, (int2)(v_cw_idx, output_16_bh_idx));
        FLOAT4 u_in0 = RI_F(matrix_u, SAMPLER, (int2)(input_c_idx.s0, u_bh_idx));
        FLOAT4 u_in1 = RI_F(matrix_u, SAMPLER, (int2)(input_c_idx.s1, u_bh_idx));
        FLOAT4 u_in2 = RI_F(matrix_u, SAMPLER, (int2)(input_c_idx.s2, u_bh_idx));
        FLOAT4 u_in3 = RI_F(matrix_u, SAMPLER, (int2)(input_c_idx.s3, u_bh_idx));

        m = mad(v_in.s0, u_in0, m);
        m = mad(v_in.s1, u_in1, m);
        m = mad(v_in.s2, u_in2, m);
        m = mad(v_in.s3, u_in3, m);
    }

    WI_F(matrix_m, (int2)(output_cw_idx, output_16_bh_idx), m);
    
}

__kernel void TransformFromMatrixM(GLOBAL_SIZE_2_DIMS __read_only image2d_t matrix_m,
                                 __read_only image2d_t bias,                     
                                 __write_only image2d_t output,
                                 __private const int round_w,
                                 __private const int round_h,
                                 __private const int out_width,
                                 __private const int out_height) {
        const int output_cw_idx = get_global_id(0); //c/4 w/2
        const int output_bh_idx  = get_global_id(1); //b h/2
        DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);
        const int c_block_idx = output_cw_idx / round_w;
        const int w_block_idx = output_cw_idx - mul24(c_block_idx, round_w);
        const int batch = output_bh_idx / round_h;
        const int h_block_idx = output_bh_idx - mul24(batch, round_h);

        FLOAT4 bias_value    = RI_F(bias, SAMPLER, (int2)(c_block_idx, 0));

        FLOAT4 m00  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mul24(16, output_bh_idx)));
        FLOAT4 m10  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 1)));
        FLOAT4 m20  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 2)));
        FLOAT4 m30  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 3)));
        FLOAT4 m01  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 4)));
        FLOAT4 m11  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 5)));
        FLOAT4 m21  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 6)));
        FLOAT4 m31  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 7)));
        FLOAT4 m02  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 8)));
        FLOAT4 m12  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 9)));
        FLOAT4 m22  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 10)));
        FLOAT4 m32  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 11)));
        FLOAT4 m03  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 12)));
        FLOAT4 m13  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 13)));
        FLOAT4 m23  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 14)));
        FLOAT4 m33  = RI_F(matrix_m, SAMPLER, (int2)(output_cw_idx, mad24(16, output_bh_idx, 15)));
        FLOAT4 out00  = m00 + m01 + m02;
        FLOAT4 out10  = m10 + m11 + m12;
        FLOAT4 out20  = m20 + m21 + m22;
        FLOAT4 out30  = m30 + m31 + m32;
        FLOAT4 out01  = m01 - m02 + m03;
        FLOAT4 out11  = m11 - m12 + m13;
        FLOAT4 out21  = m21 - m22 + m23;
        FLOAT4 out31  = m31 - m32 + m33;
        int2 ow = (int2)(w_block_idx << 1) + (int2)(0, 1);
        int2 oh = (int2)(h_block_idx << 1) + (int2)(0, 1);
        int2 ox = mad24((int2)(c_block_idx), (int2)(out_width), ow);
        int2 oy = mad24((int2)(batch), (int2)(out_height), oh);
        FLOAT4 res00  = bias_value + out00 + out10 + out20;
        res00 = ActivationProcess(res00);
        WI_F(output, (int2)(ox.s0, oy.s0), res00);
        if (ow.s1 < out_width && oh.s0 < out_height) {
            FLOAT4 res10  = bias_value + out10 - out20 + out30;
            res10 = ActivationProcess(res10);
            WI_F(output, (int2)(ox.s1, oy.s0), res10);
        }
        if (ow.s0 < out_width && oh.s1 < out_height) {
            FLOAT4 res01  = bias_value + out01 + out11 + out21;
            res01 = ActivationProcess(res01);
            WI_F(output, (int2)(ox.s0, oy.s1), res01);
        }
        if (ow.s1 < out_width && oh.s1 < out_height) {
            FLOAT4 res11  = bias_value + out11 - out21 + out31;
            res11 = ActivationProcess(res11);
            WI_F(output, (int2)(ox.s1, oy.s1), res11);
        }
}
