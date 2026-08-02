// image_process.cl
__kernel void transform_and_brighten(
    __global const uchar4* input_img,
    __global uchar4* output_img,
    const int in_width,
    const int in_height,
    const int out_width,
    const int out_height,
    __constant float* inv_matrix,
    const float brightness,
    const float contrast
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= out_width || y >= out_height) return;

    // 计算逆透视映射：通过目标图坐标 (x,y) 找回原图坐标 (u,v)
    float u_w = inv_matrix[0] * x + inv_matrix[1] * y + inv_matrix[2];
    float v_w = inv_matrix[3] * x + inv_matrix[4] * y + inv_matrix[5];
    float w   = inv_matrix[6] * x + inv_matrix[7] * y + inv_matrix[8];

    float src_x_f = u_w / w;
    float src_y_f = v_w / w;

    // 最近邻插值
    int src_x = (int)(src_x_f + 0.5f);
    int src_y = (int)(src_y_f + 0.5f);

    int out_index = y * out_width + x;

    if (src_x >= 0 && src_x < in_width && src_y >= 0 && src_y < in_height) {
        int src_index = src_y * in_width + src_x;
        uchar4 pixel = input_img[src_index];

        // 提亮与对比度增强 (简单的白板优化)
        float r = (pixel.x - 128.0f) * contrast + 128.0f + brightness;
        float g = (pixel.y - 128.0f) * contrast + 128.0f + brightness;
        float b = (pixel.z - 128.0f) * contrast + 128.0f + brightness;

        pixel.x = (uchar)clamp((int)r, 0, 255);
        pixel.y = (uchar)clamp((int)g, 0, 255);
        pixel.z = (uchar)clamp((int)b, 0, 255);

        output_img[out_index] = pixel;
    } else {
        output_img[out_index] = (uchar4)(255, 255, 255, 255); // 越界填白色
    }
}
