extern "C" {
  __global__ void kernel1(float4* pos, ulong pitch, float offset)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output pixel

    pos[y * (pitch / sizeof(float4)) + x] = make_float4(x / 1000.0, y / 1000.0, offset, 0);
}


  __global__ void kernel2(float4* pos, uchar4* out, ulong pitch, float offset)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output pixel
    
    float4 f = make_float4(x / 1000.0, y / 1000.0, offset, 0);
    pos[y * (pitch / sizeof(float4)) + x] = f;
    out[y * (blockDim.x * gridDim.x) + x] = make_uchar4(255.0 * f.x, 255.0 * f.y, 255.0 * f.z, 255.0 * f.w);
}


  __global__ void kernel_copy(float4* in, ulong pitch, uchar4* out, int dim)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output pixel

    float4 f = in[y * (pitch / sizeof(float4)) + x];

    out[y * (blockDim.x * gridDim.x) + x] = make_uchar4(255.0 * f.x, 255.0 * f.y, 255.0 * f.z, 255.0 * f.w);
}

}
