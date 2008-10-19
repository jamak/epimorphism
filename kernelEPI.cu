extern "C" {
__global__ void kernel1(float4* pos)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output vertex

    pos[y * (blockDim.x * gridDim.x) + x] = make_float4(1.0, 0.0, 0.0, 0.0);
}

}
