texture<float4, 2, cudaReadModeElementType> input_texture;

extern "C" {

  __global__ void kernel2(float4* pos, uchar4* out, ulong pitch, float offset, int kernel_dim)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // write output pixel    
    float4 f = make_float4(x / (float)kernel_dim, y / (float)kernel_dim, offset, 0);
    pos[y * pitch + x] = f;
    out[y * (blockDim.x * gridDim.x) + x] = make_uchar4(255.0 * f.x, 255.0 * f.y, 255.0 * f.z, 255.0 * f.w);
}

  __global__ void kernel_fb(float4* out, ulong out_pitch, uchar4* pbo, float offset)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int kernel_dim = gridDim.x*blockDim.y;
  
  //float2 z = make_float2(2.0 * (float) x / kernel_dim - 1.0, 2.0 * (float) y / kernel_dim - 1.0);

  //float4 f = make_float4((z.x + 1.0) / 2.0, (z.y + 1.0) / 2.0, offset, 0);
  //float4 g = tex2D(input_texture, (z.x + 1.0) / 2.0, (z.y + 1.0) / 2.0);
    float4 f = make_float4(x / (float)kernel_dim, y / (float)kernel_dim, offset, 0);
  out[y * out_pitch + x] = f;
  pbo[y * kernel_dim + x] = make_uchar4(255.0 * f.x, 255.0 * f.y, 255.0 * f.z, 255.0 * f.w);
}

}
