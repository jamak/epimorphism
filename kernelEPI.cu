texture<float4, 2, cudaReadModeElementType> input_texture;
 
extern "C" {

  __global__ void kernel_fb(float4* out, ulong out_pitch, uchar4* pbo, float offset, int kernel_dim2)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int kernel_dim = gridDim.x*blockDim.x;
  
  //float2 z = make_float2(2.0 * (float) x / kernel_dim - 1.0, 2.0 * (float) y / kernel_dim - 1.0);

  //float4 f = make_float4((z.x + 1.0) / 2.0, (z.y + 1.0) / 2.0, offset, 0);
  float4 g = tex2D(input_texture, x, y);
  //float4 f = make_float4(x / (float)kernel_dim, y / (float)kernel_dim, offset, 0);  
  g.x += 0.001;
  if(g.x >= 1.0)
    g.x = 1.0;
  out[y * out_pitch + x] = g;//make_float4(g.x / 255.0, g.y / 255.0, g.z / 255.0, g.w / 255.0);
  pbo[y * kernel_dim + x] = make_uchar4(255.0 * g.x, 255.0 * g.y, 255.0 * g.z, 255.0 * g.w);  //make_uchar4(255.0 * f.x, 255.0 * f.y, 255.0 * f.z, 255.0 * f.w);
}

}
