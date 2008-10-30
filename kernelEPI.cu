texture<float4, 2, cudaReadModeElementType> input_texture;
 
extern "C" {

  __device__ float4 operator^(const float4 v1, const float4 v2){
    return make_float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
  }

  __device__ float4 operator%(const float m, const float4 v1){
    return make_float4(m * v1.x, m * v1.y, m * v1.z, m * v1.w);
  }

  __device__ float2 operator+(const float2 z1, const float2 z2){
    return make_float2(z1.x + z2.x, z1.y + z2.y);
  }

  __device__ float2 operator*(const float2 z1, const float2 z2){
    return make_float2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
  }

  __device__ float4 get_seed(float2 z)
  {
    float d = 0.1;
    
    if(z.x < (-1.0 + d) || z.x > (1.0 - d) || z.y < (-1.0 + d) || z.y > (1.0 - d)){
      float v;
      if(z.x > -1.0 * z.y)
	v = max(z.x, z.y);
      else
	v = min(z.x, z.y);
    
      return make_float4((abs(v) - (1.0 - d)) / d, 0.0, 0.0, 1.0);
    }else{
      return make_float4(0.0, 0.0, 0.0, 0.0);
    }
  }

  __device__ float2 transform(float2 z){
    //return make_float2(2.5 * z.x, 2.5 * z.y);
    float r = z.x * z.x + z.y * z.y;
    return make_float2(z.x / r, -1.0 * z.y / r);
  }

  __device__ float4 colorify(float4 v){
    return make_float4(v.y, v.x, v.z, v.w);
  }

  __device__ float2 lattice_reduce(float2 z){
    z = make_float2((z.x + 1.0) / 2.0, (z.y + 1.0) / 2.0);
    return make_float2((z.x - floorf(z.x)) * 2.0 - 1.0, (z.y - floorf(z.y)) * 2.0 - 1.0);
  }

  __global__ void kernel_fb(float4* out, ulong out_pitch, uchar4* pbo, float offset, int kernel_dim2)
  {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    float2 z = make_float2(2.0 * (x + 0.5) / kernel_dim2 - 1.0, 2.0 * (y + 0.5) / kernel_dim2 - 1.0);    

    z = lattice_reduce(transform(z));   
    
    float4 frame = tex2D(input_texture, (z.x + 1.0) / 2.0, (z.y + 1.0) / 2.0);
    
    float4 seed = get_seed(z);
    
    float4 result = colorify((seed.w % seed) ^ ((1.0 - seed.w) % frame));     

    //int i = 0;
    //for(i = 0; i < 500; i++)
      //  result.y += 0.000001;

    // set output variable
    out[y * out_pitch + x] = result;
    pbo[y * kernel_dim2 + x] = make_uchar4(255.0 * result.x, 255.0 * result.y, 255.0 * result.z, 255.0 * result.w); 
  }

}
