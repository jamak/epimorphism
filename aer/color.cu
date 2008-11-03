extern "C" {
  __device__ float4 rg_swizzle(float4 v){
    return vec4(v.y, v.x, v.z, v.w);
  }

  __device__ float4 gb_swizzle(float4 v){
    return vec4(v.x, v.z, v.y, v.w);
  }
}
