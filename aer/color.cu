
__device__ float4 rg_swizzle(float4 v){
  return vec4(v.y, v.x, v.z, v.w);
}

__device__ float4 gb_swizzle(float4 v){
  return vec4(v.x, v.z, v.y, v.w);
}

__device__ float4 rgb_swizzle(float4 v){
  return vec4(v.y, v.z, v.x, v.w);
}

__device__ float4 rotate(float4 v){
  v = RGBtoHSV(v);
  v.x += _COLOR_DHUE;
  return HSVtoRGB(v);
}
