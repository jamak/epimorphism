__device__ float4 rgb_id(float4 v, float2 z_z){
  return v;
}

__device__ float4 rg_swizzle(float4 v, float2 z_z){
  return vec4(v.y, v.x, v.z, v.w);
}

__device__ float4 gb_swizzle(float4 v, float2 z_z){
  return vec4(v.x, v.z, v.y, v.w);
}


__device__ float4 rgb_swizzle(float4 v, float2 z_z){
  return vec4(v.y, v.z, v.x, v.w);
}


__device__ float4 rotate(float4 v, float2 z_z){
  v = RGBtoHSV(v);
  v.x += _COLOR_DHUE;
  return HSVtoRGB(v);
}


__device__ float4 rotate_hsls(float4 v, float2 z_z){
  v = RGBtoHSLs(v);

  float l = len(z_z);
  l = (4.0 * _COLOR_LEN_SC + 1.0) * l /(l + 4.0 * _COLOR_LEN_SC);

  float th =  2.0 * 3.14159 * (_COLOR_DHUE + l + count / 15);

  float vx_old = v.x;
  v.x = v.x * cos(th) - v.y * sin(th);
  v.y = vx_old * sin(th) + v.y * cos(th);
  return HSLstoRGB(v);
}
