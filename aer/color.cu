// EPIMORPH library file
// coloring functions

__device__ float4 rgb_id(float4 v, float2 z_z){
  // identity
  return v;
}


__device__ float4 rg_swizzle(float4 v, float2 z_z){
  // red/green switch
  return vec4(v.y, v.x, v.z, v.w);
}


__device__ float4 gb_swizzle(float4 v, float2 z_z){
  // green/blue switch
  return vec4(v.x, v.z, v.y, v.w);
}


__device__ float4 rgb_swizzle(float4 v, float2 z_z){
  // rotate red/green/blue
  return vec4(v.y, v.z, v.x, v.w);
}


__device__ float4 rotate_hsv(float4 v, float2 z_z){
  // hsv rotation
  v = RGBtoHSV(v);

  float l = len(z_z);
  l = (4.0f * _COLOR_LEN_SC + 1.0f) * l / (l + 4.0f * _COLOR_LEN_SC);

  float a = 0.0f;
  if(_COLOR_TH_EFF != 0 && (z_z.y != 0.0f || z_z.x != 0.0f)){
    a = atan2f(z_z.y, z_z.x);
    if(a < 0.0f)
      a += 2.0f * 3.14159f;
    a *= floorf(8.0f * _COLOR_TH_EFF) / (2.0f * 3.14159f);
  }

  float th =  2.0f * PI * (_COLOR_DHUE + l + a + count * _COLOR_SPEED_TH * _GLOBAL_SPEED / 10.0f);

  v.x += th;

  return HSVtoRGB(v);
}


__device__ float4 rotate_hsls(float4 v, float2 z_z){
  // complex hsls rotation
  // defaults
  v = RGBtoHSLs(v);

  float l = len(z_z);
  l = (4.0f * _COLOR_LEN_SC + 1.0f) * l / (l + 4.0f * _COLOR_LEN_SC);

  float a = 0.0f;
  if(_COLOR_TH_EFF != 0 && (z_z.y != 0.0f || z_z.x != 0.0f)){
    a = atan2f(z_z.y, z_z.x);
    if(a < 0.0f)
      a += 2.0f * 3.14159f;
    a *= floorf(8.0f * _COLOR_TH_EFF) / (2.0f * 3.14159f);
  }

  float th =  2.0f * PI * (_COLOR_DHUE + a + l + count * _COLOR_SPEED_TH * _GLOBAL_SPEED / 10.0f);
  float phi = 2.0f * PI * _COLOR_PHI;
  float psi = 2.0f * PI * _COLOR_PSI;
  float c = cosf(th);
  float s = sinf(th);

  float3 axis = vec3(cosf(psi) * cosf(phi), cosf(psi) * sinf(phi), sinf(psi));

  float3 tmp = vec3(0.0f, 0.0f, 0.0f);

  tmp.x = (1.0f + (1.0f - c) * (axis.x * axis.x - 1.0f)) * v.x +
          (axis.z * s + (1.0f - c) * axis.x * axis.y) * v.y +
          (-1.0f * axis.y * s + (1.0f - c) * axis.x * axis.z) * v.z;

  tmp.y = (-1.0f * axis.z * s + (1.0f - c) * axis.x * axis.y) * v.x +
          (1.0f + (1.0f - c) * (axis.y * axis.y - 1.0f)) * v.y +
          (axis.x * s + (1.0f - c) * axis.y * axis.z) * v.z;

  tmp.z = (axis.y * s + (1.0f - c) * axis.x * axis.z) * v.x +
          (-1.0f * axis.x * s + (1.0f - c) * axis.y * axis.z) * v.y +
          (1.0f + (1.0f - c) * (axis.z * axis.z - 1.0f)) * v.z;

  v = vec4(0.9999 * tmp.x, 0.9999 * tmp.y, 0.9999 * tmp.z, v.w);

  return HSLstoRGB(v);
}

