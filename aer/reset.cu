__device__ float4 reset_black(int x, int y){
  return vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ float4 reset_hsls(int x, int y){
  float phi = 2.0f * PI * _COLOR_PHI;
  float psi = 2.0f * PI * _COLOR_PSI;

  float4 pt = 2.0f * (_HSLS_RESET_Z - 0.5f) * vec4(cosf(psi) * cosf(phi), cosf(psi) * sinf(phi), sinf(psi), 0.0);

  return HSLstoRGB(pt);
}
