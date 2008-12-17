// EPIMORPH library file
// seed color functions for the seed_wca seed

__device__ float4 simple_color(float2 z, float w){
  // simple coloring function
  float a = 0.0f;
  if(z.y != 0.0f || z.x != 0.0f){
    a = atan2f(z.y, z.x);
    if(a < 0.0f)
      a += 2.0f * 3.14159f;
    a *= floorf(8.0f * _COLOR_TH_EFF) / (2.0f * 3.14159f);
  }

  return HSVtoRGB(vec4(count * _COLOR_SPEED_TH * _GLOBAL_SPEED / 10.0f + a, _COLOR_S, w * _COLOR_V, 0.0f));
}
