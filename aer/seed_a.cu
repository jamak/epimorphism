// EPIMORPH library file
// seed alpha functions for the seed_wca seed function


__device__ float solid_alpha(float w){
  // solid
  return _COLOR_A;
}


__device__ float linear_alpha(float w){
  // linear with w
  return w * _COLOR_A;
}


__device__ float circular_alpha(float w){
  // circular with w
  return sqrtf(1.0f - (1.0f - w) * (1.0f - w)) * _COLOR_A;
}
