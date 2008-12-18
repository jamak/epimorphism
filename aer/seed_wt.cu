// EPIMORPH library file
// seed width transformation functions fot the seed_wca seed


__device__ float wt_id(float w){
  // identity transform
  return w;
}

__device__ float wt_inv(float w){
  // identity transform
  return 1.0f - w;
}


__device__ float wt_circular(float w){
  // circular transform
  return sqrtf(1.0f - (1.0f - w) * (1.0f - w));
}

__device__ float wt_inv_circular(float w){
  // circular transform
  return 1.0f - sqrtf(1.0f - (1.0f - w) * (1.0f - w));
}
