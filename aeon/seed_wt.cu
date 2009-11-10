// EPIMORPH library file
// seed width transformation functions fot the seed_wca seed


__device__ float wt_id(float w){
  // identity transform
  // FULL, LIVE

  return w;
}


__device__ float wt_inv(float w){
  // identity transform
  // FULL, LIVE

  return 1.0f - w;
}


__device__ float wt_circular(float w){
  // circular transform
  // FULL, LIVE

  return sqrtf(1.0f - (1.0f - w) * (1.0f - w));
}


__device__ float wt_inv_circular(float w){
  // circular transform
  // FULL, LIVE

  return 1.0f - sqrtf(1.0f - (1.0f - w) * (1.0f - w));
}
