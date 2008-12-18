// EPIMORPH library file
// complex plane reductions


__device__ float2 grid_reduce(float2 z){
  // standard reduction based on the cartesian grid
  return rem(z + vec2(1.0f, 1.0f), 2.0f) - vec2(1.0f, 1.0f);
}


__device__ float2 torus_reduce(float2 z){
  // reduction based on the reflective torus
  z = z + vec2(1.0f, 1.0f);

  float2 tmp = rem(z, 2.0f);
  if(tmp.x >= 2.0f)
    z.x = 4.0f - z.x;
  if(tmp.y >= 2.0f)
    z.y = 4.0f - z.y;

  return tmp - vec2(1.0f, 1.0f);
}
