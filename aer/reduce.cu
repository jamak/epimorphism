// EPIMORPH library file
// complex plane reductions C -> [-1, -1] x [1, 1]


__device__ float2 grid_reduce(float2 z){
  // standard reduction based on the cartesian grid
  return rem(z + vec2(1.0f, 1.0f), 2.0f) - vec2(1.0f, 1.0f);
}


__device__ float2 torus_reduce(float2 z){
  // reduction based on the reflective torus
  z = z + vec2(1.0f, 1.0f);

  z = rem(z, 4.0f);
  if(z.x >= 2.0f)
    z.x = 4.0 - z.x;
  if(z.y >= 2.0f)
    z.y = 4.0 - z.y;

  return z - vec2(1.0f, 1.0f);
}


__device__ float2 hex_reduce(float2 z){
  float2 res = vec2(0.0f, 0.0f);
  z.y += 0.5f;
  int n_y = floorf(z.y);
  if(n_y % 2 == 0)
    z.y -= n_y;
  else
    z.y = 1.0 - (z.y - n_y);

  return res;
}
