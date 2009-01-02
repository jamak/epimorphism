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


/*
device__ float2 hex_reduce(float2 z){
  // hex reduce
  // 5 == 5

  z = z * 2.0;

  z.y += 0.5f;
  int n_y = floorf(z.y);
  z.y -= n_y;
  if(abs(n_y) % 2 == 1)
    z.y = 1.0f - z.y;


  z.x *= sqrt(3.0f) / 2.0f;
  z.x += 0.5f;
  int n_x = floorf(z.x);
  z.x -= n_x;

  //if(abs(n_x) % 2 == 1)
  //  z.x = 1.0f - z.x;

  z.x = 2.0f * (z.x) / sqrtf(3.0f);


  if(z.y < z.x * sqrt(3.0f)){
    z.x -= 2.0 / sqrt(3.0f);
  }else if(z.y > -1.0 * z.x * sqrt(3.0f)){
    float tmp = z.x;
    z.x = -0.5 * (z.x + sqrt(3.0f) * z.y);
    z.y = -0.5 * (sqrt(3.0f) * tmp - 1.0 * z.y);
  }


  z.x += 1.0 / sqrt(3.0f);

  z.y -= 0.5;

  return z / 2.0;
}
*/

__device__ float2 awesome(float2 z){
  // hex reduce
  // 5 == 5

  z = z * 2.0;

  z.y += 0.5f;
  int n_y = floorf(z.y);
  z.y -= n_y;
  if(abs(n_y) % 2 == 1)
    z.y = 1.0f - z.y;

  z.y -= 0.5;

  return z / 2.0;
}
