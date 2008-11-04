__device__ float2 grid_reduce(float2 z){    
  return vec2(rem(z.x + 1.0f, 2.0f) - 1.0f, rem(z.y + 1.0f, 2.0f) - 1.0f);
}


__device__ float2 torus_reduce(float2 z){   
  z.x += 1.0f; z.y += 1.0f;
  float2 tmp = vec2(rem(floor(z.x / 2.0f), 2.0f), rem(floor(z.y / 2.0f), 2.0f));
  float2 res = vec2(rem(z.x, 2.0f), rem(z.y, 2.0f));
  if(tmp.x >= 0.5)
    res.x = 2.0 - res.x;
  if(tmp.y >= 0.5)
    res.y = 2.0 - res.y;
  return vec2(res.x - 1.0f, res.y - 1.0f);
}
