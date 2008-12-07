
__device__ float4 fade_frame(float2 z)
{
  float d = par[0];
    
  if(z.x < (-1.0f + d) || z.x > (1.0f - d) || z.y < (-1.0f + d) || z.y > (1.0f - d)){
    float v;
    if(z.x > -1.0f * z.y)
      v = max(z.x, z.y);
    else
      v = min(z.x, z.y);
    
    return vec4(1.0001 - (abs(v) - (1.0f - d)) / d, (abs(v) - (1.0f - d)) / d, 0.0f, 1.0);
  }else{
    return vec4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

__device__ float4 cross(float2 z)
{
  float d = par[0];
   
  float w = -1;

  if(abs(z.x) < d)
    w = (1.0 - abs(z.x) / d);
  if(abs(z.y) < d)
    w = max(1.0 - abs(z.x) / d, 1.0 - abs(z.y) / d);
  if(w == -1)
    return vec4(0.0,0.0,0.0,0.0);
  else
    return vec4(w, 0.0, 0.0, 1.0);
}


__device__ float4 fade_frame2(float2 z)
{
  float d = par[0];
    
  if(z.x < (-1.0f + d) || z.x > (1.0f - d) || z.y < (-1.0f + d) || z.y > (1.0f - d)){

    return vec4(0.0, 1.0, 0.0f, 1.0);
  }else{
    return vec4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

__device__ float4 grad_2d(float2 z)
{
  return vec4(0.5f + 0.5f * z.x, 0.5f + 0.5f * z.y, 0.0, 0.5f + 0.5f * z.x);
}

