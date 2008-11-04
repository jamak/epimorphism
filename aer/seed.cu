extern "C" {
  __device__ float4 fade_frame(float2 z, float* par)
  {
    float d = par[0];
    
    if(z.x < (-1.0f + d) || z.x > (1.0f - d) || z.y < (-1.0f + d) || z.y > (1.0f - d)){
      float v;
      if(z.x > -1.0f * z.y)
	v = max(z.x, z.y);
      else
	v = min(z.x, z.y);
    
      return vec4(1.000001 - (abs(v) - (1.0f - d)) / d, (abs(v) - (1.0f - d)) / d, 0.0f, 1.0);
    }else{
      return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  __device__ float4 grad_2d(float2 z, float* par)
  {
    return vec4(0.5f + 0.5f * z.x, 0.5f + 0.5f * z.y, 0.0, 0.5f);
  }

}
