extern "C" {
  __device__ float4 fade_frame(float2 z)
  {
    float d = 0.1;
    
    if(z.x < (-1.0 + d) || z.x > (1.0 - d) || z.y < (-1.0 + d) || z.y > (1.0 - d)){
      float v;
      if(z.x > -1.0 * z.y)
	v = max(z.x, z.y);
      else
	v = min(z.x, z.y);
    
      return vec4((abs(v) - (1.0 - d)) / d, 0.0, 0.0, (abs(v) - (1.0 - d)) / d);
    }else{
      return vec4(0.0, 0.0, 0.0, 0.0);
    }
  }

}
