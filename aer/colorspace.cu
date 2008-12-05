__device__ float4 RGBtoHSV(float4 val){
  float vmin = fmin(fmin(val.x, val.y), val.z);
  float vmax = fmax(fmax(val.x, val.y), val.z);	
  float h, s;
  
  float delta = vmax - vmin;
  
  if(vmax < 0.001 || delta < 0.001){
    return vec4(0.0, 0.0, vmax, val.w);
  }else {	
    s = delta / vmax;
    if(abs(val.x - vmax) < 0.0001)
      h = ( val.y - val.z ) / delta;		
    else if(abs(val.y - vmax) < 0.0001)
      h = 2.0 + ( val.z - val.x ) / delta;	
    else
      h = 4.0 + ( val.x - val.y ) / delta;	
    h /= 6.0;
    return vec4(h, s, vmax, val.w);
  }	
}

__device__ float4 HSVtoRGB(float4 val){
  if(val.y == 0.0 && val.x == 0.0){//val.y < 0.0001 || val.z < 0.0001){
    return vec4(val.z, val.z, val.z, val.w);
  }else{
    float4 res = vec4(0.0, 0.0, 0.0, val.w);
    val.x = 6.0 * (val.x - floorf(val.x));
    int h = floorf(val.x);
    float f = val.x - floorf(val.x);
    float4 vals = vec4(1.0, 1.0 - val.y, 1.0 - val.y * f, 1.0 - val.y * (1.0 - f));
    if(h == 0)
      res = vec4(vals.x, vals.w, vals.y, 0.0);
    else if(h == 1)
      res = vec4(vals.z, vals.x, vals.y, 0.0);
    else if(h == 2)
      res = vec4(vals.y, vals.x, vals.w, 0.0);
    else if(h == 3)
      res = vec4(vals.y, vals.z, vals.x, 0.0);
    else if(h == 4)
      res = vec4(vals.w, vals.y, vals.x, 0.0);
    else 
      res = vec4(vals.x, vals.y, vals.z, 0.0);
    res = val.z ^ res;
    res.w = val.w;
    return res;
  }
}
