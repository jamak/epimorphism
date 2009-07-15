__device__ float4 RGBtoHSV(float4 val){
  float vmin = fminf(fminf(val.x, val.y), val.z);
  float vmax = fmaxf(fmaxf(val.x, val.y), val.z);
  float h, s;

  float delta = vmax - vmin;

  if(vmax < 0.001f || delta < 0.001f){
    return vec4(0.0f, 0.0f, vmax, val.w);
  }else {
    s = delta / vmax;
    if(fabsf(val.x - vmax) < 0.0001f)
      h = ( val.y - val.z ) / delta;
    else if(fabsf(val.y - vmax) < 0.0001f)
      h = 2.0f + ( val.z - val.x ) / delta;
    else
      h = 4.0f + ( val.x - val.y ) / delta;
    h /= 6.0f;
    return vec4(h, s, vmax, val.w);
  }
}

__device__ float4 HSVtoRGB(float4 val){
  if(val.y == 0.0f && val.x == 0.0f){//val.y < 0.0001f || val.z < 0.0001f){
    return vec4(val.z, val.z, val.z, val.w);
  }else{
    float4 res = vec4(0.0f, 0.0f, 0.0f, val.w);
    val.x = 6.0f * (val.x - floorf(val.x));
    float f = val.x - floorf(val.x);
    int h = floorf(val.x);
    float4 vals = vec4(1.0f, 1.0f - val.y, 1.0f - val.y * f, 1.0f - val.y * (1.0f - f));
    if(h == 0)
      res = vec4(vals.x, vals.w, vals.y, 0.0f);
    else if(h == 1)
      res = vec4(vals.z, vals.x, vals.y, 0.0f);
    else if(h == 2)
      res = vec4(vals.y, vals.x, vals.w, 0.0f);
    else if(h == 3)
      res = vec4(vals.y, vals.z, vals.x, 0.0f);
    else if(h == 4)
      res = vec4(vals.w, vals.y, vals.x, 0.0f);
    else
      res = vec4(vals.x, vals.y, vals.z, 0.0f);
    res = val.z * res;
    res.w = val.w;
    return res;
  }
}

__device__ float4 HSLstoRGB(float4 val){

  float s = len(vec2(val.x, val.y));
  float h;

  if(s < 0.0001f){
    h = 0.0f;
  }else{
    h = atan2f(val.y, val.x);
  }

  if(h <= 0.0f)
    h += 2.0f * 3.14159f;
  h /= (2.0f * 3.14159f);

  float l = val.z;

  if(s < 0.0001f)
    return vec4((l + 1.0f) / 2.0f, (l + 1.0f) / 2.0f, (l + 1.0f) / 2.0f, val.w);

  float delta = s / sqrtf(1.0f - l * l);

  if(l > 0)
    delta *= (2.0f - l - 1.0f);
  else
    delta *= (l + 1.0f);

  float v = (l + 1.0f + delta) / 2.0f;
  float min = v - delta;
  s = 1.0f - min / v;

  return HSVtoRGB(vec4(h, s, v, val.w));
}

__device__ float4 RGBtoHSLs(float4 val){
  float h, s, l;
  float vmin = fminf(fminf(val.x, val.y), val.z);
  float vmax = fmaxf(fmaxf(val.x, val.y), val.z);

  float delta = vmax - vmin;

  l = (vmax + vmin) - 1.0f;

  s = delta * sqrtf(1.0f - l * l);

  if(l < -0.9999f || l > 0.9999f)
    s = 0.0f;
  else if(l > 0)
    s /= (2.0f - l - 1.0f);
  else if(l <= 0)
    s /= (l + 1.0f);

  if(s < 0.0001f){
    h = 0.0f;
  }else {
    if(fabsf(val.x - vmax) < 0.0001f)
      h = ( val.y - val.z ) / delta;            // between yellow & magenta
    else if(fabsf(val.y - vmax) < 0.0001f)
      h = 2.0f + ( val.z - val.x ) / delta;     // between cyan & yellow
    else
      h = 4.0f + ( val.x - val.y ) / delta;
    h *= PI / 3.0f;
  }
  return vec4(s * cosf(h), s * sinf(h), l, val.w);

}
