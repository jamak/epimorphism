__device__ float2 sq(float2 v0){
  return v0 * v0;
}

__device__ float2 sin(float2 v0){
  float s, c;
  sincosf(v0.x, &s, &c);
  return vec2(s * coshf(v0.y), c * sinhf(v0.y));
}

__device__ float2 cos(float2 v0){
  float s, c;
  sincosf(v0.x, &s, &c);
  return vec2(c * coshf(v0.y), -1.0f * s * sinhf(v0.y));
}

__device__ float2 tan(float2 v0){
  float s, c;
  sincosf(2.0f * v0.x, &s, &c);
  float r = c + coshf(2.0f * v0.y);
  return vec2(s, sinhf(2.0f * v0.y)) / r;
}

__device__ float2 sinh(float2 v0){
  float s, c;
  sincosf(v0.y, &s, &c);
  return vec2(sinhf(v0.x) * c, coshf(v0.x) * s);
}

__device__ float2 cosh(float2 v0){
  float s, c;
  sincosf(v0.y, &s, &c);
  return vec2(coshf(v0.x) * c, sinhf(v0.x) * s);
}

__device__ float2 tanh(float2 v0){
  float s, c;
  sincosf(2.0f * v0.y, &s, &c);
  float r = coshf(2.0f * v0.x) + c;
  return vec2(sinhf(2.0f * v0.x), s) / r;
}

__device__ float2 exp(float2 v0){
  float f = expf(v0.x);
  float s, c;
  sincosf(v0.y, &s, &c);
  return vec2(f * c, f * s);
}

__device__ float2 sqrt(float2 v0){
  return vec2(rint(v0.x), rint(v0.y));
}

__device__ float2 G(float2 v0){
  return vec2((v0.x > 0 ? floorf(v0.x) : -1.0f * floorf(-1.0f * v0.x)), (v0.y > 0 ? floorf(v0.y) : -1.0f * floorf(-1.0f * v0.y)));
  // return vec2(floorf(v0.x), floorf(v0.y));
}

__device__ float2 F(float2 v0){
  return v0 - G(v0);
}

__device__ float2 P(float2 v0, float2 v1){
  return vec2(v0.x * v1.x, v0.y * v1.y);
}

//__device__ float2 n(float2 v0){
//  return noise2(v0.x + v0.y);
//}

__device__ float2 H(float2 v0){
  float2 v1 = vec2(par[32], par[33]);
  return vec2(1.0f - v1.x * v0.x * v0.x + v0.y, v1.y * v0.x);
}

__device__ float2 B(float2 v0){
  float K = par[32];
  float pi = PI;
  float mid = v0.x + K * sinf( pi * (v0.y + 1) ) / pi - 1;
  return vec2(mid, v0.y + mid);
}

__device__ float3 rotate3D(float3 v, float3 axis, float th){
  // compute constants
  float c = cosf(th);
  float s = sinf(th);

  // compute rotation
  float3 res = vec3(0.0f, 0.0f, 0.0f);
  res.x = (1.0f + (1.0f - c) * (axis.x * axis.x - 1.0f)) * v.x +
          (axis.z * s + (1.0f - c) * axis.x * axis.y) * v.y +
          (-1.0f * axis.y * s + (1.0f - c) * axis.x * axis.z) * v.z;

  res.y = (-1.0f * axis.z * s + (1.0f - c) * axis.x * axis.y) * v.x +
          (1.0f + (1.0f - c) * (axis.y * axis.y - 1.0f)) * v.y +
          (axis.x * s + (1.0f - c) * axis.y * axis.z) * v.z;

  res.z = (axis.y * s + (1.0f - c) * axis.x * axis.z) * v.x +
          (-1.0f * axis.x * s + (1.0f - c) * axis.y * axis.z) * v.y +
          (1.0f + (1.0f - c) * (axis.z * axis.z - 1.0f)) * v.z;

  return res;
}
