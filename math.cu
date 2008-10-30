extern "C" {

  __device__ float2 X(float2 v0, float2 v1){	
    return v0 * v1;
  }

  __device__ float2 D(float2 v0, float2 v1){	
    float r = dot(v1, v1);
    return vec2((v0.x * v1.x + v0.y * v1.y) / r, (v0.y * v1.x - v0.x * v1.y) / r);
  }

  __device__ float2 A(float2 v0, float2 v1){
    return v0 + v1;
  }

  __device__ float2 i(float2 v1){
    float r = dot(v1, v1);
    return vec2(v1.x / r, -1.0 * v1.y / r);
  }

  __device__ float2 d(float2 v1){
    return X(v1, v1);
  }

  __device__ float2 s(float2 v1){
    return vec2(sinf(v1.x) * coshf(v1.y), cosf(v1.x) * sinhf(v1.y));
  }

  __device__ float2 c(float2 v1){
    return vec2(cosf(v1.x) * coshf(v1.y), -1.0 * sinf(v1.x) * sinhf(v1.y));
  }

  __device__ float2 t(float2 v1){
    float r = (4.0 * cosf(v1.x) * cosf(v1.x)  - 2.0) * expf(2.0 * v1.y) + expf(4.0 * v1.y) + 1.0;
    return vec2((2.0 * sinf(2.0 * v1.x) * expf(2.0 * v1.y)) / r, (expf(4.0 * v1.y) - 1.0) / r);
  }

  __device__ float2 S(float2 v1){
    return vec2(sinhf(v1.x) * cosf(v1.y), coshf(v1.x) * sinf(v1.y));
  }

  __device__ float2 C(float2 v1){
    return vec2(coshf(v1.x) * cosf(v1.y), sinhf(v1.x) * sinf(v1.y));
  }

  __device__ float2 T(float2 v1){
    float k = 1.0 / expf(2.0 * v1.x);
    float s = sinf(2.0 * v1.y);
    float r = 2.0 * cosf(2.0 * v1.y) * k + k * k + 1.0;
    return vec2((1.0 - k * k) / r, (2.0 * k * s) / r);
  }

  __device__ float2 e(float2 v1){
    float f = expf(v1.x);
    return vec2(f * cosf(v1.y), f * sinf(v1.y));
  }

  //__device__ float2 R(float2 v1){
  //  return round(v1);
  //}

  //__device__ float2 G(float2 v1){
  //  return vec2((v1.x > 0 ? floor(v1.x) : -1.0 * floor(-1.0 * v1.x)), (v1.y > 0 ? floor(v1.y) : -1.0 * floor(-1.0 * v1.y)));
  //  return floor(v1);
  // }

  //__device__ float2 F(float2 v1){
  //  return v1 - G(v1);
  //}

  __device__ float2 P(float2 v1, float2 v2){
    return vec2(v1.x * v2.x, v1.y * v2.y);
  }

  //__device__ float2 n(float2 v1){
  //  return noise2(v1.x + v1.y);
  //}

  __device__ float2 H(float2 v1){
    float2 v2 = vec2(1.4, 0.3);
    return vec2(1 - v2.x * v1.x * v1.x + v1.y, v2.y * v1.x);
  }

  __device__ float2 B(float2 v1){
    float K = 0.6;
    float pi = 3.14159265;
    float mid = v1.x + K * sinf( pi * (v1.y + 1) ) / pi - 1;
    return vec2(mid, v1.y + mid);
  }

}
