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

  __device__ float2 i(float2 v0){
    float r = dot(v0, v0);
    return vec2(v0.x / r, -1.0 * v0.y / r);
  }

  __device__ float2 d(float2 v0){
    return X(v0, v0);
  }

  __device__ float2 s(float2 v0){
    return vec2(sinf(v0.x) * coshf(v0.y), cosf(v0.x) * sinhf(v0.y));
  }

  __device__ float2 c(float2 v0){
    return vec2(cosf(v0.x) * coshf(v0.y), -1.0 * sinf(v0.x) * sinhf(v0.y));
  }

  __device__ float2 t(float2 v0){
    float r = (4.0 * cosf(v0.x) * cosf(v0.x)  - 2.0) * expf(2.0 * v0.y) + expf(4.0 * v0.y) + 1.0;
    return vec2((2.0 * sinf(2.0 * v0.x) * expf(2.0 * v0.y)) / r, (expf(4.0 * v0.y) - 1.0) / r);
  }

  __device__ float2 S(float2 v0){
    return vec2(sinhf(v0.x) * cosf(v0.y), coshf(v0.x) * sinf(v0.y));
  }

  __device__ float2 C(float2 v0){
    return vec2(coshf(v0.x) * cosf(v0.y), sinhf(v0.x) * sinf(v0.y));
  }

  __device__ float2 T(float2 v0){
    float k = 1.0 / expf(2.0 * v0.x);
    float s = sinf(2.0 * v0.y);
    float r = 2.0 * cosf(2.0 * v0.y) * k + k * k + 1.0;
    return vec2((1.0 - k * k) / r, (2.0 * k * s) / r);
  }

  __device__ float2 e(float2 v0){
    float f = expf(v0.x);
    return vec2(f * cosf(v0.y), f * sinf(v0.y));
  }

  __device__ float2 R(float2 v0){
    return vec2(rint(v0.x), rint(v0.y));
  }

  __device__ float2 G(float2 v0){
    return vec2((v0.x > 0 ? floorf(v0.x) : -1.0 * floorf(-1.0 * v0.x)), (v0.y > 0 ? floorf(v0.y) : -1.0 * floorf(-1.0 * v0.y)));
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
    float2 v1 = vec2(par[5], par[6]);
    return vec2(1 - v1.x * v0.x * v0.x + v0.y, v1.y * v0.x);
  }

  __device__ float2 B(float2 v0){
    float K = par[5];
    float pi = 3.14159265;
    float mid = v0.x + K * sinf( pi * (v0.y + 1) ) / pi - 1;
    return vec2(mid, v0.y + mid);
  }

}
