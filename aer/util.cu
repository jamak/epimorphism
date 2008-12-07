__device__ float4 vec4(float x, float y, float z, float w){
  return make_float4(x, y, z, w);
}

__device__ float2 vec2(float x, float y){
  return make_float2(x, y);
}

__device__ float dot(float2 z1, float2 z2){
  return z1.x * z2.x + z1.y * z2.y;
}

__device__ float4 operator*(const float m, const float4 z1){
  return vec4(m * z1.x, m * z1.y, m * z1.z, m * z1.w);
}

__device__ float4 operator+(const float4 z1, const float4 z2){
  return vec4(z1.x + z2.x, z1.y + z2.y, z1.z + z2.z, z1.w + z2.w);
}

__device__ float2 operator*(const float m, const float2 z1){
  return vec2(m * z1.x, m * z1.y);
}

__device__ float2 operator*(const float2 z1, const float m){
  return vec2(m * z1.x, m * z1.y);
}

__device__ float2 operator+(const float2 z1, const float2 z2){
  return vec2(z1.x + z2.x, z1.y + z2.y);
}

__device__ float2 operator-(const float2 z1, const float2 z2){
  return vec2(z1.x - z2.x, z1.y - z2.y);
}

__device__ float2 operator*(const float2 z1, const float2 z2){
  return vec2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
}

__device__ float2 operator/(const float2 z1, const float2 z2){
  float r = dot(z2, z2);
  return vec2((z1.x * z2.x + z1.y * z2.y) / r, (z1.y * z2.x - z1.x * z2.y) / r);
}

__device__ float2 operator/(const float m, const float2 z2){
  float r = dot(z2, z2);
  return vec2((m * z2.x) / r, (-1.0 * m * z2.y) / r);
}

__device__ float mag(const float2 z1){
  return sqrt(z1.x * z1.x + z1.y * z1.y);
}

__device__ float4 tex2D(texture<float4, 2, cudaReadModeElementType> tex, float2 z){
  return tex2D(tex, z.x, z.y);
}

__device__ float4 _gamma3(float4 v, float gamma){
  return vec4(pow(v.x, gamma), pow(v.y, gamma), pow(v.z, gamma), v.w);
}

__device__ float rem(float a, float b){
  float tmp = a / b;
  return b * (tmp - floorf(tmp));
}

__device__ float2 rem(float2 z, float b){
  return vec2(rem(z.x, b), rem(z.y, b));
}



__device__ float2 recover(float2 z){
  if(isnan(z.x))
    z.x = 0;
  if(isnan(z.y))
    z.y = 0; 
  return z;
}


