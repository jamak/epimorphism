extern "C" {

  __device__ float4 vec4(float x, float y, float z, float w){
    return make_float4(x, y, z, w);
  }

  __device__ float2 vec2(float x, float y){
    return make_float2(x, y);
  }

  __device__ float4 operator^(const float m, const float4 v1){
    return vec4(m * v1.x, m * v1.y, m * v1.z, m * v1.w);
  }

  __device__ float2 operator%(const float m, const float2 v1){
    return vec2(m * v1.x, m * v1.y);
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

  __device__ float4 texv2D(texture<float4, 2, cudaReadModeElementType> tex, float2 z){
    return tex2D(tex, z.x, z.y);
  }

  __device__ float4 merge_blend(const float4 v, const float4 v1, const float4 v2, const float a){
    return vec4(v.x + a * v1.x + (1.0f - a) * v2.x, v.y + a * v1.y + (1.0f - a) * v2.y, 
		v.z + a * v1.z + (1.0f - a) * v2.z, v.w + a * v1.w + (1.0f - a) * v2.w);
  }

  __device__ float dot(float2 z1, float2 z2){
    return z1.x * z2.x + z1.y * z2.y;
  }

  __device__ float4 _gamma(float4 v, float gamma){
    return vec4(pow(v.x, gamma), pow(v.y, gamma), pow(v.z, gamma), v.w);
  }

  __device__ float rem(float a, float b){
    float tmp = a / b;
    return b * (tmp - floorf(tmp));
  }

  __device__ float2 recover(float2 z){
    if(isnan(z.x))
      z.x = 0;
    if(isnan(z.y))
      z.y = 0; 
    return z;
  }

}
