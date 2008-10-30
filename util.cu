extern "C" {

  __device__ float4 vec4(float x, float y, float z, float w){
    return make_float4(x, y, z, w);
  }

  __device__ float2 vec2(float x, float y){
    return make_float2(x, y);
  }


  __device__ float4 operator^(const float4 v1, const float4 v2){
    return make_float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
  }

  __device__ float4 operator%(const float m, const float4 v1){
    return make_float4(m * v1.x, m * v1.y, m * v1.z, m * v1.w);
  }

  __device__ float2 operator+(const float2 z1, const float2 z2){
    return make_float2(z1.x + z2.x, z1.y + z2.y);
  }

  __device__ float2 operator*(const float2 z1, const float2 z2){
    return make_float2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
  }


  __device__ float dot(float2 z1, float2 z2){
    return z1.x * z2.x + z1.y * z2.y;
  }

}
