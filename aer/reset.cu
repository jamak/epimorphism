__device__ float4 reset_black(int x, int y){
  return vec4(0.0, 0.0, 0.0, 0.0);
}

__device__ float4 reset_hsls(int x, int y){
  return vec4(0.5, 0.5, 0.5, 0.0);
}
