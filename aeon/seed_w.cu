// EPIMORPH library file
// seed shape functions for the seed_wca seed function

__device__ float trans_w(float w){
  // EXCLUDE
  if(w < _SEED_W_THRESH && w > 0.0f)
    w = 1.0f;
  if(_SEED_W_BASE != 0.0f && w < 0.0f)
    w = _SEED_W_BASE / 20.0;
  return w;
}


__device__ float2 fade(float2 z){
  // linear l-r gradient
  // FULL, LIVE

  z = grid_reduce(z);
  float w = (z.x + 1.0f) / 2.0f;
  return vec2(trans_w(w), 1.0f);
}


__device__ float2 wave(float2 z){
  // sinousoid
  // FULL, LIVE

  z = grid_reduce(z);
  float w = (2.0f + sinf(2.0f * PI * (z.y + _clock * _GLOBAL_SPEED / 10.0f))) / 4.0f;
  return vec2(trans_w(w), 1.0f);
}


__device__ float2 circle(float2 z){
  // circle
  // FULL, LIVE

  z = grid_reduce(z);
  float r = len(z);
  float w = nextafterf(0.0f, -1.0f);
  if(r > _SEED_CIRCLE_R - _SEED_W / 2.0f && r  < _SEED_CIRCLE_R + _SEED_W / 2.0f)
    w = (1.0f - 2.0f * fabsf(r - _SEED_CIRCLE_R) / _SEED_W);
  return vec2(trans_w(w), 1.0f);
}


__device__ float2 lines_lr(float2 z){
  // parallel vertical lines
  // FULL, LIVE

  z = grid_reduce(z);
  float w = nextafterf(0.0f, -1.0f);
  if(z.x > (1.0f - _SEED_W))
    w = (z.x - (1.0f - _SEED_W)) / _SEED_W;
  else if(z.x < -1.0f * (1.0f - _SEED_W))
    w = (-1.0f * (1.0f - _SEED_W) - z.x) / _SEED_W;
  return vec2(trans_w(w), 1.0f);
}


__device__ float2 lines_box(float2 z){
  // 4 lines in a box
  // FULL, LIVE

  z = grid_reduce(z);
  float w = nextafterf(0.0f, -1.0f);
  if(z.x > (1.0f - _SEED_W))
    w =  (z.y < 0.0f ? fmaxf((z.x - (1.0f - _SEED_W)), (-1.0f * (1.0f - _SEED_W) - z.y)) : max((z.x - (1.0f - _SEED_W)), (z.y - (1.0f - _SEED_W)))) / _SEED_W;
  else if(z.y > (1.0f - _SEED_W))
    w =  (z.x > 0.0f ? (z.y - (1.0f - _SEED_W)) : fmaxf((z.y - (1.0f - _SEED_W)), -1.0f * (1.0f - _SEED_W) - z.x)) / _SEED_W;
  else if(z.x < -1.0f * (1.0f - _SEED_W))
    w =  (z.y > 0.0f ? (-1.0f * (1.0f - _SEED_W) - z.x) : fmaxf((-1.0f * (1.0f - _SEED_W) - z.y), -1.0f * (1.0f - _SEED_W) - z.x)) / _SEED_W;
  else if(z.y < -1.0f * (1.0f - _SEED_W))
    w =  (z.x < 0.0f ? (-1.0f * (1.0f - _SEED_W) - z.y) : fmaxf((-1.0f * (1.0f - _SEED_W) - z.y), (z.x - (1.0f - _SEED_W)))) / _SEED_W;
  return vec2(trans_w(w), 1.0f);
}


/*
device__ float hex_lattice(float2 z){
  // hex lattice
  // 5 == 5


  z.y += 0.5f;
  int n_y = floorf(z.y);
  z.y -= n_y;
  if(abs(n_y) % 2 == 1)
    z.y = 1.0f - z.y;

  //z.y = 2.0 * z.y - 1.0;

  z.x *= sqrt(3.0f) / 2.0f;
  z.x += 0.5f;
  int n_x = floorf(z.x);
  z.x -= n_x;

  //if(abs(n_x) % 2 == 1)
  //  z.x = 1.0f - z.x;

  z.x = 2.0f * (z.x) / sqrtf(3.0f);


  if(z.y < z.x * sqrt(3.0f)){
    z.x -= 2.0 / sqrt(3.0f);
  }else if(z.y > -1.0 * z.x * sqrt(3.0f)){
    float tmp = z.x;
    z.x = -0.5 * (z.x + sqrt(3.0f) * z.y);
    z.y = -0.5 * (sqrt(3.0f) * tmp - 1.0 * z.y);
  }

  float w = nextafterf(0.0f, -1.0f);

  z.x += 1.0 / sqrt(3.0f);

  if(2.0 * z.y  < _SEED_W)
    w = 1.0 - 2.0 * z.y / _SEED_W;
  else if(fabsf(-1.0 * z.x * sqrt(3.0f) + 1.0 - z.y) < _SEED_W)
    w = 1.0 - fabsf(-1.0 * z.x * sqrt(3.0f) + 1.0 - z.y) / _SEED_W;
  else if(fabsf(z.y - (z.x * sqrt(3.0f) + 1.0)) < _SEED_W)
    w = 1.0 - fabsf(z.y - (z.x * sqrt(3.0f) + 1.0)) / _SEED_W;


  return w;//vec2(trans_w(w), 1.0f);
}
*/


__device__ float2 lines_box_stag(float2 z){
  // 4 lines in a box, staggered
  // FULL, LIVE

  z = grid_reduce(z);
  float w = nextafterf(0.0f, -1.0f);
  if(z.x > (1.0f - _SEED_W))
    w = (z.x - (1.0f - _SEED_W)) / _SEED_W;
  if(z.y > (1.0f - _SEED_W))
    w = (z.y - (1.0f - _SEED_W)) / _SEED_W;
  if(z.x < -1.0f * (1.0f - _SEED_W))
    w = (-1.0f * (1.0f - _SEED_W) - z.x) / _SEED_W;
  if(z.y < -1.0f * (1.0f - _SEED_W) && z.x < (1.0f - _SEED_W))
    w = (-1.0f * (1.0f - _SEED_W) - z.y) / _SEED_W;
  return vec2(trans_w(w), 1.0f);
}


__device__ float2 lines_inner(float2 z){
  // lines in a cross
  // FULL, LIVE

  z = grid_reduce(z);
  float w = nextafterf(0.0f, -1.0f);
  if(fabsf(z.x) < _SEED_W)
    w = (1.0f - fabsf(z.x) / _SEED_W);
  if(fabsf(z.y) < _SEED_W)
    w = fmaxf(1.0f - fabsf(z.x) / _SEED_W, 1.0f - fabsf(z.y) / _SEED_W);
  return vec2(trans_w(w), 1.0f);
}


__device__ float2 anti_grid_fade(float2 z){
  // inverse grid, radially shaded
  // FULL, LIVE

  z = grid_reduce(z);
  float w = nextafterf(0.0f, -1.0f);
  z = rem(floorf(5.0f * _SEED_GRID_N) / 2.0f * z, 1.0f);
  if((z.x > 0.5f * (1.0f - _SEED_W) && z.x < 0.5f * (1.0f + _SEED_W)) && (z.y < 0.5f * (1.0f + _SEED_W) && z.y > 0.5f * (1.0f - _SEED_W)))
    w = min((1.0f - 2.0f * fabsf(z.y - 0.5f) / _SEED_W), (1.0f - 2.0f * fabsf(z.x - 0.5f) / _SEED_W));
  return vec2(trans_w(w), 1.0f);
}


__device__ float2 grid_fade(float2 z){
  // grid, radially shaded
  // FULL, LIVE

  z = grid_reduce(z);
  float w = nextafterf(0.0f, -1.0f);
  z = rem(floorf(5.0f * _SEED_GRID_N) /2.0f * z, 1.0f);
  if((z.x > 0.5f * (1.0f - _SEED_W) && z.x < 0.5f * (1.0f + _SEED_W)))
    w = (1.0f - 2.0f * fabsf(z.x - 0.5f) / _SEED_W);
  if((z.y < 0.5f * (1.0f + _SEED_W) && z.y > 0.5f * (1.0f - _SEED_W)))
    w = fmaxf((1.0f - 2.0f * fabsf(z.x - 0.5f) / _SEED_W), (1.0f - 2.0f * fabsf(z.y - 0.5f) / _SEED_W));
  return vec2(trans_w(w), 1.0f);
}
