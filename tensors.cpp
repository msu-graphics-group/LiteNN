#include "tensors.h"

int main(int argc, char **argv)
{
  /*
    float f = 0;
    Tensor<float, 1> vector(nullptr, std::array<uint32_t, 1>{4u});
    vector.fill(1);
    Tensor<float, 2> matrix(nullptr, std::array<uint32_t, 2>{4u,4u});
    matrix.fill(0);
    Tensor<float, 3> cube(nullptr, std::array<uint32_t, 3>{4u,4u,4u});
    cube.fill(3);

    matrix[{0,0}] = 1;
    matrix[{1,1}] = 2;
    matrix[{2,2}] = 3;
    matrix[{3,3}] = 4;

    vector[{0}] = 1;
    vector[{1}] = 10;
    vector[{2}] = 100;
    vector[{3}] = 1000;

    auto cube_2 = cube.dot(vector);
    cube_2.compact_print();

    Tensor<int, 3> large_cube({500u,500u,500u});
    int v = 0;
    for (int i=0;i<large_cube.total_size;i++)
      large_cube.data[i] = 1;
    auto t1 = std::chrono::steady_clock::now();
    auto t2 = std::chrono::steady_clock::now();
    auto t3 = std::chrono::steady_clock::now();

    uint64_t d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    uint64_t d2 = std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();

    //printf("sum %d %d\n", tsum1[21], tsum2[{21}]);
    printf("time %lu %lu\n", d1, d2);
  
  auto M1 = tp::get_matrix<float>({1, 2, 3, 4, 5, 6}, 3, 2);
  auto M2 = tp::get_matrix<float>({1, 2, 3, 4, 5, 6}, 2, 3);
  auto M3 = tp::get_matrix<float>(2, 2);
  tp::compact_print(M1);
  tp::compact_print(M2);
  tp::mat_mul(M1, M2, M3);
  tp::compact_print(M3);

  auto bM1 = tp::get_3_tensor<float>(400, 500, 32);
  auto bM2 = tp::get_matrix<float>(600, 400);
  auto bM3 = tp::get_3_tensor<float>(500, 600, 32);
  for (u_int32_t i = 0; i < bM1.get_size(); i++)
    bM1.get(i) = rand() % 10;
  for (u_int32_t i = 0; i < bM2.get_size(); i++)
    bM2.get(i) = rand() % 10;
  auto t1 = std::chrono::steady_clock::now();
  tp::mat_mul(bM1, bM2, bM3);
  auto t2 = std::chrono::steady_clock::now();
  auto t3 = std::chrono::steady_clock::now();

  uint64_t d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  uint64_t d2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
  printf("time %lu %lu\n", d1, d2);
  */
  auto v1 = tp::get_vertex<float>({1,2,3,4});
  auto v2 = tp::get_vertex<float>({1,10,100,1000});
  auto op = tp::get_matrix<float>(4,4);
  tp::outer_product(v1, v2, op);
  tp::compact_print(op);
}