#include "nn_tests.h"
#include <cstdio>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <vector>
#include <functional>
#include <array>
#include <type_traits>
#include <string>
#include <cstring>
#include <chrono>
#include <cmath>

#include "tensor_processor.h"
#include "tensor_compiler.h"
#include "neural_network.h"
#include "siren.h"
#include "dataset.h"

#include "stb_image.h"
#include "stb_image_write.h"

namespace nn
{
  void perf_test_1_matmul()
  {
    printf("TEST 1. MATRIX MULTIPLICATION BENCHMARK\n");

    std::vector<unsigned> sizes = {1, 4, 16, 64, 256, 1024, 4096};

    std::vector<float> A(sizes.back()*sizes.back(), 0);
    std::vector<float> B(sizes.back()*sizes.back(), 0);
    std::vector<float> C(sizes.back()*sizes.back(), 0);
    TensorCompiler tc;

    for (unsigned size_A : sizes)
    {
      for (unsigned size_B : sizes)
      {
        unsigned n_tries = size_A == 4096 ? 10 : (size_A == 1024 ? 50 : 500);

    auto t_1 = std::chrono::steady_clock::now();
        tc.start_program();
        {
          TensorToken A = TensorToken(size_A, size_A);
          TensorToken B = TensorToken(size_A, size_B);
          TensorToken C = TensorToken::mat_mul_t(A, B);

          tc.input(A, "A");
          tc.input(B, "B");
          tc.output(C, "C");
        }     
        TensorProgram p = tc.finish_program();   
        
    auto t_2 = std::chrono::steady_clock::now();

        TensorProcessor::set_program(p);
        TensorProcessor::set_input("A", A.data(), A.size());
        TensorProcessor::set_input("B", B.data(), B.size());

    auto t_3 = std::chrono::steady_clock::now();

        for (int i=0;i<n_tries;i++)
          TensorProcessor::execute();

    auto t_4 = std::chrono::steady_clock::now();

        TensorProcessor::get_output("C", C.data(), C.size());
    
    auto t_5 = std::chrono::steady_clock::now();
        
        float t_compile = 0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
        float t_input =   0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_3 - t_2).count();
        float t_execute = 0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_4 - t_3).count();
        float t_output =  0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_5 - t_4).count();

        printf("matmul (%4ux%4u)*(%4ux%4u): exec %7.2f ms, overhead %5.2f+%5.2f+%5.2f ms\n", size_A, size_A, size_B, size_A, 
               t_execute, t_compile, t_input, t_output);
      }      
    }
  }

  void perform_tests_performance(const std::vector<int> &test_ids)
  {
    srand(time(NULL));
    std::vector<int> tests = test_ids;

    std::vector<std::function<void(void)>> test_functions = {
      perf_test_1_matmul,
    };

    if (tests.empty())
    {
      tests.resize(test_functions.size());
      for (int i=0;i<test_functions.size();i++)
        tests[i] = i+1;
    }
    
    TensorProcessor::init("GPU");

    for (int i=0;i<80;i++)
      printf("#");
    printf("\nPERFORMANCE TESTS\n");
    for (int i=0;i<80;i++)
      printf("#");
    printf("\n");
    
    for (int i : tests)
    {
      assert(i > 0 && i <= test_functions.size());
      test_functions[i-1]();
    }
  }
}