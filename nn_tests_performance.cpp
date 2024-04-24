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

    std::vector<unsigned> sizes_A = {16, 64, 256, 1024, 2048};
    std::vector<unsigned> sizes_B = {16, 64, 256, 1024, 2048};

    std::vector<float> A(sizes_A.back()*sizes_A.back(), 0);
    std::vector<float> B(sizes_A.back()*sizes_B.back(), 0);
    std::vector<float> C(sizes_B.back()*sizes_B.back(), 0);
    std::vector<float> C_ref(sizes_B.back()*sizes_B.back(), 0);
    TensorCompiler tc;

    for (int m_mode = 0; m_mode <= 1; m_mode++)
    {
      nn::TensorProcessor::RuntimeSettings settings;
      settings.use_coop_mat_mul = m_mode;

      TensorProcessor::set_runtime_settings(settings);
      printf("\n Use cooperative matrices: %d\n", m_mode);
      for (unsigned size_A : sizes_A)
      {
        for (unsigned size_B : sizes_B)
        {
          unsigned n_tries = size_A == 2048 ? 25 : (size_A == 1024 ? 50 : 500);

          for (int i=0;i<size_A*size_A;i++)
            A[i] = 2.0*rand()/RAND_MAX + 1.0;

          for (int i=0;i<size_A*size_B;i++)
            B[i] = 2.0*rand()/RAND_MAX + 1.0;
          
          for (int i=0;i<size_B;i++)
          {
            for (int j=0;j<size_A;j++)
            {
              long double val = 0;
              for (int k=0;k<size_A;k++)
                val += (long double)A[j*size_A + k] * (long double)B[i*size_A + k];
              C_ref[j*size_B + i] = val;
            }
          }

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

          long double sum_diff = 0;
          double max_diff = 0;
          unsigned max_diff_i=0, max_diff_j=0;

          for (int i=0;i<size_A;i++)
          {
            for (int j=0;j<size_B;j++)
            {
              //if (i<1 && j<256)
              //  printf("(%u %u) %.8f %.8f\n",i,j, C_ref[i*size_B + j], C[i*size_B + j]);
              double diff = abs(C_ref[i*size_B + j] - C[i*size_B + j]);
              sum_diff += diff;
              if (diff > max_diff)
              {
                max_diff = diff;
                max_diff_i = i;
                max_diff_j = j;
              }
            }
          }

          
          float t_compile = 0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
          float t_input =   0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_3 - t_2).count();
          float t_execute = 0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_4 - t_3).count();
          float t_output =  0.001/n_tries*std::chrono::duration_cast<std::chrono::microseconds>(t_5 - t_4).count();

          printf("matmul (%4ux%4u)*(%4ux%4u): exec %7.2f ms, overhead %5.2f+%5.2f+%5.2f ms, av. diff %.2Le, max diff %.2le at (%u %u)\n", 
                size_A, size_A, size_B, size_A, t_execute, t_compile, t_input, t_output,
                sum_diff/(size_A*size_B), max_diff, max_diff_i, max_diff_j);
        }      
      }
    }
  }

  void perf_test_2_MLP()
  {
    Dataset dataset;
    read_MNIST_dataset(base_path + std::string("resources/MNIST-dataset"), &dataset);

    NeuralNetwork nn2;
    nn2.add_layer(std::make_shared<FlattenLayer>(28,28,1));
    nn2.add_layer(std::make_shared<DenseLayer>(28*28*1, 256), Initializer::He);
    nn2.add_layer(std::make_shared<ReLULayer>());
    nn2.add_layer(std::make_shared<DenseLayer>(256, 256), Initializer::He);
    nn2.add_layer(std::make_shared<ReLULayer>());
    nn2.add_layer(std::make_shared<DenseLayer>(256, 10), Initializer::He);
    nn2.add_layer(std::make_shared<SoftMaxLayer>());

auto t_1 = std::chrono::steady_clock::now();
    nn2.train(dataset.train_data.data(), dataset.train_labels.data(), dataset.train_elements,
              128, 100, false, OptimizerAdam(0.0001f), Loss::CrossEntropy);
auto t_2 = std::chrono::steady_clock::now();

    std::vector<float> y_res(dataset.train_labels.size(),0);

auto t_3 = std::chrono::steady_clock::now();
    nn2.evaluate(dataset.train_data, y_res);
auto t_4 = std::chrono::steady_clock::now();

    float acc = 0.0f;
    float cnt = 0.0f;
    for (int i=0;i<dataset.train_labels.size();i+=10)
    {
      int max_pos = 0;
      int ref_max_pos = 0;
      for (int j=0;j<10;j++)
      {
        if (y_res[i+j] > y_res[i+max_pos])
          max_pos = j;
        if (dataset.train_labels[i+j] > dataset.train_labels[i+ref_max_pos])
          ref_max_pos = j;
      }

      acc += (max_pos == ref_max_pos);
      cnt++;
    }
    float error_rate = 1 - acc/cnt;

    float t_train =    0.001*std::chrono::duration_cast<std::chrono::microseconds>(t_2 - t_1).count();
    float t_evaluate = 0.001*std::chrono::duration_cast<std::chrono::microseconds>(t_4 - t_3).count();

    printf("error rate %f\n", error_rate);
    printf("t_train = %f ms\n", t_train);
    printf("t_eval  = %f ms\n", t_evaluate);
  }

  void perform_tests_performance(const std::vector<int> &test_ids)
  {
    srand(time(NULL));
    std::vector<int> tests = test_ids;

    std::vector<std::function<void(void)>> test_functions = {
      perf_test_1_matmul, perf_test_2_MLP,
    };

    if (tests.empty())
    {
      tests.resize(test_functions.size());
      for (int i=0;i<test_functions.size();i++)
        tests[i] = i+1;
    }
    
    TensorProcessor::init(TensorProcessor::Backend::GPU);

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