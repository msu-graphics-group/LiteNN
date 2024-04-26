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
#include <map>

#include "tensor_processor.h"
#include "tensor_compiler.h"
#include "neural_network.h"
#include "siren.h"
#include "nn_tests.h"
#include "direct/nnd_tests.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void perform_tests_args(char **argv, int argc, int start)
{
  std::vector<std::string> test_groups = {"base", "gpu", "nn", "benchmark"};
  std::map<std::string, std::vector<int>> test_ids;

  for (std::string &g : test_groups)
    test_ids[g] = {-1};

  int idx = start;
  std::string active_group = "";
  bool in_group = false;
  while (idx < argc)
  {
    std::string arg = argv[idx];
    for (std::string &g : test_groups)
    {
      if (arg == g)
      {
        active_group = g;
        break;
      }
    }

    if (arg == "all")
    {
      if (active_group == "") //perform all tests in all groups
      {
        for (std::string &g : test_groups)
          test_ids[g] = {};
      }
      else //perform all tests in this groups
        test_ids[active_group] = {};
    }
    else if (active_group != "")
    {
      char* p;
      int test_num = strtol(arg.c_str(), &p, 10);
      if (p != arg.c_str())
      {
        if (test_ids[active_group][0] == -1)
          test_ids[active_group][0] = test_num;
        else
          test_ids[active_group].push_back(test_num);
      }
      else
        printf("invalid argument %s. It should be \"all\" or number\n", arg.c_str());
    }

    idx++;
  }
  
  if (test_ids["base"].empty() || test_ids["base"][0] != -1)
    nn::perform_tests_tensor_processor(test_ids["base"]);
  
  if (test_ids["gpu"].empty() || test_ids["gpu"][0] != -1)
    nn::perform_tests_tensor_processor_GPU(test_ids["gpu"]);
  
  if (test_ids["nn"].empty() || test_ids["nn"][0] != -1)
    nn::perform_tests_neural_networks(test_ids["nn"]);
  
  if (test_ids["benchmark"].empty() || test_ids["benchmark"][0] != -1)
    nn::perform_tests_performance(test_ids["benchmark"]);
}

int main(int argc, char **argv)
{
  //nnd::perform_tests();
  if (argc == 1)
  {
    nn::perform_all_tests();
  }
  else if (argv[1] == "-h" || argv[1] == "-help" || argv[1] == "--help")
  {
    printf("./nn_test <test_group_1> <test_1> <test_2> ... <test_group_k> <test_1>  ...\n");
    printf("test groups are [base, gpu, nn, benchmark]\n");
    printf("test_i is either test number or \"all\" for all tests\n");
    printf("e.g. ./nn_test base all gpu all   ./nn_test benchmark 1 2\n");
  }
  else
    perform_tests_args(argv, argc, 1);
  return 0;
}