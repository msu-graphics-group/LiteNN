#ifndef INCLUDE_LITENN_LOSS_FUNCTIONS_H_
#define INCLUDE_LITENN_LOSS_FUNCTIONS_H_

#include "tensor_compiler.h"
#include <vector>
#include <cstdint>
#include <array>
#include <cmath>
#include <memory>
#include <functional>
#include <tuple>
#include <variant>

namespace nn
{
  class LossFunction
  {
  public:
    virtual ~LossFunction() {};
    virtual void init() {};
    virtual int parameters_count() { return 0; }
    virtual TensorToken forward(const TensorToken &output, const TensorToken &target_output) = 0;
    virtual TensorToken backward(const TensorToken &output, const TensorToken &target_output) = 0;
    virtual std::string get_name() = 0;
  };

  class MSELoss : public LossFunction
  {
    int batch_size;
  public:
    MSELoss(int b_size)
    {
      batch_size = b_size;
    }
    virtual void init() override { };
    virtual int parameters_count() override { return 0; };
    virtual TensorToken forward(const TensorToken &output, const TensorToken &target_output) override
    {
      TensorToken diff = output - target_output;
      return (diff*diff).sum()/(float)(output.total_size());
    }
    virtual TensorToken backward(const TensorToken &output, const TensorToken &target_output) override
    {
      TensorToken diff = output - target_output;
      return 2.0f*diff;
    }
    virtual std::string get_name() override { return "MSELoss"; }
  };

  class CrossEntropyLoss : public LossFunction
  {
    int batch_size;
  public:
    CrossEntropyLoss(int b_size)
    {
      batch_size = b_size;
    }
    virtual void init() override { };
    virtual int parameters_count() override { return 0; };
    virtual TensorToken forward(const TensorToken &output, const TensorToken &target_output) override
    {
      TensorToken mo = -1.0f*target_output;
      return (mo * TensorToken::log(output + 1e-15f)).sum()/(float)(batch_size);
    }
    virtual TensorToken backward(const TensorToken &output, const TensorToken &target_output) override
    {
      TensorToken mo = -1.0f*target_output;
      return mo / (output + 1e-15f);
    }
    virtual std::string get_name() override { return "CrossEntropyLoss"; }
  };

  class NBVHLoss : public LossFunction
  {
    int batch_size;
  public:
    NBVHLoss(int b_size)
    {
      batch_size = b_size;
    }
    virtual void init() override { };
    virtual int parameters_count() override { return 0; };
    virtual TensorToken forward(const TensorToken &output, const TensorToken &target_output) override
    {
      TensorToken t_target_output = target_output.transpose();
      TensorToken t_output = output.transpose();
      // classification (binary cross entropy loss, size = 1)
      TensorToken cl_target = t_target_output.get(0);
      TensorToken cl_output = t_output.get(0);
      TensorToken l = (-1.0f * (cl_target * TensorToken::log(cl_output + 1e-15f) + (1.0f - cl_target) * TensorToken::log(1.0f - cl_output + 1e-15f))).sum()/(float)(batch_size);
      // distance + normal (l1 loss, size = 3 + 3)
      TensorToken t_diff = t_output.get({1, 7}) - t_target_output.get({1, 7});
      t_diff.set(0, cl_target * t_diff.get(0));
      t_diff.set(1, cl_target * t_diff.get(1));
      t_diff.set(2, cl_target * t_diff.get(2));
      t_diff.set(3, cl_target * t_diff.get(3));
      t_diff.set(4, cl_target * t_diff.get(4));
      t_diff.set(5, cl_target * t_diff.get(5));
      l += TensorToken::sqrt(t_diff*t_diff).sum()/(float)(t_diff.total_size()); // TO DO TensorToken.abs()
      // mse loss
      //TensorToken t_diff = t_output.get({1, 4}) - t_target_output.get({1, 4});
      //l += (t_diff*t_diff).sum()/(float)(t_diff.total_size());

      return l;
    }
    virtual TensorToken backward(const TensorToken &output, const TensorToken &target_output) override
    {
      TensorToken t_target_output = target_output.transpose();
      TensorToken t_output = output.transpose();
      TensorToken dLoss_dOutput = TensorToken(output.sizes).transpose();
      dLoss_dOutput.fill(0.f);
      // classification (binary cross entropy loss, size = 1)
      TensorToken cl_target = t_target_output.get(0);
      // distance + normal (l1 loss, size = 3 + 3)
      TensorToken t_diff = t_output.get({1, 7}) - t_target_output.get({1, 7});
      t_diff.set(0, cl_target * t_diff.get(0));
      t_diff.set(1, cl_target * t_diff.get(1));
      t_diff.set(2, cl_target * t_diff.get(2));
      t_diff.set(3, cl_target * t_diff.get(3));
      t_diff.set(4, cl_target * t_diff.get(4));
      t_diff.set(5, cl_target * t_diff.get(5));
      TensorToken ones(t_diff.sizes);
      ones.fill(1.0f);
      dLoss_dOutput.set({1, 7}, TensorToken::g_2op(TensorProgram::WHERE, t_diff, ones) + TensorToken::g_2op(TensorProgram::WHERE, -1.0f * t_diff, -1.0f * ones));
      dLoss_dOutput.set(1, cl_target * dLoss_dOutput.get(1));
      dLoss_dOutput.set(2, cl_target * dLoss_dOutput.get(2));
      dLoss_dOutput.set(3, cl_target * dLoss_dOutput.get(3));
      dLoss_dOutput.set(4, cl_target * dLoss_dOutput.get(4));
      dLoss_dOutput.set(5, cl_target * dLoss_dOutput.get(5));
      dLoss_dOutput.set(6, cl_target * dLoss_dOutput.get(6));
      // mse loss
      //TensorToken t_diff = t_output.get({1, 4}) - t_target_output.get({1, 4});
      //dLoss_dOutput.set({1, 4}, 2.0f*t_diff);

      // final transforms 
      return dLoss_dOutput.transpose();
    }
    virtual std::string get_name() override { return "NBVHLoss"; }
  };
}

#endif