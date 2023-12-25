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

namespace tp
{
  using AxisIdType = uint8_t;
  using IndexType = uint32_t;
  using Shape = std::vector<IndexType>;
  using ValueType = float;
  constexpr int MAX_DIM = 8;
  struct DimInfo
  {
    IndexType size = 0;
    IndexType step = 0;
  };
  using Scheme = std::array<DimInfo, MAX_DIM>;
  using FullIndex = std::array<IndexType, MAX_DIM>;

  struct TensorView
  {
    int Dim = 0;
    ValueType *_data = nullptr;
    Scheme scheme;
    IndexType total_size = 0;

    TensorView() = default; //empty view
    TensorView(ValueType *data, const Scheme &_scheme);
    TensorView(ValueType *data, const std::vector<IndexType> &sizes);
    ~TensorView() = default;
    TensorView(const TensorView &other) = default;
    TensorView(TensorView &&other) = default;
    TensorView &operator=(const TensorView &other) = default;
    TensorView &operator=(TensorView &&other) = default;

    inline IndexType size(int dimension) const
    { return scheme[dimension].size; }

    inline const ValueType &get(IndexType i) const 
      { return _data[i]; }
    inline ValueType &get(IndexType i) 
      { return _data[i]; }

    //TODO: support non-compact tensors
    inline const ValueType &get(IndexType column, IndexType row) const 
      { return _data[scheme[1].step*row + column]; }
    inline ValueType &get(IndexType column, IndexType row) 
      { return _data[scheme[1].step*row + column]; }
  
    //TODO: support non-compact tensors
    inline const ValueType &get(IndexType column, IndexType row, IndexType layer) const 
      { return _data[scheme[2].step*layer + scheme[1].step*row + column]; }
    inline ValueType &get(IndexType column, IndexType row, IndexType layer) 
      { return _data[scheme[2].step*layer + scheme[1].step*row + column]; }
  
    //TODO: support non-compact tensors
    inline const ValueType &get(IndexType column, IndexType row, IndexType layer, IndexType group) const 
      { return _data[scheme[3].step*group + scheme[2].step*layer + scheme[1].step*row + column]; }
    inline ValueType &get(IndexType column, IndexType row, IndexType layer, IndexType group) 
      { return _data[scheme[3].step*group + scheme[2].step*layer + scheme[1].step*row + column]; }
  };

  class Tensor
  {

  };

  int get_dimensions(const Scheme &scheme);
  Scheme get_scheme(const std::vector<IndexType> &sizes);
  IndexType get_total_size(const Scheme &scheme, int Dim);
  IndexType get_total_size(const std::vector<IndexType> &sizes);
  FullIndex linear_to_full_index(const TensorView &t, IndexType index);
  int compact_dims(const Scheme &scheme);

  void print(const TensorView &view);
  TensorView reshape(const TensorView &t, const std::vector<IndexType> &sizes);
  TensorView reshape(const TensorView &t, const Scheme &scheme);
  TensorView slice(const TensorView &t, IndexType index);
  TensorView slice(const TensorView &t, std::pair<IndexType, IndexType> range);

  void fill(TensorView &t, ValueType value);
  void vec_mul(const TensorView &t, const TensorView &v, TensorView out);
  void transpose(const TensorView &t, TensorView out);
  void sum(const TensorView &t, TensorView out, const std::vector<int> &dimensions);
  void add(TensorView t1, const TensorView &t2);
  void add(const TensorView &t1, const TensorView &t2, TensorView out);
  void mul(TensorView t1, float a);
  void mul(const TensorView &t1, float a, TensorView out);

  //TODO: support non-compact tensors
  #define TV_ITERATE(t, step_size, offset, F) \
  {\
    IndexType steps_count = t.total_size/step_size; \
    for (IndexType __step=0; __step<steps_count; __step++)\
    {\
      offset = __step*step_size;\
      F\
    }\
  }

  //TODO: support non-compact tensors
  #define TV_ITERATE_2(t1, step_size1, offset1, t2, step_size2, offset2, F) \
  {\
    IndexType steps_count = t1.total_size/step_size; \
    for (IndexType __step=0; __step<steps_count; __step++)\
    {\
      offset1 = __step*step_size1;\
      offset2 = __step*step_size2;\
      F\
    }\
  }

  TensorView::TensorView(ValueType *data, const std::vector<IndexType> &sizes):
    TensorView(data, get_scheme(sizes))
  {

  }

  TensorView::TensorView(ValueType *data, const Scheme &_scheme):
    Dim(get_dimensions(_scheme)),
    _data(data),
    scheme(_scheme),
    total_size(get_total_size(_scheme, get_dimensions(_scheme)))
  {

  }

  int get_dimensions(const Scheme &scheme)
  {
    int dims = 0;
    while (dims < MAX_DIM && scheme[dims].size > 0)
      dims++;
    return dims;
  }

  Scheme get_scheme(const std::vector<IndexType> &sizes)
  {
    assert(sizes.size() <= MAX_DIM);
    IndexType step = 1;
    Scheme s;
    for (int i=0;i<sizes.size();i++)
    {
      s[i].size = sizes[i];
      s[i].step = step;
      step *= sizes[i];
    }
    return s;
  }

  IndexType get_total_size(const Scheme &scheme, int Dim)
  {
    if (Dim == 0)
      return 0;
    //TODO: support non-compact tensors
    return scheme[Dim-1].size*scheme[Dim-1].step;
  }

  IndexType get_total_size(const std::vector<IndexType> &sizes)
  {
    IndexType total_sz = 1;
    for (auto sz : sizes)
      total_sz *= sz;
    return total_sz;
  }

  FullIndex linear_to_full_index(const TensorView &t, IndexType index)
  {
    FullIndex full_index;
    for (int i=0;i<t.Dim;i++)
    {
      full_index[i] = index % t.scheme[i].size;
      index /= t.scheme[i].size;
    }
    return full_index;
  } 

  void print_scheme(int Dim, const Scheme &scheme)
  {
    printf("[ ");
    for (int i = 0; i < Dim; i++)
      printf("%u ", scheme[i].size);
    printf("]");
  }

  int compact_dims(const Scheme &scheme)
  {
    //TODO: support non-compact tensors
    return get_dimensions(scheme);
  }

  void print(const TensorView &t)
  {
    FullIndex prev_i({0u});
    std::vector<std::string> delims = {" ", "\n", "\n========\n", "\n\n#####4#####\n\n",
                                       "\n\n#####5#####\n\n", "\n\n#####6#####\n\n", "\n\n#####7#####\n\n"};
    printf("%d-dimentional tensor ", t.Dim);
    print_scheme(t.Dim, t.scheme);
    printf("\n");

    IndexType offset = 0;
    TV_ITERATE(t, 1, offset, 
    {
      const ValueType &val = t.get(offset);
      FullIndex index = linear_to_full_index(t, offset);
      int delim = -1;
      for (int i=0;i<t.Dim;i++)
      {
        if (index[i] != prev_i[i])
          delim = i;
      }
      if (delim >= 0)
        printf("%s", delims[std::min((int)(delims.size()-1), delim)].c_str());
      prev_i = index;

      if constexpr (std::is_floating_point<ValueType>::value)
        printf("%8.4f ",(float)val);
      else
        printf("%d ",(int)val); 
    });
    printf("\n");
  }

  // Creates new tensor view with the same data, but different interpretation
  TensorView reshape(const TensorView &t, const Scheme &scheme)
  {
    int non_compact = t.Dim - compact_dims(t.scheme);

    IndexType new_size = 1;
    int new_dims = get_dimensions(scheme);
    for (int i=0;i<new_dims;i++)
      new_size *= scheme[i].size;
    
    //total size shouldn't change
    assert(new_size == t.total_size);

    //should have the same number of non-compact dims
    assert(non_compact == new_dims - compact_dims(scheme));

    //it should preserve non-compact dims layout, i.e. 
    //[20,10,<5>] --> [2,10,10,<5>] is ok
    //[10, <20> ] --> [2, ]
    for (int i=0;i<non_compact;i++)
    {
      assert(t.scheme[t.Dim-i-1].size == scheme[new_dims-i-1].size);
      assert(t.scheme[t.Dim-i-1].step == scheme[new_dims-i-1].step);
    }

    return TensorView(t._data, scheme);
  }
  TensorView reshape(const TensorView &t, const std::vector<IndexType> &sizes)
  {
    return reshape(t, get_scheme(sizes));
  }

  // Returns (Dim-1)-dimentional tensor with given index
  TensorView slice(const TensorView &t, IndexType index)
  {
    assert(t.Dim>1);
    assert(index < t.scheme[t.Dim-1].size);

    Scheme new_scheme;
    for (int i=0;i<t.Dim-1;i++)
      new_scheme[i] = t.scheme[i];

    //TODO: support non-compact tensors
    IndexType offset = index*t.scheme[t.Dim-1].step;
    return TensorView(t._data + offset, new_scheme);
  }

  // Returns (Dim-1)-dimentional tensor with given range [first, second)
  TensorView slice(const TensorView &t, std::pair<IndexType, IndexType> range)
  {
    assert(t.Dim == compact_dims(t.scheme));
    assert(range.first < range.second);
    assert(range.first < t.scheme[t.Dim-1].size);
    assert(range.second <= t.scheme[t.Dim-1].size);

    Scheme new_scheme = t.scheme;
    new_scheme[t.Dim-1].size = range.second - range.first;
    IndexType offset = range.first*t.scheme[t.Dim-1].step;
    return TensorView(t._data + offset, new_scheme);
  }

  void fill(TensorView &t, ValueType value)
  {
    if (compact_dims(t.scheme) == t.Dim)
      std::fill_n(t._data, t.total_size, value);
    else
    {
      IndexType offset = 0;
      TV_ITERATE(t, 1, offset, {t.get(offset) = value;});
    }
  }

  // Multiplies tensor with Dim>=2 by a vertex
  // if Dim=2 it is standart matrix-vector multiplication
  // if Dim>2 tensor is treated as a (Dim-2)-dimentional array of matrices, each
  // of them is multiplied by a vector. 
  // out is a (Dim-1)-dimentional tensor 
  void vec_mul(const TensorView &t, const TensorView &v, TensorView out)
  {
    assert(compact_dims(t.scheme) >= 2);
    assert(compact_dims(out.scheme) >= 1);
    assert(compact_dims(v.scheme) >= 1);
    assert(v.scheme[0].size==t.scheme[0].size);
    assert(out.Dim == t.Dim-1);
    assert(out.scheme[0].size == t.scheme[1].size);
    for (int i=2;i<t.Dim;i++)
      assert(t.scheme[i].size == out.scheme[i-1].size);

    IndexType step_size = v.total_size;
    IndexType offset1 = 0;
    IndexType offset2 = 0;
    TV_ITERATE_2(t,t.scheme[0].size,offset1, out,1,offset2,
    {
      out.get(offset2) = 0;
      for (IndexType i = 0; i < step_size; i++)
        out.get(offset2) += v.get(i) * t.get(offset1 + i);
    });
  }

  // Tensor is treated as a (Dim-2)-dimentional array of compact matrices,
  // each of them is transposed
  void transpose(const TensorView &t, TensorView out)
  {
    assert(compact_dims(t.scheme) >= 2);
    assert(compact_dims(out.scheme) >= 2);
    assert(out.Dim == t.Dim);
    assert(t.scheme[0].size == out.scheme[1].size);
    assert(t.scheme[1].size == out.scheme[0].size);
    for (int i=2;i<t.Dim;i++)
      assert(t.scheme[i].size == out.scheme[i].size);
    assert(t.scheme[0].size == out.scheme[1].size);
    assert(t.scheme[1].size == out.scheme[0].size);

    for (int i = 2; i < t.Dim; i++)
      assert(t.scheme[i].size == out.scheme[i].size);

    IndexType step_size = t.scheme[0].size * t.scheme[1].size;
    IndexType offset1 = 0;
    IndexType offset2 = 0;
    TV_ITERATE_2(t,step_size,offset1, out,step_size,offset2,
    {
      for (IndexType i = 0; i < t.scheme[1].size; i++)
        for (IndexType j = 0; j < t.scheme[0].size; j++)
          out.get(offset2 + j * t.scheme[1].size + i) = t.get(offset1 + i * t.scheme[0].size + j);
    });
  }

  //t1 += t2
  void add(TensorView t1, const TensorView &t2)
  {
    add(t1, t2, t1);
  }

  //out = t1+t2
  //if Dim(t1) = Dim(t2) perform element-wise operation
  //else if Dim(t1) > Dim(t2), t1 is treated as an array of Dim(t2)
  //tensors and t2 is added to each of them
  //out should have exactly the same size as t1 by every dimension
  //t2 should be compact
  void add(const TensorView &t1, const TensorView &t2, TensorView out)
  {
    assert(t1.Dim == out.Dim);
    for (int i=0;i<t1.Dim;i++)
      assert(t1.scheme[i].size == out.scheme[i].size);
    
    assert(t1.Dim >= t2.Dim);
    assert(compact_dims(t2.scheme) == t2.Dim);
    for (int i=0;i<t2.Dim;i++)
      assert(t1.scheme[i].size == t2.scheme[i].size);
    
    IndexType step_size = t2.total_size;
    IndexType offset1 = 0;
    IndexType offset2 = 0;
    TV_ITERATE_2(t1,step_size,offset1, out,step_size,offset2,
    {
      for (IndexType i = 0; i < step_size; i++)
        out.get(offset2 + i) = t1.get(offset1 + i) + t2.get(i);
    });
  }

  //t1 *= a
  void mul(TensorView t1, float a)
  {
    mul(t1, a, t1);
  }

  //out = a*t1
  //Perform element-wise operation
  //out should have exactly the same size as t1 by every dimension
  void mul(const TensorView &t1, float a, TensorView out)
  {
    assert(t1.Dim == out.Dim);
    for (int i=0;i<t1.Dim;i++)
      assert(t1.scheme[i].size == out.scheme[i].size);
    
    IndexType step_size = t1.total_size;
    IndexType offset1 = 0;
    IndexType offset2 = 0;
    TV_ITERATE_2(t1,step_size,offset1, out,step_size,offset2,
    {
      for (IndexType i = 0; i < step_size; i++)
        out.get(offset2 + i) = a*t1.get(offset1 + i);
    });
  }

  void sum(const TensorView &t, TensorView out, const std::vector<int> &dimensions)
  {

  }

  // performs operation ⊗ between two vectors
  // returns matrix
  // [x, y]⊗[a, b] = [x*a, x*b]
  //                 [y*a, y*b] 
  void vector_outer_product(const TensorView &v1, const TensorView &v2, TensorView out)
  {
    assert(v1.Dim == 1);
    assert(compact_dims(v1.scheme) == 1);
    assert(v2.Dim == 1);
    assert(compact_dims(v2.scheme) == 1);
    assert(out.Dim == 2);
    assert(compact_dims(out.scheme) == 2);
    assert(v1.total_size*v2.total_size == out.total_size);

    for (IndexType i=0;i<v1.total_size;i++)
      for (IndexType j=0;j<v2.total_size;j++)
        out.get(i*v1.total_size + j) = v1.get(i)*v2.get(j);
  }
}
using namespace tp;

class Layer
{
public:
  tp::Shape input_shape, output_shape;

  virtual void init(int batch_size, float *param_mem, float *gradient_mem, float *tmp_mem) = 0;
  virtual void forward(const TensorView &input, TensorView &output) = 0; 
  virtual void backward(const TensorView &input, const TensorView &output, 
                        const TensorView &dLoss_dOutput, TensorView dLoss_dInput, bool first_layer) = 0; 
  virtual int parameters_memory_size() = 0;
  virtual int tmp_memory_size() = 0;
};

class DenseLayer : public Layer
{
public:
  DenseLayer() = default;
  DenseLayer(int input_size, int output_size);
  virtual void init(int batch_size, float *param_mem, float *gradient_mem, float *tmp_mem) override;
  virtual void forward(const TensorView &input, TensorView &output) override; 
  virtual void backward(const TensorView &input, const TensorView &output, 
                        const TensorView &dLoss_dOutput, TensorView dLoss_dInput, bool first_layer) override; 
  virtual int parameters_memory_size() override;
  virtual int tmp_memory_size() override;
private:
  int batch_size;
  TensorView A, b;
  TensorView dLoss_dA, dLoss_db;
  TensorView At, op;
};

DenseLayer::DenseLayer(int input_size, int output_size)
{
  input_shape.push_back(input_size);
  output_shape.push_back(output_size);
}

void DenseLayer::init(int _batch_size, float *param_mem, float *gradient_mem, float *tmp_mem)
{
  batch_size = _batch_size;
  
  A = TensorView(param_mem + 0, Shape{input_shape[0], output_shape[0]});
  b = TensorView(param_mem + A.total_size, Shape{output_shape[0]});

  dLoss_dA = TensorView(gradient_mem + 0, Shape{input_shape[0], output_shape[0]});
  dLoss_db = TensorView(gradient_mem + A.total_size, Shape{output_shape[0]});

  At = TensorView(tmp_mem + 0, Shape{output_shape[0], input_shape[0]});
  op = TensorView(tmp_mem + 0, Shape{input_shape[0], output_shape[0]});//never used together
}

void DenseLayer::forward(const TensorView &input, TensorView &output)
{
  for (int i=0;i<batch_size;i++)
    vec_mul(A, slice(input, i), slice(output, i));
  add(output, b);
  //printf("AA");
  //print(slice(input, 0));
  //print(slice(output, 0));
  //print(A);
}

void DenseLayer::backward(const TensorView &input, const TensorView &output, 
                        const TensorView &dLoss_dOutput, TensorView dLoss_dInput, bool first_layer)
{
  if (!first_layer)
  {
    //dLoss_dInput = A^T * dloss_dOutput
    transpose(A, At);
    for (int i=0;i<batch_size;i++)
      vec_mul(At, slice(dLoss_dOutput, i), slice(dLoss_dInput, i));
  }
  
  //dLoss_dA = average{batches}(dLoss_dOutput ⊗ input)
  fill(dLoss_dA, 0);
  for (int i=0;i<batch_size;i++)
  {
    vector_outer_product(slice(dLoss_dOutput, i), slice(input, i), op);
    add(dLoss_dA, op);
  }
  mul(dLoss_dA, 1.0f/batch_size);

  //dLoss_db = average{batches}(dLoss_dOutput);
  fill(dLoss_db, 0);
  for (int i=0;i<batch_size;i++)
    add(dLoss_db, slice(dLoss_dOutput, i));
  mul(dLoss_db, 1.0f/batch_size);
}

int DenseLayer::parameters_memory_size()
{
  return (input_shape[0] + 1)*output_shape[0];
}

int DenseLayer::tmp_memory_size()
{
  return input_shape[0]*output_shape[0];
}

class Optimizer
{
public:
  Optimizer(int _params_count)
  {
    params_count = _params_count;
  }
  virtual void step(float *params_ptr, float const *grad_ptr) = 0;
protected:
  int params_count = 0;
};

class OptimizerGD : public Optimizer
{
public:
  OptimizerGD(int _params_count, float _lr = 0.01f):
  Optimizer(_params_count)
  {
    lr = _lr;
  }
  virtual void step(float *params_ptr, float const *grad_ptr) override
  {
    for (int i=0;i<params_count;i++)
      params_ptr[i] -= lr*grad_ptr[i];
  }
private:
  float lr;
};

class OptimizerAdam : public Optimizer
{
public:
  OptimizerAdam(int _params_count, float _lr = 0.01f, float _beta_1 = 0.9f, float _beta_2 = 0.999f, float _eps = 1e-8):
  Optimizer(_params_count)
  {
    lr = _lr;
    beta_1 = _beta_1;
    beta_2 = _beta_2;
    eps = _eps;

    V = std::vector<float>(_params_count, 0); 
    S = std::vector<float>(_params_count, 0); 
  }
  virtual void step(float *params_ptr, float const *grad_ptr) override
  {
    for (int i = 0; i < params_count; i++)
    {
      float g = grad_ptr[i];
      V[i] = beta_1 * V[i] + (1 - beta_1) * g;
      float Vh = V[i] / (1 - pow(beta_1, iter + 1));
      S[i] = beta_2 * S[i] + (1 - beta_2) * g * g;
      float Sh = S[i] / (1 - pow(beta_2, iter + 1));
      params_ptr[i] -= lr * Vh / (sqrt(Sh) + eps);
    }
    iter++;
  }

private:
  float lr, beta_1, beta_2, eps;
  int iter = 0;
  std::vector<float> V;
  std::vector<float> S;
};


void loss_MSE(const TensorView &values/*[output_size, batch_size]*/, const TensorView &target_values/*[output_size, batch_size]*/,
              TensorView loss/*[batch_size]*/, TensorView dLoss_dValues/*[output_size, batch_size]*/)
{
  int len = values.size(0);
  int batches = values.size(1);
  assert(len == 1);

  if (len == 1)
  {
    for (int i=0;i<batches;i++)
    {
      float l = values.get(0, i) - target_values.get(0, i);
      loss.get(i) = l*l;
      dLoss_dValues.get(0, i) = 2*l;
    }
  }
}

class NeuralNetwork
{
public:
  enum Opt
  {
    GD,
    Adam
  };
  enum Loss
  {
    MSE
  };

  using LossFunction = std::function<void(const TensorView &, const TensorView &, TensorView, TensorView)>;

  void add_layer(std::shared_ptr<Layer> layer);
  bool initialize(const float *weights = nullptr);
  void train(const TensorView &inputs/*[input_size, count]*/, const TensorView &outputs/*[output_size, count]*/,
             int batch_size, int epochs, Opt optimizer, Loss loss, float lr = 0.1f);
  void evaluate(const TensorView &input, TensorView output);
  NeuralNetwork() {};
  NeuralNetwork(const NeuralNetwork &other) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &other) = delete;
private:
  std::vector<std::shared_ptr<Layer>> layers;
  std::vector<float> weights;
};

  void NeuralNetwork::add_layer(std::shared_ptr<Layer> layer)
  {
    layers.push_back(layer);
  }

  bool NeuralNetwork::initialize(const float *init_weights)
  {
    for (int i=1;i<layers.size();i++)
    {
      if (layers[i]->input_shape.size() != layers[i-1]->output_shape.size())
      {
        printf("NeuralNetwork: layers %d and %d have incompatible shapes!\n", i-1, i);
        return false;
      }
      for (int j=0;j<layers[i]->input_shape.size();j++)
      {
        if (layers[i]->input_shape[j] != layers[i-1]->output_shape[j])
        {
          printf("NeuralNetwork: layers %d and %d have incompatible sizes!\n", i-1, i);
          return false;
        }
      }
    }

    int total_params = 0;
    for (int i=0;i<layers.size();i++)
    {
      printf("Layer %d has %d parameters\n", i, layers[i]->parameters_memory_size());
      total_params += layers[i]->parameters_memory_size();
    }

    if (init_weights)
      weights = std::vector<float>(init_weights, init_weights+total_params);
    else
      weights = std::vector<float>(total_params, 0);

    printf("Neural Network succesfully created\n");
    printf("%d layers\n", (int)(layers.size()));
    printf("%d input size\n", get_total_size(layers[0]->input_shape));
    printf("%d output size\n", get_total_size(layers.back()->output_shape));
    printf("%d weights\n", total_params);

    return true;
  }

  void NeuralNetwork::train(const TensorView &inputs/*[input_size, count]*/, const TensorView &outputs/*[output_size, count]*/,
             int batch_size, int epochs, Opt opt, Loss loss, float lr)
  {
    //check if the input is correct
    assert(inputs.Dim == 2);
    assert(compact_dims(inputs.scheme) == 2);
    assert(outputs.Dim == 2);
    assert(compact_dims(outputs.scheme) == 2);
    assert(inputs.size(1) == outputs.size(1));
    assert(batch_size > 0 && batch_size <= inputs.size(1));

    //calculate memory requirements
    int layers_tmp_size = 1;
    int io_size = 0;
    int network_tmp_size = get_total_size(layers[0]->input_shape);
    for (auto &l : layers)
    {
      layers_tmp_size = std::max(layers_tmp_size, l->tmp_memory_size());
      io_size += batch_size*get_total_size(l->output_shape);
      network_tmp_size = std::max(network_tmp_size, (int)get_total_size(l->output_shape));
    }
    network_tmp_size *= batch_size;

    //check if the network is correct
    bool inited = initialize();
    if (!inited)
      return;
    
    //allocate memory needed for training
    std::vector<float> gradients(weights.size(), 0);
    std::vector<float> tmp_mem(layers_tmp_size, 0);
    std::array<std::vector<float>, 2> network_tmp_mem = {std::vector<float>(network_tmp_size, 0), std::vector<float>(network_tmp_size, 0)};
    std::vector<float> loss_tmp_mem(batch_size, 0);
    std::vector<float> io_mem(io_size, 0);

    //initialize weights. Uniform in [0, 1]
    for (auto &w : weights)
      w = ((float)rand())/RAND_MAX;

    //initialize optimizer
    std::unique_ptr<Optimizer> optimizer;
    if (opt == Opt::GD)
      optimizer.reset(new OptimizerGD(weights.size(), lr));
    else if (opt == Opt::Adam)
      optimizer.reset(new OptimizerAdam(weights.size(), lr));
    
    //initialize loss function
    LossFunction loss_function;
    if (loss == Loss::MSE)
      loss_function = loss_MSE;
    
    //create tensors to store layers' outputs
    TensorView loss_batch(loss_tmp_mem.data(), Shape{(IndexType)batch_size});
    std::vector<TensorView> layer_outputs;
    std::vector<TensorView> layer_dLoss_dOutputs;
    float *cur_io_ptr = io_mem.data();
    int li = 0;
    for (auto &l : layers)
    {
      Shape shape = l->output_shape;
      shape.push_back(batch_size);
      layer_outputs.push_back(TensorView(cur_io_ptr, shape));
      cur_io_ptr += get_total_size(shape);

      layer_dLoss_dOutputs.push_back(TensorView(network_tmp_mem[li].data(), shape));
      li = (li+1)%2;
    }

    //init layer for training
    float *cur_param_ptr = weights.data();
    float *cur_grad_ptr = gradients.data();
    for (auto &l : layers)
    {
      l->init(batch_size, cur_param_ptr, cur_grad_ptr, tmp_mem.data());
      cur_param_ptr += l->parameters_memory_size();
      cur_grad_ptr += l->parameters_memory_size();
    }

    //start main training loop
    int count = inputs.size(inputs.Dim-1);
    int batches = count/batch_size;
    int iterations = epochs*batches;

    for (int iter=0;iter<iterations;iter++)
    {
      int batch_id = rand()%batches;
      //batch_id = 0;
      TensorView input_batch = slice(inputs, {batch_id*batch_size, (batch_id+1)*batch_size});
      TensorView target_output_batch = slice(outputs, {batch_id*batch_size, (batch_id+1)*batch_size});
      TensorView cur_input = input_batch;

      //forward pass
      layers[0]->forward(input_batch, layer_outputs[0]);
      for (int i=1;i<layers.size();i++)
      {
        layers[i]->forward(layer_outputs[i-1], layer_outputs[i]);
      }

      loss_function(layer_outputs.back(), target_output_batch, loss_batch, layer_dLoss_dOutputs.back());

      //backward pass
      for (int i=layers.size()-1;i>=1;i--)
        layers[i]->backward(layer_outputs[i-1], layer_outputs[i], layer_dLoss_dOutputs[i], layer_dLoss_dOutputs[-1], false);
      layers[0]->backward(input_batch, layer_outputs[0], layer_dLoss_dOutputs[0], TensorView(), true);
/*
      printf("weights [ ");
      for (auto &w: weights)
        printf("%f ", w);
      printf("]\n");
      printf("grads [ ");
      for (auto &w: gradients)
        printf("%f ", w);
      printf("]\n");


      print(layer_outputs.back());
      print(target_output_batch);
      print(loss_batch);*/
      if (iter%batches == 0)
      {
        //print(loss_batch);
        float average_loss = 0;
        for (int i=0;i<batch_size;i++)
          average_loss += loss_batch.get(i);
        average_loss /= batch_size;
        printf("epoch %d: average loss %f\n", iter/batches, average_loss);
      }
      
      optimizer->step(weights.data(), gradients.data());
    }
          printf("weights [ ");
      for (auto &w: weights)
        printf("%f ", w);
      printf("]\n");
  }

  void NeuralNetwork::evaluate(const TensorView &input, TensorView output)
  {

  }

  int main(int argc, char **argv)
  {
    /*
    std::vector<float> values = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Scheme s1;
    s1[0] = {10,1};
    Scheme s2;
    s2[0] = {4,1};
    s2[1] = {4,4};
    Scheme s3;
    s3[0] = {3,1};
    s3[1] = {2,3};
    s3[2] = {2,6};
    TensorView tv1(values.data(), Shape{10});
    TensorView tv2(values.data(), Shape{4, 4});
    TensorView tv3(values.data(), Shape{2, 2, 4});
    print(tv1);
    print(tv2);
    print(tv3);

    float *data = new float[1000];
    TensorView tmp(data, tv3.scheme);
    transpose(tv3, tmp);
    print(tmp);
    print(slice(tmp, 2));
    print(slice(slice(tmp, 2),1));
    tmp = reshape(tmp, tv2.scheme);
    print(tmp);
    transpose(tv2, tmp);
    print(tmp);

    printf("2 3 %f 3 2 %f\n", tmp.get(2,3), tmp.get(3,2));

    delete[] data;
    */

    std::vector<float> X, y;
    uint size = 1000;
    uint dim = 100;

    //y = 3*x + 1
    for (int i=0;i<size;i++)
    {
      float r = 1;
      for (int j=0;j<dim;j++)
      {
        float x0 = 2*((float)rand())/RAND_MAX - 1;
        X.push_back(x0);
        r += ((j%2) ? 0.5 : -0.5)*x0;
      }
      y.push_back(r);
    }

    TensorView Xv(X.data(), Shape{dim, size});
    TensorView yv(y.data(), Shape{1  , size});

    print(Xv);
    print(yv);

    NeuralNetwork nn;
    nn.add_layer(std::make_shared<DenseLayer>(dim, 1));
    nn.train(Xv, yv, 100, 60, NeuralNetwork::Opt::Adam, NeuralNetwork::Loss::MSE, 0.01);

    return 0;
  }