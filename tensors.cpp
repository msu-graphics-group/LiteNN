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

struct tp
{
  using AxisIdType = uint8_t;
  using IndexType = uint32_t;

#define FOR_EACH(t, F)                              \
  {                                                 \
    for (IndexType li = 0; li < t.total_size; li++) \
    {                                               \
      T &val = t.data[li];                          \
      F                                             \
    }                                               \
  }

#define REDUCE(t, F, init) \
  {                        \
    T acc = init;          \
    FOR_EACH(t, F);        \
    return acc;            \
  }

#define DIM_REDUCE(In, Out, F, init)                          \
  {                                                           \
    static_assert(ReduceDim < Dim);                           \
    (Out).fill(init);                                         \
    IndexType step_size = (In).total_size / (Out).total_size; \
    for (IndexType step = 0; step < (Out).total_size; step++) \
    {                                                         \
      for (IndexType i = 0; i < step_size; i++)               \
      {                                                       \
        const T &val = (In).get(step * step_size + i);        \
        T &acc = (Out).get(step);                             \
        F                                                     \
      }                                                       \
    }                                                         \
  }

#define SIMPLE_MATH(t1, t2, out, F)                                                       \
  {                                                                                       \
    static_assert(SecondDim <= Dim);                                                      \
    bool match = true;                                                                    \
    for (int i = 0; i < SecondDim; i++)                                                   \
      if (t1.scheme[i].size != t2.scheme[i].size)                                         \
        match = false;                                                                    \
    if (!match)                                                                           \
    {                                                                                     \
      printf("Tensors have incompatible dimensions: ");                                   \
      print_scheme(t1.scheme);                                                            \
      print_scheme(t2.scheme);                                                            \
      printf("\n");                                                                       \
      return;                                                                             \
    }                                                                                     \
    IndexType step_size = (SecondDim == Dim) ? t1.total_size : t1.scheme[SecondDim].step; \
    IndexType steps = t1.total_size / step_size;                                          \
    for (IndexType step = 0; step < steps; step++)                                        \
    {                                                                                     \
      for (IndexType i = 0; i < step_size; i++)                                           \
      {                                                                                   \
        const T &v1 = t1.get(step * step_size + i);                                       \
        const T &v2 = t2.get(i);                                                          \
        T &res = out.get(step * step_size + i);                                           \
        F                                                                                 \
      }                                                                                   \
    }                                                                                     \
  }

  template <typename T, int Dim>
  struct Tensor
  {
    friend struct tp;
    using Index = std::array<IndexType, Dim>;
    using Subspace = std::array<AxisIdType, Dim>;

    struct DimInfo
    {
      IndexType size = 0;
      IndexType step = 0;
    };
    using Scheme = std::array<DimInfo, Dim>;

  private:
    T *data = nullptr;
    IndexType total_size = 0;
    Scheme scheme;

    template <int DimN>
    inline void iterate(Index &i, IndexType &li, const std::function<void(T &, const Index &)> &func)
    {
      while (i[DimN] < scheme[DimN].size)
      {
        if constexpr (DimN == 0)
          func(get(li++), i);
        else
        {
          i[DimN - 1] = 0;
          iterate<DimN - 1>(i, li, func);
        }
        i[DimN]++;
      }
    }
    inline void for_each(const std::function<void(T &, const Index &)> &func)
    {
      Index i = Index({0u});
      IndexType li = 0u;
      iterate<Dim - 1>(i, li, func);
    }

    template <int DimN>
    inline void iterate(Index &i, IndexType &li, const std::function<void(const T &, const Index &)> &func) const
    {
      while (i[DimN] < scheme[DimN].size)
      {
        if constexpr (DimN == 0)
          func(get(li++), i);
        else
        {
          i[DimN - 1] = 0;
          iterate<DimN - 1>(i, li, func);
        }
        i[DimN]++;
      }
    }
    inline void for_each(const std::function<void(const T &, const Index &)> &func) const
    {
      Index i = Index({0u});
      IndexType li = 0u;
      iterate<Dim - 1>(i, li, func);
    }

  public:
    Tensor() = default;

    ~Tensor()
    {
      delete[] data;
    }

    explicit Tensor(const std::array<IndexType, Dim> &shape) : Tensor((T *)nullptr, shape)
    {
    }

    explicit Tensor(const std::vector<T> &values, const std::array<IndexType, Dim> &shape) : Tensor(values.data(), shape)
    {
    }

    explicit Tensor(const T *raw_data, const std::array<IndexType, Dim> &shape)
    {
      Scheme s;
      IndexType step = 1;
      for (int i = 0; i < Dim; i++)
      {
        s[i] = {shape[i], step};
        step *= shape[i];
      }
      init(raw_data, s);
    }

    explicit Tensor(const T *raw_data, const Scheme &s)
    {
      init(raw_data, s);
    }

    Tensor(const Tensor &other) // copy constructor
        : Tensor(other.data, other.scheme)
    {
      printf("COPY constructor\n");
    }

    Tensor(Tensor &&other) noexcept // move constructor
    {
      scheme = other.scheme;
      total_size = other.total_size;
      data = other.data;

      other.data = nullptr;
    }

    void init(const T *raw_data, const Scheme &s)
    {
      if (data)
        delete[] data;
      scheme = s;
      total_size = s.back().size * s.back().step;

      data = new T[total_size];
      if (raw_data)
        std::memcpy(data, raw_data, sizeof(T) * total_size);
    }

    Tensor &operator=(const Tensor &other) // copy assignment
    {
      printf("COPY\n");
      return *this = Tensor(other);
    }

    Tensor &operator=(Tensor &&other) noexcept // move assignment
    {
      std::swap(data, other.data);
      std::swap(scheme, other.scheme);
      std::swap(total_size, other.total_size);
      return *this;
    }

    Scheme get_scheme() const
    {
      return scheme;
    }

    IndexType get_size()
    {
      return total_size;
    }

    inline IndexType linear_index(const Index &i)
    {
      IndexType li = 0;
      for (AxisIdType di = 0; di < Dim; di++)
        li += scheme[di].step * i[di];
      return li;
    }

    inline T &get(const IndexType &li)
    {
      return data[li];
    }
    inline const T &get(const IndexType &li) const
    {
      return data[li];
    }

    inline T &operator[](const Index &i)
    {
      return get(linear_index(i));
    }
    inline const T &operator[](const Index &i) const
    {
      return get(linear_index(i));
    }
  };
  template <typename T, int Dim>
  static inline void fill(Tensor<T, Dim> &t, T value)
  {
    std::fill_n(t.data, t.total_size, value);
  }

  template <typename T, int Dim>
  static inline void print_index(const typename Tensor<T, Dim>::Index &index)
  {
    printf("[ ");
    for (int i = 0; i < Dim; i++)
      printf("%u ", index[i]);
    printf("]");
  }

  template <typename T, int Dim>
  static inline void print_scheme(const typename Tensor<T, Dim>::Scheme &scheme)
  {
    printf("[ ");
    for (int i = 0; i < Dim; i++)
      printf("%u ", scheme[i].size);
    printf("]");
  }

  template <typename T, int Dim>
  static inline void print(const Tensor<T, Dim> &t)
  {
    printf("%d-dimentional tensor ", Dim);
    print_scheme<T, Dim>(t.scheme);
    printf("\n");
    t.for_each([&t](const T &val, const typename Tensor<T, Dim>::Index &index)
               {
      print_index(index);
      if constexpr (std::is_floating_point<T>::value)
        printf("%f ",(float)val);
      else
        printf("%d ",(int)val);
      printf("\n"); });
    printf("-------------\n");
  }

  template <typename T, int Dim>
  static inline void compact_print(const Tensor<T, Dim> &t)
  {
    typename Tensor<T, Dim>::Index prev_i({0u});
    std::vector<std::string> delims = {" ", "\n", "\n========\n", "\n\n#####4#####\n\n",
                                       "\n\n#####5#####\n\n", "\n\n#####6#####\n\n", "\n\n#####7#####\n\n"};
    printf("%d-dimentional tensor ", Dim);
    print_scheme<T, Dim>(t.scheme);
    printf("\n");
    t.for_each([&prev_i, &delims](const T &val, const typename Tensor<T, Dim>::Index &index)
               {
      int delim = -1;
      for (int i=0;i<Dim;i++)
      {
        if (index[i] != prev_i[i])
          delim = i;
      }
      if (delim >= 0)
        printf("%s", delims[std::min((int)(delims.size()-1), delim)].c_str());
      prev_i = index;

      if constexpr (std::is_floating_point<T>::value)
        printf("%f ",(float)val);
      else
        printf("%d ",(int)val); });
    printf("\n");
  }

  template <typename T, int Dim>
  static inline T reduce(const Tensor<T, Dim> &t, const std::function<void(T, T &)> &reduce_func, T reduce_init)
  {
    REDUCE(
        t, { reduce_func(val, acc); }, reduce_init);
  }

  template <typename T, int Dim>
  static inline T max(const Tensor<T, Dim> &t)
  {
    REDUCE(
        t, { acc = std::max(val, acc); }, t.get(0));
  }

  template <typename T, int Dim>
  static inline T min(const Tensor<T, Dim> &t)
  {
    REDUCE(
        t, { acc = std::min(val, acc); }, t.get(0));
  }

  template <typename T, int Dim>
  static inline T sum(const Tensor<T, Dim> &t)
  {
    REDUCE(
        t, { acc += val; }, 0);
  }

  template <typename T, int Dim>
  static inline T mean(const Tensor<T, Dim> &t)
  {
    return sum(t) / t.total_size;
  }

  template <typename T, int Dim, int ReduceDim>
  static inline Tensor<T, Dim - ReduceDim> get_tensor(const Tensor<T, Dim> &t)
  {
    std::array<IndexType, Dim - ReduceDim> output_sizes;
    for (int i = ReduceDim; i < Dim; i++)
      output_sizes[i - ReduceDim] = t.scheme[i].size;

    Tensor<T, Dim - ReduceDim> output_tensor = Tensor<T, Dim - ReduceDim>(nullptr, output_sizes);

    return output_tensor;
  }

  // reduce by the last ReduceDim
  // i.e Tensor<float, 2>::reduce<1> applies reduce function to every line in matrix
  // Tensor<float, 3>::reduce<2> applies reduce function to every layer in 3D-grid
  template <typename T, int Dim, int ReduceDim>
  static inline Tensor<T, Dim - ReduceDim> reduce(const Tensor<T, Dim> &t, const std::function<void(T, T &)> &reduce_func, T reduce_init)
  {
    Tensor<T, Dim - ReduceDim> output_tensor = get_tensor<T, Dim, ReduceDim>(t);
    DIM_REDUCE(
        t, output_tensor, { reduce_func(val, acc); }, reduce_init);
    return output_tensor;
  }

  template <typename T, int Dim, int ReduceDim>
  static inline void reduce(const Tensor<T, Dim> &t, Tensor<T, Dim - ReduceDim> &output_tensor, const std::function<void(T, T &)> &reduce_func, T reduce_init)
  {
    DIM_REDUCE(
        t, output_tensor, { reduce_func(val, acc); }, reduce_init);
  }

  template <typename T, int Dim, int ReduceDim>
  static inline Tensor<T, Dim - ReduceDim> sum(const Tensor<T, Dim> &t)
  {
    Tensor<T, Dim - ReduceDim> output_tensor = get_tensor<ReduceDim>();
    DIM_REDUCE(
        t, output_tensor, { acc += val; }, 0);
    return output_tensor;
  }

  template <typename T, int Dim, int ReduceDim>
  static inline void sum(const Tensor<T, Dim> &t, Tensor<T, Dim - ReduceDim> &output_tensor)
  {
    DIM_REDUCE(
        t, output_tensor, { acc += val; }, 0);
  }

  template <typename T, int Dim, int ReduceDim>
  static inline Tensor<T, Dim - ReduceDim> max(const Tensor<T, Dim> &t)
  {
    Tensor<T, Dim - ReduceDim> output_tensor = get_tensor<ReduceDim>();
    DIM_REDUCE(
        t, output_tensor, { acc = std::max(acc, val); }, t.get(0));
    return output_tensor;
  }

  template <typename T, int Dim, int ReduceDim>
  static inline void max(const Tensor<T, Dim> &t, Tensor<T, Dim - ReduceDim> &output_tensor)
  {
    DIM_REDUCE(
        t, output_tensor, { acc = std::max(acc, val); }, t.get(0));
  }

  template <typename T, int Dim, int ReduceDim>
  static inline Tensor<T, Dim - ReduceDim> min(const Tensor<T, Dim> &t)
  {
    Tensor<T, Dim - ReduceDim> output_tensor = get_tensor<ReduceDim>();
    DIM_REDUCE(
        t, output_tensor, { acc = std::min(acc, val); }, t.get(0));
    return output_tensor;
  }

  template <typename T, int Dim, int ReduceDim>
  static inline void min(const Tensor<T, Dim> &t, Tensor<T, Dim - ReduceDim> &output_tensor)
  {
    DIM_REDUCE(
        t, output_tensor, { acc = std::min(acc, val); }, t.get(0));
  }

  template <typename T, int Dim>
  static inline void add(const Tensor<T, Dim> &t, T value, Tensor<T, Dim> &out)
  {
    assert(out.total_size == t.total_size);
    for (IndexType li = 0; li < t.total_size; li++)
      out.get(li) = t.get(li) + value;
  }
  template <typename T, int Dim>
  static inline Tensor<T, Dim> add(const Tensor<T, Dim> &t, T value)
  {
    Tensor<T, Dim> res(nullptr, t.scheme);
    add(t, value, res);
    return res;
  }

  template <typename T, int Dim, int SecondDim>
  static inline void add(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2, Tensor<T, Dim> &out)
  {
    SIMPLE_MATH(t1, t2, out, { res = v1 + v2; });
  }
  template <typename T, int Dim, int SecondDim>
  static inline Tensor<T, Dim> add(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2)
  {
    Tensor<T, Dim> res(nullptr, t1.scheme);
    add(t1, t2, res);
    return res;
  }

  template <typename T, int Dim>
  static inline void sub(const Tensor<T, Dim> &t, T value, Tensor<T, Dim> &out)
  {
    assert(out.total_size == t.total_size);
    for (IndexType li = 0; li < t.total_size; li++)
      out.get(li) = t.get(li) - value;
  }
  template <typename T, int Dim>
  static inline Tensor<T, Dim> sub(const Tensor<T, Dim> &t, T value)
  {
    Tensor<T, Dim> res(nullptr, t.scheme);
    sub(t, value, res);
    return res;
  }

  template <typename T, int Dim, int SecondDim>
  static inline void sub(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2, Tensor<T, Dim> &out)
  {
    SIMPLE_MATH(t1, t2, out, { res = v1 - v2; });
  }
  template <typename T, int Dim, int SecondDim>
  static inline Tensor<T, Dim> sub(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2)
  {
    Tensor<T, Dim> res(nullptr, t1.scheme);
    sub(t1, t2, res);
    return res;
  }

  template <typename T, int Dim>
  static inline void mul(const Tensor<T, Dim> &t, T value, Tensor<T, Dim> &out)
  {
    assert(out.total_size == t.total_size);
    for (IndexType li = 0; li < t.total_size; li++)
      out.get(li) = t.get(li) * value;
  }
  template <typename T, int Dim>
  static inline Tensor<T, Dim> mul(const Tensor<T, Dim> &t, T value)
  {
    Tensor<T, Dim> res(nullptr, t.scheme);
    mul(t, value, res);
    return res;
  }

  template <typename T, int Dim, int SecondDim>
  static inline void mul(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2, Tensor<T, Dim> &out)
  {
    SIMPLE_MATH(t1, t2, out, { res = v1 * v2; });
  }
  template <typename T, int Dim, int SecondDim>
  static inline Tensor<T, Dim> mul(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2)
  {
    Tensor<T, Dim> res(nullptr, t1.scheme);
    mul(t1, t2, res);
    return res;
  }

  template <typename T, int Dim>
  static inline void div(const Tensor<T, Dim> &t, T value, Tensor<T, Dim> &out)
  {
    assert(out.total_size == t.total_size);
    for (IndexType li = 0; li < t.total_size; li++)
      out.get(li) = t.get(li) / value;
  }
  template <typename T, int Dim>
  static inline Tensor<T, Dim> div(const Tensor<T, Dim> &t, T value)
  {
    Tensor<T, Dim> res(nullptr, t.scheme);
    div(t, value, res);
    return res;
  }

  template <typename T, int Dim, int SecondDim>
  static inline void div(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2, Tensor<T, Dim> &out)
  {
    SIMPLE_MATH(t1, t2, out, { res = v1 / v2; });
  }
  template <typename T, int Dim, int SecondDim>
  static inline Tensor<T, Dim> div(const Tensor<T, Dim> &t1, const Tensor<T, SecondDim> &t2)
  {
    Tensor<T, Dim> res(nullptr, t1.scheme);
    div(t1, t2, res);
    return res;
  }

  // multiplies tensor by a vertex
  // if Dim=1 it is dot product
  // if Dim=2 it is standart matrix multiplication
  // if Dim>2 tensor is treated as a (Dim-2)-dimentional array of matrices, each
  // of them is multiplied by a vector
  template <typename T, int Dim>
  static inline void dot(const Tensor<T, Dim> &t, const Tensor<T, 1> &v, Tensor<T, Dim - 1> &out)
  {
    if (v.total_size != t.scheme[0].size)
    {
      printf("Tensors have incompatible dimensions: ");
      print_scheme<T, Dim>(t.scheme);
      print_scheme<T, 1>(v.scheme);
      printf("\n");
      return;
    }
    IndexType step_size = v.total_size;
    IndexType steps = t.total_size / v.total_size;

    out.fill(0);
    for (IndexType step = 0; step < steps; step++)
      for (IndexType i = 0; i < step_size; i++)
        out.get(step) += v.get(i) * t.get(step * step_size + i);
  }
  template <typename T, int Dim>
  static inline Tensor<T, Dim - 1> dot(const Tensor<T, Dim> &t, const Tensor<T, 1> &v)
  {
    typename Tensor<T, Dim - 1>::Scheme res_scheme;

    IndexType step = 1;
    for (int i = 1; i < Dim; i++)
      res_scheme[i - 1] = {t.scheme[i].size, t.scheme[i].step / t.scheme[1].step};
    Tensor<T, Dim - 1> res(nullptr, res_scheme);
    dot(t, v, res);
    return res;
  }

  // tensor is treated as a (Dim-2)-dimentional array of matrices, each of
  // them is transposed
  template <typename T, int Dim>
  static inline void transpose(const Tensor<T, Dim> &t, Tensor<T, Dim> &out)
  {
    static_assert(Dim >= 2);
    assert(t.scheme[0].size == out.scheme[1].size);
    assert(t.scheme[1].size == out.scheme[0].size);
    for (int i = 2; i < Dim; i++)
      assert(t.scheme[i].size == out.scheme[i].size);

    IndexType step_size = t.scheme[0].size * t.scheme[1].size;
    IndexType steps = t.total_size / step_size;
    for (IndexType step = 0; step < steps; step++)
    {
      for (IndexType i = 0; i < t.scheme[1].size; i++)
        for (IndexType j = 0; j < t.scheme[0].size; j++)
          out.get(step * step_size + j * t.scheme[1].size + i) = t.get(step * step_size + i * t.scheme[0].size + j);
    }
  }

  template <typename T, int Dim>
  static inline Tensor<T, Dim> transpose(const Tensor<T, Dim> &t)
  {
    static_assert(Dim >= 2);
    typename Tensor<T, Dim>::Scheme res_scheme;
    res_scheme[0] = {t.scheme[1].size, 1};
    res_scheme[1] = {t.scheme[0].size, t.scheme[1].size};
    for (int i = 2; i < Dim; i++)
      res_scheme[i] = t.scheme[i];

    Tensor<T, Dim> res(nullptr, res_scheme);
    transpose(t, res);

    return res;
  }

  // matrix multiplication
  // if Dim=2 it simply performs matrix multiplication
  // if Dim>2 tensor is treated as a (Dim-2)-dimentional array of matrices (MxN), and
  // each of them is multiplied by m (NxK)
  template <typename T, int Dim>
  static inline void mat_mul(const Tensor<T, Dim> &t, const Tensor<T, 2> &m, Tensor<T, Dim> &out)
  {
    static_assert(Dim >= 2);
    assert(out.data != t.data);
    if (t.scheme[0].size != m.scheme[1].size)
    {
      printf("MMul: Tensors have incompatible dimensions: ");
      print_scheme<T, Dim>(t.scheme);
      print_scheme<T, 2>(m.scheme);
      printf("\n");
      return;
    }

    IndexType M = t.scheme[1].size;
    IndexType N = m.scheme[1].size;
    IndexType K = m.scheme[0].size;
    IndexType steps = t.total_size / (M * N); // how many matrices we have

    const Tensor<T, 2> m_tr = transpose(m);

    fill(out, 0.0f);
    for (IndexType step = 0; step < steps; step++)
    {
      for (IndexType i = 0; i < M; i++)
        for (IndexType j = 0; j < K; j++)
          for (IndexType k = 0; k < N; k++)
            out.get(step * (M * K) + i * K + j) += t.get(step * (M * N) + i * N + k) * m_tr.get(step * (N * K) + j * N + k);
    }
  }

  // a bunch of functions for convenience
  template <typename T>
  static inline Tensor<T, 1> get_vertex(int size)
  {
    return Tensor<T, 1>(std::array<uint32_t, 1>{(IndexType)size});
  }
  template <typename T>
  static inline Tensor<T, 1> get_vertex(const std::vector<float> &values, int size)
  {
    return Tensor<T, 1>(values, std::array<uint32_t, 1>{(IndexType)size});
  }

  template <typename T>
  static inline Tensor<T, 2> get_matrix(int rows, int columns)
  {
    return Tensor<T, 2>(std::array<uint32_t, 2>{(IndexType)rows, (IndexType)columns});
  }
  template <typename T>
  static inline Tensor<T, 2> get_matrix(const std::vector<float> &values, int rows, int columns)
  {
    return Tensor<T, 2>(values, std::array<uint32_t, 2>{(IndexType)rows, (IndexType)columns});
  }

  template <typename T>
  static inline Tensor<T, 3> get_3_tensor(int rows, int columns, int layers)
  {
    return Tensor<T, 3>(std::array<uint32_t, 3>{(IndexType)rows, (IndexType)columns, (IndexType)layers});
  }
  template <typename T>
  static inline Tensor<T, 3> get_3_tensor(const std::vector<float> &values, int rows, int columns, int layers)
  {
    return Tensor<T, 3>(values, std::array<uint32_t, 3>{(IndexType)rows, (IndexType)columns, (IndexType)layers});
  }

  template <typename T>
  static inline Tensor<T, 4> get_4_tensor(int rows, int columns, int layers, int groups)
  {
    return Tensor<T, 4>(std::array<uint32_t, 4>{(IndexType)rows, (IndexType)columns, (IndexType)layers, (IndexType)groups});
  }
  template <typename T>
  static inline Tensor<T, 4> get_4_tensor(const std::vector<float> &values, int rows, int columns, int layers, int groups)
  {
    return Tensor<T, 4>(values, std::array<uint32_t, 4>{(IndexType)rows, (IndexType)columns, (IndexType)layers, (IndexType)groups});
  }
};

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
  */
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
}