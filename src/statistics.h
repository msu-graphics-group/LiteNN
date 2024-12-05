#ifndef INCLUDE_LITENN_STATISTICS_H_
#define INCLUDE_LITENN_STATISTICS_H_

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
    struct TrainStatistics
    {
        float avg_loss;
    };
}

#endif