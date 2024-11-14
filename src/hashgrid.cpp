#include "hashgrid.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <algorithm>

namespace nn {



    HashGridEncoding::HashGridEncoding(uint32_t layers, uint32_t table_size, uint32_t feature_dim, uint32_t N_min, uint32_t N_max, bool verbose)
        : layers{layers}, table_size{table_size}, feature_dim{feature_dim}, N_min{N_min}, N_max{N_max}, verbose{verbose},
          N_sizes{layers}, embeddings{layers * table_size * feature_dim}
    {
        // set hash grid resolutions
        float b = std::exp(std::log(N_max / N_min) / (layers - 1));
        float b_i = 1.0f;
        for (uint32_t i = 0; i < layers; ++i, b_i *= b) {
            N_sizes[i] = N_min * b_i;
        }

        // set embeddings
        init_embeddings();

        if (verbose) {
            printf("[%s] Multi-Resolution Hash Grid Encoding init\n", name);
            printf("[%s] L: %d\n", name, layers);
            printf("[%s] T: %d\n", name, table_size);
            printf("[%s] F: %d\n", name, feature_dim);
            printf("[%s] Resolutions:\n", name);
            for (uint32_t i = 0; i < layers; ++i) {
                printf("[%s] N_sizes[%d] = %d\n", name,i, N_sizes[i]);
            }
        }
    }

    namespace {

        constexpr uint64_t PI_1 = 1;
        constexpr uint64_t PI_2 = 2'654'435'761;
        constexpr uint64_t PI_3 = 805'459'861;


        uint64_t spatial_hash_3d(uint32_t x, uint32_t y, uint32_t z)
        {
            return (PI_1 * x) ^ (PI_2 * y) ^ (PI_3 * z);
        }

        inline std::pair<uint32_t, uint32_t> get_i(float val, uint32_t n)
        {
            uint32_t v0 = floor(val * (n - 1));
            uint32_t v1 = ceil(val * (n - 1));
            if(v1 == 0) {
                v1 = 1;
            }
            else if(v0 == n - 1) {
                v0 = n - 2;
            }

            return {v0, v1};
        }
    }

    void HashGridEncoding::encode(const LiteMath::float3 &vec, float *out_feature, uint32_t layer_i, uint32_t *out_indices, float *out_weights)
    {
        uint32_t n = N_sizes[layer_i];
        uint32_t indices[8];
        float weights[8];

        std::vector<float> features(feature_dim * 8);

        const auto [x0, x1] = get_i(vec.x, n);
        const auto [y0, y1] = get_i(vec.y, n);
        const auto [z0, z1] = get_i(vec.z, n);

        const float wx = vec.x - x0;
        const float wy = vec.y - y0;
        const float wz = vec.z - z0;

        indices[0b000] = spatial_hash_3d(x0, y0, z0) % table_size;
        indices[0b001] = spatial_hash_3d(x0, y0, z1) % table_size;
        indices[0b010] = spatial_hash_3d(x0, y1, z0) % table_size;
        indices[0b011] = spatial_hash_3d(x0, y1, z1) % table_size;
        indices[0b100] = spatial_hash_3d(x1, y0, z0) % table_size;
        indices[0b101] = spatial_hash_3d(x1, y0, z1) % table_size;
        indices[0b110] = spatial_hash_3d(x1, y1, z0) % table_size;
        indices[0b111] = spatial_hash_3d(x1, y1, z1) % table_size;


        weights[0b000] = (1.0f - wx) * (1.0f - wy) * (1.0f - wz);
        weights[0b001] = (1.0f - wx) * (1.0f - wy) * wz;
        weights[0b010] = (1.0f - wx) * wy * (1.0f - wz);
        weights[0b011] = (1.0f - wx) * wy * wz;
        weights[0b100] = wx * (1.0f - wy) * (1.0f - wz);
        weights[0b101] = wx * (1.0f - wy) * wz;
        weights[0b110] = wx * wy * (1.0f - wz);
        weights[0b111] = wx * wy * wz;

        const uint32_t emb_layer_offset = layer_i * feature_dim * table_size;
        const float *emb_ptr = embeddings.data() + emb_layer_offset;

        std::fill_n(out_feature, feature_dim, 0.0f);

        for(int i = 0; i < 8; ++i) {

            for(int j = 0; j < feature_dim; ++j) {
                out_feature[j] += weights[i] * (emb_ptr[indices[i] + j]);
            }
        }

        if(out_weights) {
            std::copy(weights, weights + 8, out_weights);
        }
        if(out_indices) {
            std::copy(indices, indices + 8, out_indices);
        }

    }

    void HashGridEncoding::forward(const std::vector<LiteMath::float3> &in_data, int batch_size, std::vector<float> &out_features)
    {
        out_features.resize(batch_size * feature_dim * layers);

        last_weights.resize(batch_size * 8);
        last_indices.resize(batch_size * 8);

        #pragma omp parallel for
        for(int batch = 0; batch < batch_size; ++batch) {
            const uint32_t out_batch_offset = batch * feature_dim * layers;

            for(uint32_t layer_i = 0; layer_i < layers; ++layer_i) {
                const uint32_t layer_offset = layer_i * feature_dim;
                encode(in_data[batch], out_features.data() + out_batch_offset + layer_offset, layer_i, last_indices.data() + batch * 8, last_weights.data() + batch * 8);
            }
        }
    }

    void HashGridEncoding::init_embeddings() {
        // unit uniformly -10^-4 to 10^-4 as in original paper
        for (uint32_t i = 0; i < embeddings.size(); ++i) {
            embeddings[i] = (rand() % 20001 - 10000) / 1'000'000.0f;
        }
    }

}