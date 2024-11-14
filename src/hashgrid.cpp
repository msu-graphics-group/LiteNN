#include "hashgrid.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace nn {



    HashGridEncoding::HashGridEncoding(uint32_t layers, uint32_t table_size, uint32_t feature_dim, uint32_t N_min, uint32_t N_max, bool verbose)
        : layers{layers}, table_size{table_size}, feature_dim{feature_dim}, N_min{N_min}, N_max{N_max}, verbose{verbose},
          N{layers}, embeddings{layers * table_size * feature_dim}
    {
        // set hash grid resolutions
        float b = std::exp(std::log(N_max / N_min) / (layers - 1));
        float b_i = 1.0f;
        for (uint32_t i = 0; i < layers; ++i, b_i *= b) {
            N[i] = N_min * b_i;
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
                printf("[%s] N[%d] = %d\n", name,i, N[i]);
            }
        }
    }

    namespace {

        constexpr uint64_t PI_1 = 1;
        constexpr uint64_t PI_2 = 2'654'435'761;
        constexpr uint64_t PI_3 = 805'459'861;


        uint64_t spatial_hash_3d(uint32_t x, uint32_t y, uint32_t z) {
            return (PI_1 * x) ^ (PI_2 * y) ^ (PI_3 * z);
        }
    }


    void HashGridEncoding::forward(const LiteMath::float3 &vec, std::vector<float> &out_features) {

        for(uint32_t layer_i = 0; layer_i < layers; ++layer_i) {
            uint32_t n = N[layer_i];

            uint32_t x_int = vec.x * n;
            if(x_int == n) {
                x_int = n - 1;
            }

            uint32_t y_int = vec.y * n;
            if(y_int == n) {
                y_int = n - 1;
            }

            uint32_t z_int = vec.z * n;
            if(z_int == n) {
                z_int = n - 1;
            }

            uint64_t hash = spatial_hash_3d(x_int, y_int, z_int);
            uint32_t feature_idx = hash % table_size;

            float *emb_ptr = embeddings.data() + layer_i * table_size * feature_dim + feature_idx * feature_dim;
            std::copy(emb_ptr, emb_ptr + feature_dim, out_features.data() + layer_i * feature_dim);
        }
    }

    void HashGridEncoding::init_embeddings() {
        // unit uniformly -10^-4 to 10^-4 as in original paper
        for (uint32_t i = 0; i < embeddings.size(); ++i) {
            embeddings[i] = (rand() % 20001 - 10000) / 1'000'000.0f;
        }
    }

}