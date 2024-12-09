#include <cstdint>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>


uint32_t spatial_hash_3d(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t pi_1 = 1;
    uint32_t pi_2 = 2'654'435'761;
    uint32_t pi_3 = 805'459'861;

    uint32_t hash = pi_1 * x ^ pi_2 * y ^ pi_3 * z;
    return hash;
}


struct HashGridEncoding {
    const char *name = "HashGrid";

    uint32_t L, T, F, N_min, N_max;
    bool verbose;

    std::vector<uint32_t> N;
    std::vector<float> embeddings;

    HashGridEncoding(
        uint32_t L, // number of levels
        uint32_t T, // max entries per level
        uint32_t F, // number of features per entry
        uint32_t N_min, // coarsest resolution
        uint32_t N_max, // finest resolution
        bool verbose = false
    ) {
        this->L = L;
        this->T = T;
        this->F = F;
        this->N_min = N_min;
        this->N_max = N_max;
        this->verbose = verbose;

        // set hash grid resolutions
        N.resize(L);
        float b = std::exp(std::log(N_max / N_min) / (L - 1));
        float b_i = 1.0f;
        for (uint32_t i = 0; i < L; ++i, b_i *= b) {
            N[i] = N_min * b_i;
        }

        // set embeddings
        embeddings.resize(L * T * F);
        init_embeddings();

        if (verbose) {
            printf("[%s] Multi-Resolution Hash Grid Encoding init\n", name);
            printf("[%s] L: %d\n", name, L);
            printf("[%s] T: %d\n", name,T);
            printf("[%s] F: %d\n", name,F);
            printf("[%s] Resolutions:\n", name);
            for (uint32_t i = 0; i < L; ++i) {
                printf("[%s] N[%d] = %d\n", name,i, N[i]);
            }
        }
    }

    void forward(float x, float y, float z, std::vector<float> &features) {
        // x, y, z are in [0, 1]

        for (uint32_t layer_i = 0; layer_i < L; ++layer_i) {
            uint32_t n = N[layer_i];

            uint32_t x_int = x * n;
            if (x_int == n) {
                x_int = n - 1;
            }

            uint32_t y_int = y * n;
            if (y_int == n) {
                y_int = n - 1;
            }

            uint32_t z_int = z * n;
            if (z_int == n) {
                z_int = n - 1;
            }

            uint32_t hash = spatial_hash_3d(x, y, z);

            float *emb_ptr = embeddings.data() + layer_i * T * F + hash * F;
            memcpy(features.data() + layer_i * F, emb_ptr, sizeof(float) * F);
        }
    }

    void init_embeddings() {
        // unit uniformly -10^-4 to 10^-4 as in original paper
        for (uint32_t i = 0; i < embeddings.size(); ++i) {
            embeddings[i] = (rand() % 20001 - 10000) / 1'000'000.0f;
        }
    }
};


int main() {
    HashGridEncoding hge(3, 10, 3, 4, 64, true);

    std::vector<float> features(3 * 3);
    hge.forward(0.5f, 0.5f, 0.5f, features);

    printf("Features:\n");
    for (uint32_t i = 0; i < features.size(); ++i) {
        printf("%f ", features[i]);
    }
    printf("\n");

    return 0;
}