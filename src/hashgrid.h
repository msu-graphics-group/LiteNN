#ifndef INCLUDE_LITENN_HASHGRID_H_
#define INCLUDE_LITENN_HASHGRID_H_
#include <cstdint>
#include <vector>
#include <LiteMath.h>


namespace nn 
{

    class HashGridEncoding
    {
    public:

        HashGridEncoding(
            uint32_t layers, // number of levels
            uint32_t table_size, // max entries per level
            uint32_t feature_dim, // number of features per entry
            uint32_t N_min, // coarsest resolution
            uint32_t N_max, // finest resolution
            bool verbose = false
        );

        void forward(const LiteMath::float3 &vec, std::vector<float> &out_features);

        void init_embeddings();
    private:
        static constexpr char *name = "HashGrid";

        uint32_t layers;             // number of levels
        uint32_t table_size;             // max entries per level
        uint32_t feature_dim;             // number of features per entry
        uint32_t N_min, N_max;  // coarsest/finest resolutions
        bool verbose;

        std::vector<uint32_t> N;
        std::vector<float> embeddings;
    };

}

#endif