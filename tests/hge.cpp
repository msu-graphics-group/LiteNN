#include <vector>
#include <cstdio>
#include <LiteMath.h>
#include <hashgrid.h>

int main() {
    nn::HashGridEncoding hge(3, 10, 3, 4, 64, true);

    std::vector<float> features(3 * 3);
    hge.forward({{0.5f, 0.5f, 0.5f}}, 1, features);

    printf("Features:\n");
    for (uint32_t i = 0; i < features.size(); ++i) {
        printf("%f ", features[i]);
    }
    printf("\n");

    return 0;
}