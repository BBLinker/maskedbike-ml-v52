// Deterministic BIKE v5.2 case-seed replay helper.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "api.h"
#include "defs.h"
#include "hash_wrapper.h"
#include "kem.h"
#include "FromNIST/rng.h"

extern uint8_t maskedbike_last_e0[R_SIZE];
extern uint8_t maskedbike_last_e1[R_SIZE];

namespace {
constexpr std::size_t kSeedBytes = 32;
constexpr std::size_t kEntropyBytes = 48;

bool parse_hex(uint8_t *output, std::size_t length, const std::string &text) {
    if (text.size() != 2 * length) return false;
    bool nonzero = false;
    for (std::size_t i = 0; i < length; ++i) {
        unsigned value = 0;
        if (std::sscanf(text.c_str() + 2 * i, "%2x", &value) != 1) return false;
        output[i] = static_cast<uint8_t>(value);
        nonzero |= output[i] != 0;
    }
    return nonzero;
}

void derive_entropy(uint8_t output[kEntropyBytes], const char *domain,
                    const uint8_t seed[kSeedBytes]) {
    uint8_t input[64] = {0};
    const std::size_t domain_length = std::strlen(domain);
    std::memcpy(input, domain, domain_length);
    std::memcpy(input + domain_length, seed, kSeedBytes);
    sha3_384(output, input, domain_length + kSeedBytes);
}

void print_hex(const uint8_t *values, std::size_t length) {
    static const char digits[] = "0123456789abcdef";
    for (std::size_t i = 0; i < length; ++i) {
        std::putchar(digits[values[i] >> 4]);
        std::putchar(digits[values[i] & 15]);
    }
}

void print_positions(const uint8_t *packed) {
    bool first = true;
    for (std::size_t bit = 0; bit < R_BITS; ++bit) {
        if ((packed[bit >> 3] >> (bit & 7u)) & 1u) {
            if (!first) std::putchar(',');
            std::printf("%zu", bit);
            first = false;
        }
    }
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s KEY_SEED_HEX\n", argv[0]);
        return 2;
    }
    uint8_t key_seed[kSeedBytes] = {0};
    if (!parse_hex(key_seed, sizeof(key_seed), argv[1])) return 2;
    uint8_t key_entropy[kEntropyBytes] = {0};
    derive_entropy(key_entropy, "maskedbike-bike-v5.2-key", key_seed);
    uint8_t public_key[CRYPTO_PUBLICKEYBYTES] = {0};
    uint8_t secret_key[CRYPTO_SECRETKEYBYTES] = {0};
    randombytes_init(key_entropy, nullptr, 256);
    if (crypto_kem_keypair(public_key, secret_key) != 0) return 3;

    std::string seed_text;
    while (std::getline(std::cin, seed_text)) {
        if (seed_text.empty()) continue;
        uint8_t case_seed[kSeedBytes] = {0};
        if (!parse_hex(case_seed, sizeof(case_seed), seed_text)) return 4;
        uint8_t case_entropy[kEntropyBytes] = {0};
        derive_entropy(case_entropy, "maskedbike-bike-v5.2-case", case_seed);
        uint8_t ciphertext[CRYPTO_CIPHERTEXTBYTES] = {0};
        uint8_t shared_secret[CRYPTO_BYTES] = {0};
        randombytes_init(case_entropy, nullptr, 256);
        if (crypto_kem_enc(ciphertext, shared_secret, public_key) != 0) return 5;
        std::cout << seed_text << '\t';
        print_hex(ciphertext, sizeof(ciphertext));
        std::putchar('\t'); print_positions(maskedbike_last_e0);
        std::putchar('\t'); print_positions(maskedbike_last_e1);
        std::putchar('\n'); std::fflush(stdout);
    }
    return 0;
}
