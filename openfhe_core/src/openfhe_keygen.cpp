#include "io_utils.h"

#include <filesystem>
#include <iostream>

using namespace lbcrypto;
using namespace openfhe_migration;

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    if (contextDir.empty()) {
      throw std::runtime_error("--context-dir is required");
    }

    const uint32_t polyModulusDegree = static_cast<uint32_t>(
        std::stoul(GetArg(args, "--poly-modulus-degree", "8192")));
    const std::string coeffCsv = GetArg(args, "--coeff-mod-bit-sizes", "60,40,40,60");
    const auto coeffSizes = ParseUintCsv(coeffCsv);
    if (coeffSizes.empty()) {
      throw std::runtime_error("No coeff modulus sizes provided");
    }

    std::filesystem::create_directories(contextDir);
    const ContextBundlePaths paths = ResolveContextPaths(contextDir);

    if (std::filesystem::exists(paths.contextFile) &&
        std::filesystem::exists(paths.publicKeyFile) &&
        std::filesystem::exists(paths.secretKeyFile) &&
        std::filesystem::exists(paths.evalMultKeyFile) &&
        std::filesystem::exists(paths.evalSumKeyFile)) {
      std::cout << "Context and keys already exist, skipping key generation.\n";
      return 0;
    }

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(coeffSizes.size() > 2 ? coeffSizes.size() - 2 : 2);
    parameters.SetScalingModSize(coeffSizes.size() > 1 ? coeffSizes[1] : 40);
    parameters.SetBatchSize(polyModulusDegree / 2);
    parameters.SetRingDim(polyModulusDegree);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    if (!keyPair.good()) {
      throw std::runtime_error("Key generation failed");
    }
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    SaveContextAndKeys(paths, cc, keyPair.publicKey, keyPair.secretKey);
    std::cout << "Generated OpenFHE context and keys in " << contextDir << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_keygen error: " << ex.what() << "\n";
    return 1;
  }
}

