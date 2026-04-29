#include "io_utils.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>

using namespace lbcrypto;
using namespace openfhe_migration;

namespace {

std::vector<int32_t> ParseIntCsv(const std::string& csv) {
  std::vector<int32_t> out;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      out.push_back(static_cast<int32_t>(std::stoi(token)));
    }
  }
  return out;
}

}  // namespace

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
    const std::string securityLevel = GetArg(args, "--security-level", "");
    const auto rotationIndices =
        ParseIntCsv(GetArg(args, "--rotation-indices", ""));
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
      if (!rotationIndices.empty() &&
          (!std::filesystem::exists(paths.evalAutomorphismKeyFile) ||
           std::filesystem::file_size(paths.evalAutomorphismKeyFile) == 0)) {
        PrivateKey<DCRTPoly> sk = LoadSecretKey(paths.secretKeyFile);
        CryptoContext<DCRTPoly> cc = sk->GetCryptoContext();
        cc->EvalRotateKeyGen(sk, rotationIndices);
        SaveEvalAutomorphismKeys(paths.evalAutomorphismKeyFile, cc);
        std::cout << "Generated OpenFHE rotation keys in " << contextDir << "\n";
        return 0;
      }
      std::cout << "Context and keys already exist, skipping key generation.\n";
      return 0;
    }

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(coeffSizes.size() > 2 ? coeffSizes.size() - 2 : 2);
    parameters.SetScalingModSize(coeffSizes.size() > 1 ? coeffSizes[1] : 40);
    parameters.SetBatchSize(polyModulusDegree / 2);
    parameters.SetRingDim(polyModulusDegree);
    if (securityLevel == "none" || securityLevel == "notset" ||
        securityLevel == "HEStd_NotSet") {
      parameters.SetSecurityLevel(HEStd_NotSet);
    }

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
    if (!rotationIndices.empty()) {
      cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
    }

    SaveContextAndKeys(paths, cc, keyPair.publicKey, keyPair.secretKey);
    if (!rotationIndices.empty()) {
      SaveEvalAutomorphismKeys(paths.evalAutomorphismKeyFile, cc);
    }
    std::cout << "Generated OpenFHE context and keys in " << contextDir << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_keygen error: " << ex.what() << "\n";
    return 1;
  }
}

