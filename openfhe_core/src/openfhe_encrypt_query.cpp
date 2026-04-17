#include "io_utils.h"

#include <filesystem>
#include <iostream>
#include <numeric>

using namespace lbcrypto;
using namespace openfhe_migration;

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string inputVectorPath = GetArg(args, "--input-vector");
    const std::string outputDir = GetArg(args, "--output-dir");
    if (contextDir.empty() || inputVectorPath.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --input-vector, and --output-dir are required");
    }

    std::filesystem::create_directories(outputDir);

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    PublicKey<DCRTPoly> pk = LoadPublicKey(paths.publicKeyFile);
    CryptoContext<DCRTPoly> cc = pk->GetCryptoContext();
    LoadEvalKeys(paths);

    const std::vector<double> query = ReadVectorFile(inputVectorPath);
    Plaintext queryPlain = cc->MakeCKKSPackedPlaintext(query);
    Ciphertext<DCRTPoly> encQuery = cc->Encrypt(pk, queryPlain);

    const double normSquared =
        std::inner_product(query.begin(), query.end(), query.begin(), 0.0);
    Plaintext normPlain = cc->MakeCKKSPackedPlaintext(std::vector<double>{normSquared});
    Ciphertext<DCRTPoly> encNorm = cc->Encrypt(pk, normPlain);

    SaveCiphertext(outputDir + "/encrypted_query.bin", encQuery);
    SaveCiphertext(outputDir + "/encrypted_norm_squared.bin", encNorm);

    // Provide public artifacts in the same directory for server convenience.
    lbcrypto::Serial::SerializeToFile(
        outputDir + "/context.bin", cc, lbcrypto::SerType::BINARY);
    lbcrypto::Serial::SerializeToFile(
        outputDir + "/public_key.bin", pk, lbcrypto::SerType::BINARY);

    std::cout << "Encrypted query and norm to " << outputDir << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_encrypt_query error: " << ex.what() << "\n";
    return 1;
  }
}

