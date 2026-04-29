#include "io_utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace lbcrypto;
using namespace openfhe_migration;

namespace {

void WriteCentroidBatchQueryMetadata(
    const std::string& path,
    int queryDim,
    int paddedDim,
    int centroidsPerCiphertext,
    int slotsPerCiphertext) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write metadata file: " + path);
  }
  out << "{\n";
  out << "  \"format_version\": \"openfhe_centroid_batch_query_v1\",\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"query_dim\": " << queryDim << ",\n";
  out << "  \"padded_dim\": " << paddedDim << ",\n";
  out << "  \"centroids_per_ciphertext\": " << centroidsPerCiphertext << ",\n";
  out << "  \"slots_per_ciphertext\": " << slotsPerCiphertext << "\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string inputVectorPath = GetArg(args, "--input-vector");
    const std::string outputDir = GetArg(args, "--output-dir");
    const int centroidsPerCiphertext =
        std::stoi(GetArg(args, "--centroids-per-ciphertext", "8"));
    const int paddedDim = std::stoi(GetArg(args, "--padded-dim", "1024"));
    if (contextDir.empty() || inputVectorPath.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --input-vector, and --output-dir are required");
    }
    if (centroidsPerCiphertext <= 0 || paddedDim <= 0) {
      throw std::runtime_error("--centroids-per-ciphertext and --padded-dim must be positive");
    }

    std::filesystem::create_directories(outputDir);

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    PublicKey<DCRTPoly> pk = LoadPublicKey(paths.publicKeyFile);
    CryptoContext<DCRTPoly> cc = pk->GetCryptoContext();
    LoadEvalKeys(paths);

    const std::vector<double> query = ReadVectorFile(inputVectorPath);
    const int queryDim = static_cast<int>(query.size());
    if (queryDim > paddedDim) {
      throw std::runtime_error("Query dimension exceeds --padded-dim");
    }

    const int slotsPerCiphertext = paddedDim * centroidsPerCiphertext;
    std::vector<double> packedQuery(static_cast<size_t>(slotsPerCiphertext), 0.0);
    for (int d = 0; d < queryDim; ++d) {
      for (int b = 0; b < centroidsPerCiphertext; ++b) {
        packedQuery[static_cast<size_t>(d * centroidsPerCiphertext + b)] =
            query[static_cast<size_t>(d)];
      }
    }

    const double normSquared =
        std::inner_product(query.begin(), query.end(), query.begin(), 0.0);
    std::vector<double> packedNorm(static_cast<size_t>(slotsPerCiphertext), 0.0);
    std::fill_n(packedNorm.begin(), centroidsPerCiphertext, normSquared);

    Plaintext queryPlain = cc->MakeCKKSPackedPlaintext(packedQuery);
    Plaintext normPlain = cc->MakeCKKSPackedPlaintext(packedNorm);
    Ciphertext<DCRTPoly> encQuery = cc->Encrypt(pk, queryPlain);
    Ciphertext<DCRTPoly> encNorm = cc->Encrypt(pk, normPlain);

    SaveCiphertext(outputDir + "/encrypted_query_centroid_batched.bin", encQuery);
    SaveCiphertext(outputDir + "/encrypted_norm_centroid_batched.bin", encNorm);
    WriteCentroidBatchQueryMetadata(
        outputDir + "/centroid_batch_query_metadata.json",
        queryDim,
        paddedDim,
        centroidsPerCiphertext,
        slotsPerCiphertext);

    std::cout << "Encrypted replicated query for centroid batching to " << outputDir << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_encrypt_query_centroid_batched error: " << ex.what() << "\n";
    return 1;
  }
}
