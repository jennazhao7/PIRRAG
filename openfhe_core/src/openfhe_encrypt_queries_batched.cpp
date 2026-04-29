#include "io_utils.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

using namespace lbcrypto;
using namespace openfhe_migration;

namespace {

std::string FormatIndex(int value) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << value;
  return oss.str();
}

void WriteQueriesMetadata(
    const std::string& path,
    int nQueries,
    int nCiphertexts,
    int queriesPerCiphertext,
    int queryDim,
    int slotsPerCiphertext,
    int polyModulusDegree) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write metadata file: " + path);
  }
  out << "{\n";
  out << "  \"format_version\": \"openfhe_batch_v1\",\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"n_queries\": " << nQueries << ",\n";
  out << "  \"n_ciphertexts\": " << nCiphertexts << ",\n";
  out << "  \"queries_per_ciphertext\": " << queriesPerCiphertext << ",\n";
  out << "  \"query_dim\": " << queryDim << ",\n";
  out << "  \"slots_per_ciphertext\": " << slotsPerCiphertext << ",\n";
  out << "  \"poly_modulus_degree\": " << polyModulusDegree << "\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string inputMatrixPath = GetArg(args, "--input-matrix");
    const std::string outputDir = GetArg(args, "--output-dir");
    const int polyModulusDegree = std::stoi(GetArg(args, "--poly-modulus-degree", "8192"));
    const int queryDimArg = std::stoi(GetArg(args, "--query-dim", "0"));
    if (contextDir.empty() || inputMatrixPath.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --input-matrix, and --output-dir are required");
    }
    if (polyModulusDegree <= 0 || (polyModulusDegree % 2) != 0) {
      throw std::runtime_error("--poly-modulus-degree must be a positive even integer");
    }

    std::filesystem::create_directories(outputDir);

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    PublicKey<DCRTPoly> pk = LoadPublicKey(paths.publicKeyFile);
    CryptoContext<DCRTPoly> cc = pk->GetCryptoContext();
    LoadEvalKeys(paths);

    const std::vector<std::vector<double>> queries = ReadMatrixFile(inputMatrixPath);
    const int nQueries = static_cast<int>(queries.size());
    if (nQueries <= 0) {
      throw std::runtime_error("No queries found in input matrix");
    }
    const int queryDim =
        queryDimArg > 0 ? queryDimArg : static_cast<int>(queries.front().size());
    if (queryDim <= 0) {
      throw std::runtime_error("Query dimension must be > 0");
    }
    for (size_t i = 0; i < queries.size(); ++i) {
      if (static_cast<int>(queries[i].size()) != queryDim) {
        throw std::runtime_error("All query vectors must have identical dimensions");
      }
    }

    const int slotsPerCiphertext = polyModulusDegree / 2;
    const int queriesPerCiphertext = slotsPerCiphertext / queryDim;
    if (queriesPerCiphertext <= 0) {
      throw std::runtime_error(
          "No query fits in one ciphertext: increase --poly-modulus-degree or reduce --query-dim");
    }

    int ciphertextIdx = 0;
    for (int start = 0; start < nQueries; start += queriesPerCiphertext, ++ciphertextIdx) {
      const int end = std::min(start + queriesPerCiphertext, nQueries);
      const int queriesInCt = end - start;

      std::vector<double> packedQuery(slotsPerCiphertext, 0.0);
      std::vector<double> packedNorm(slotsPerCiphertext, 0.0);
      for (int q = 0; q < queriesInCt; ++q) {
        const int queryGlobalIdx = start + q;
        const int offset = q * queryDim;
        const std::vector<double>& query = queries[static_cast<size_t>(queryGlobalIdx)];
        std::copy(query.begin(), query.end(), packedQuery.begin() + offset);

        const double normSquared =
            std::inner_product(query.begin(), query.end(), query.begin(), 0.0);
        packedNorm[static_cast<size_t>(offset)] = normSquared;
      }

      Plaintext queryPlain = cc->MakeCKKSPackedPlaintext(packedQuery);
      Plaintext normPlain = cc->MakeCKKSPackedPlaintext(packedNorm);
      Ciphertext<DCRTPoly> encQuery = cc->Encrypt(pk, queryPlain);
      Ciphertext<DCRTPoly> encNorm = cc->Encrypt(pk, normPlain);

      SaveCiphertext(
          outputDir + "/encrypted_query_batch_" + FormatIndex(ciphertextIdx) + ".bin",
          encQuery);
      SaveCiphertext(
          outputDir + "/encrypted_norm_batch_" + FormatIndex(ciphertextIdx) + ".bin",
          encNorm);
    }

    WriteQueriesMetadata(
        outputDir + "/queries_metadata.json",
        nQueries,
        ciphertextIdx,
        queriesPerCiphertext,
        queryDim,
        slotsPerCiphertext,
        polyModulusDegree);
    std::cout << "Encrypted " << nQueries << " queries into " << ciphertextIdx
              << " ciphertext batches.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_encrypt_queries_batched error: " << ex.what() << "\n";
    return 1;
  }
}
