#include "io_utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace lbcrypto;
using namespace openfhe_migration;

namespace {

std::string FormatIndex(int value) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << value;
  return oss.str();
}

void WriteMetadata(
    const std::string& path,
    int nQueries,
    int nQueryBatches,
    int queriesPerBatch,
    int centroidsPerBatch,
    int queryDim,
    int paddedDim,
    int slotsUsed) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write metadata file: " + path);
  }
  out << "{\n";
  out << "  \"format_version\": \"openfhe_query_centroid_batch_queries_v1\",\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"n_queries\": " << nQueries << ",\n";
  out << "  \"n_query_batches\": " << nQueryBatches << ",\n";
  out << "  \"queries_per_batch\": " << queriesPerBatch << ",\n";
  out << "  \"centroids_per_batch\": " << centroidsPerBatch << ",\n";
  out << "  \"query_dim\": " << queryDim << ",\n";
  out << "  \"padded_dim\": " << paddedDim << ",\n";
  out << "  \"slots_used\": " << slotsUsed << "\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string inputMatrixPath = GetArg(args, "--input-matrix");
    const std::string outputDir = GetArg(args, "--output-dir");
    const int queriesPerBatch = std::stoi(GetArg(args, "--queries-per-batch", "2"));
    const int centroidsPerBatch = std::stoi(GetArg(args, "--centroids-per-batch", "4"));
    const int paddedDim = std::stoi(GetArg(args, "--padded-dim", "1024"));
    if (contextDir.empty() || inputMatrixPath.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --input-matrix, and --output-dir are required");
    }
    if (queriesPerBatch <= 0 || centroidsPerBatch <= 0 || paddedDim <= 0) {
      throw std::runtime_error(
          "--queries-per-batch, --centroids-per-batch, and --padded-dim must be positive");
    }

    std::filesystem::create_directories(outputDir);

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    PublicKey<DCRTPoly> pk = LoadPublicKey(paths.publicKeyFile);
    CryptoContext<DCRTPoly> cc = pk->GetCryptoContext();
    LoadEvalKeys(paths);

    const auto queries = ReadMatrixFile(inputMatrixPath);
    if (queries.empty()) {
      throw std::runtime_error("No queries found in input matrix");
    }
    const int nQueries = static_cast<int>(queries.size());
    const int queryDim = static_cast<int>(queries.front().size());
    if (queryDim <= 0 || queryDim > paddedDim) {
      throw std::runtime_error("Query dimension must be > 0 and <= --padded-dim");
    }
    for (size_t i = 0; i < queries.size(); ++i) {
      if (static_cast<int>(queries[i].size()) != queryDim) {
        throw std::runtime_error("All query vectors must have identical dimensions");
      }
    }

    const int lanes = queriesPerBatch * centroidsPerBatch;
    const int slotsUsed = paddedDim * lanes;
    const int nQueryBatches = (nQueries + queriesPerBatch - 1) / queriesPerBatch;

    for (int qb = 0; qb < nQueryBatches; ++qb) {
      const int queryStart = qb * queriesPerBatch;
      const int queriesInBatch = std::min(queriesPerBatch, nQueries - queryStart);
      std::vector<double> packedQuery(static_cast<size_t>(slotsUsed), 0.0);
      std::vector<double> packedNorm(static_cast<size_t>(slotsUsed), 0.0);

      for (int q = 0; q < queriesInBatch; ++q) {
        const auto& query = queries[static_cast<size_t>(queryStart + q)];
        const double normSquared =
            std::inner_product(query.begin(), query.end(), query.begin(), 0.0);
        for (int c = 0; c < centroidsPerBatch; ++c) {
          packedNorm[static_cast<size_t>(q * centroidsPerBatch + c)] = normSquared;
        }
        for (int d = 0; d < queryDim; ++d) {
          for (int c = 0; c < centroidsPerBatch; ++c) {
            const int slot = d * lanes + q * centroidsPerBatch + c;
            packedQuery[static_cast<size_t>(slot)] = query[static_cast<size_t>(d)];
          }
        }
      }

      Plaintext queryPlain = cc->MakeCKKSPackedPlaintext(packedQuery);
      Plaintext normPlain = cc->MakeCKKSPackedPlaintext(packedNorm);
      SaveCiphertext(
          outputDir + "/encrypted_query_qbatch_" + FormatIndex(qb) + ".bin",
          cc->Encrypt(pk, queryPlain));
      SaveCiphertext(
          outputDir + "/encrypted_norm_qbatch_" + FormatIndex(qb) + ".bin",
          cc->Encrypt(pk, normPlain));
    }

    WriteMetadata(
        outputDir + "/query_centroid_batch_metadata.json",
        nQueries,
        nQueryBatches,
        queriesPerBatch,
        centroidsPerBatch,
        queryDim,
        paddedDim,
        slotsUsed);
    std::cout << "Encrypted " << nQueries << " queries into " << nQueryBatches
              << " query batches for query+centroid batching.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_encrypt_queries_centroid_batched error: " << ex.what() << "\n";
    return 1;
  }
}
