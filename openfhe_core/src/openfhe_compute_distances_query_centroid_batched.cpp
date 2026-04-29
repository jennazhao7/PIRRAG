#include "io_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lbcrypto;
using namespace openfhe_migration;

namespace {

std::string FormatIndex(int value) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << value;
  return oss.str();
}

int ParseRequiredInt(const std::string& jsonPath, const std::string& key) {
  std::ifstream in(jsonPath);
  if (!in) {
    throw std::runtime_error("Failed to open metadata file: " + jsonPath);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const std::regex rx("\"" + key + "\"\\s*:\\s*(\\d+)");
  std::smatch match;
  if (!std::regex_search(text, match, rx)) {
    throw std::runtime_error("Key not found in metadata: " + key);
  }
  return std::stoi(match[1].str());
}

void WriteMetadata(
    const std::string& path,
    int nQueries,
    int nCentroids,
    int queryDim,
    int paddedDim,
    int queriesPerBatch,
    int centroidsPerBatch,
    int nQueryBatches,
    int nCentroidBatches) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write metadata file: " + path);
  }
  out << "{\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"format_version\": \"openfhe_query_centroid_batch_distances_v1\",\n";
  out << "  \"n_queries\": " << nQueries << ",\n";
  out << "  \"n_centroids\": " << nCentroids << ",\n";
  out << "  \"query_dim\": " << queryDim << ",\n";
  out << "  \"padded_dim\": " << paddedDim << ",\n";
  out << "  \"queries_per_batch\": " << queriesPerBatch << ",\n";
  out << "  \"centroids_per_batch\": " << centroidsPerBatch << ",\n";
  out << "  \"n_query_batches\": " << nQueryBatches << ",\n";
  out << "  \"n_centroid_batches\": " << nCentroidBatches << "\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string centroidsFile = GetArg(args, "--centroids-file");
    const std::string encryptedQueriesDir = GetArg(args, "--encrypted-queries-dir");
    const std::string outputDir = GetArg(args, "--output-dir");
    const int batchSize = std::max(1, std::stoi(GetArg(args, "--batch-size", "1")));
    const int numThreads = std::stoi(GetArg(args, "--num-threads", "0"));
    if (contextDir.empty() || centroidsFile.empty() || encryptedQueriesDir.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --centroids-file, --encrypted-queries-dir, and --output-dir are required");
    }

    std::filesystem::create_directories(outputDir);

    const std::string queryMetadataPath =
        encryptedQueriesDir + "/query_centroid_batch_metadata.json";
    const int nQueries = ParseRequiredInt(queryMetadataPath, "n_queries");
    const int nQueryBatches = ParseRequiredInt(queryMetadataPath, "n_query_batches");
    const int queriesPerBatch = ParseRequiredInt(queryMetadataPath, "queries_per_batch");
    const int centroidsPerBatch = ParseRequiredInt(queryMetadataPath, "centroids_per_batch");
    const int queryDim = ParseRequiredInt(queryMetadataPath, "query_dim");
    const int paddedDim = ParseRequiredInt(queryMetadataPath, "padded_dim");
    const int lanes = queriesPerBatch * centroidsPerBatch;
    const int slotsUsed = paddedDim * lanes;

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    LoadEvalKeys(paths);

    const auto centroids = ReadMatrixFile(centroidsFile);
    if (centroids.empty()) {
      throw std::runtime_error("No centroids found");
    }
    const int nCentroids = static_cast<int>(centroids.size());
    const int centroidDim = static_cast<int>(centroids.front().size());
    if (centroidDim != queryDim || centroidDim > paddedDim) {
      throw std::runtime_error("Centroid dimension must match query_dim and be <= padded_dim");
    }
    for (const auto& centroid : centroids) {
      if (static_cast<int>(centroid.size()) != centroidDim) {
        throw std::runtime_error("Centroid dimension mismatch in centroids file");
      }
    }
    if (numThreads > 0) {
      omp_set_num_threads(numThreads);
    }

    const int nCentroidBatches =
        (nCentroids + centroidsPerBatch - 1) / centroidsPerBatch;
    std::cout << "Computing query+centroid batched distances for " << nQueries
              << " queries and " << nCentroids << " centroids"
              << " (queries-per-batch=" << queriesPerBatch
              << ", centroids-per-batch=" << centroidsPerBatch
              << ", padded-dim=" << paddedDim
              << ", threads=" << (numThreads > 0 ? numThreads : omp_get_max_threads())
              << ")\n";

    auto wallStart = std::chrono::high_resolution_clock::now();
    for (int qb = 0; qb < nQueryBatches; ++qb) {
      const std::string queryPath =
          encryptedQueriesDir + "/encrypted_query_qbatch_" + FormatIndex(qb) + ".bin";
      const std::string normPath =
          encryptedQueriesDir + "/encrypted_norm_qbatch_" + FormatIndex(qb) + ".bin";
      Ciphertext<DCRTPoly> encQuery = LoadCiphertext(queryPath);
      Ciphertext<DCRTPoly> encNorm = LoadCiphertext(normPath);
      CryptoContext<DCRTPoly> cc = encQuery->GetCryptoContext();

      std::atomic<bool> failed(false);
      std::string failureMessage;
      #pragma omp parallel for schedule(dynamic, batchSize)
      for (int cb = 0; cb < nCentroidBatches; ++cb) {
        if (failed.load()) {
          continue;
        }
        try {
          const int centroidStart = cb * centroidsPerBatch;
          const int centroidsInBatch =
              std::min(centroidsPerBatch, nCentroids - centroidStart);
          std::vector<double> centroidPacked(static_cast<size_t>(slotsUsed), 0.0);
          std::vector<double> centroidNormPacked(static_cast<size_t>(slotsUsed), 0.0);

          for (int c = 0; c < centroidsInBatch; ++c) {
            const auto& centroid = centroids[static_cast<size_t>(centroidStart + c)];
            const double centroidNorm =
                std::inner_product(centroid.begin(), centroid.end(), centroid.begin(), 0.0);
            for (int q = 0; q < queriesPerBatch; ++q) {
              centroidNormPacked[static_cast<size_t>(q * centroidsPerBatch + c)] =
                  centroidNorm;
            }
            for (int d = 0; d < centroidDim; ++d) {
              for (int q = 0; q < queriesPerBatch; ++q) {
                const int slot = d * lanes + q * centroidsPerBatch + c;
                centroidPacked[static_cast<size_t>(slot)] =
                    centroid[static_cast<size_t>(d)];
              }
            }
          }

          Plaintext centroidPlain = cc->MakeCKKSPackedPlaintext(centroidPacked);
          Ciphertext<DCRTPoly> product = cc->EvalMult(encQuery, centroidPlain);
          Ciphertext<DCRTPoly> dot = product;
          for (int step = 1; step < paddedDim; step <<= 1) {
            Ciphertext<DCRTPoly> rotated = cc->EvalAtIndex(dot, step * lanes);
            dot = cc->EvalAdd(dot, rotated);
          }

          Ciphertext<DCRTPoly> twoDot = cc->EvalAdd(dot, dot);
          Plaintext centroidNormPlain = cc->MakeCKKSPackedPlaintext(centroidNormPacked);
          Ciphertext<DCRTPoly> sumNorms = cc->EvalAdd(encNorm, centroidNormPlain);
          Ciphertext<DCRTPoly> distance = cc->EvalSub(sumNorms, twoDot);

          SaveCiphertext(
              outputDir + "/encrypted_distance_qbatch_" + FormatIndex(qb) +
                  "_cbatch_" + FormatIndex(cb) + ".bin",
              distance);
        } catch (const std::exception& ex) {
          failed.store(true);
          #pragma omp critical
          {
            failureMessage = ex.what();
          }
        }
      }
      if (failed.load()) {
        throw std::runtime_error(failureMessage);
      }
    }

    const double elapsedSec =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wallStart)
            .count();
    WriteMetadata(
        outputDir + "/distances_metadata.json",
        nQueries,
        nCentroids,
        queryDim,
        paddedDim,
        queriesPerBatch,
        centroidsPerBatch,
        nQueryBatches,
        nCentroidBatches);
    std::cout << "Computed " << (static_cast<long long>(nQueries) * nCentroids)
              << " query-centroid distances in " << elapsedSec << " s.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_compute_distances_query_centroid_batched error: "
              << ex.what() << "\n";
    return 1;
  }
}
