#include "io_utils.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <numeric>
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

void WriteBatchedDistanceMetadata(
    const std::string& path,
    int nCentroids,
    int centroidDim,
    int nCiphertexts,
    int nQueries,
    int queriesPerCiphertext) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write metadata file: " + path);
  }
  out << "{\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"format_version\": \"openfhe_batch_v1\",\n";
  out << "  \"n_centroids\": " << nCentroids << ",\n";
  out << "  \"centroid_dim\": " << centroidDim << ",\n";
  out << "  \"n_ciphertexts\": " << nCiphertexts << ",\n";
  out << "  \"n_queries\": " << nQueries << ",\n";
  out << "  \"queries_per_ciphertext\": " << queriesPerCiphertext << "\n";
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
    const int scheduleBatchSize = std::max(1, std::stoi(GetArg(args, "--batch-size", "64")));
    const int numThreads = std::stoi(GetArg(args, "--num-threads", "0"));
    if (contextDir.empty() || centroidsFile.empty() || encryptedQueriesDir.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --centroids-file, --encrypted-queries-dir, and --output-dir are required");
    }
    std::filesystem::create_directories(outputDir);

    const std::string metadataPath = encryptedQueriesDir + "/queries_metadata.json";
    const int nQueries = ParseRequiredInt(metadataPath, "n_queries");
    const int nCiphertexts = ParseRequiredInt(metadataPath, "n_ciphertexts");
    const int queriesPerCiphertext = ParseRequiredInt(metadataPath, "queries_per_ciphertext");
    const int queryDim = ParseRequiredInt(metadataPath, "query_dim");
    const int slotsPerCiphertext = ParseRequiredInt(metadataPath, "slots_per_ciphertext");
    if (nQueries <= 0 || nCiphertexts <= 0 || queriesPerCiphertext <= 0 || queryDim <= 0 ||
        slotsPerCiphertext <= 0) {
      throw std::runtime_error("Invalid batch metadata values");
    }

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    LoadEvalKeys(paths);

    const auto centroids = ReadMatrixFile(centroidsFile);
    if (centroids.empty()) {
      throw std::runtime_error("No centroids found");
    }
    const int nCentroids = static_cast<int>(centroids.size());
    const int centroidDim = static_cast<int>(centroids.front().size());
    if (centroidDim <= 0) {
      throw std::runtime_error("Centroid dimension is invalid");
    }
    for (size_t i = 0; i < centroids.size(); ++i) {
      if (static_cast<int>(centroids[i].size()) != centroidDim) {
        throw std::runtime_error("Centroid dimension mismatch in centroids file");
      }
    }
    if (centroidDim < queryDim) {
      throw std::runtime_error("Centroid dimension must be >= query_dim from metadata");
    }
    if (numThreads > 0) {
      omp_set_num_threads(numThreads);
    }

    std::cout << "Computing batched distances for " << nQueries << " queries, " << nCentroids
              << " centroids with batch-size=" << scheduleBatchSize
              << ", threads=" << (numThreads > 0 ? numThreads : omp_get_max_threads()) << "\n";

    for (int ctIdx = 0; ctIdx < nCiphertexts; ++ctIdx) {
      const std::string queryPath =
          encryptedQueriesDir + "/encrypted_query_batch_" + FormatIndex(ctIdx) + ".bin";
      const std::string normPath =
          encryptedQueriesDir + "/encrypted_norm_batch_" + FormatIndex(ctIdx) + ".bin";
      if (!std::filesystem::exists(queryPath) || !std::filesystem::exists(normPath)) {
        throw std::runtime_error("Missing encrypted batch files for index " + std::to_string(ctIdx));
      }

      Ciphertext<DCRTPoly> encQuery = LoadCiphertext(queryPath);
      Ciphertext<DCRTPoly> encNorm = LoadCiphertext(normPath);
      CryptoContext<DCRTPoly> cc = encQuery->GetCryptoContext();
      const int queryOffset = ctIdx * queriesPerCiphertext;
      const int queriesInCt = std::min(queriesPerCiphertext, nQueries - queryOffset);
      if (queriesInCt <= 0) {
        continue;
      }

      std::cout << "Processing ciphertext batch " << (ctIdx + 1) << "/" << nCiphertexts
                << " (" << queriesInCt << " queries)\n";

      std::atomic<bool> failed(false);
      std::string failureMessage;
      #pragma omp parallel for schedule(dynamic, scheduleBatchSize)
      for (int centroidIdx = 0; centroidIdx < nCentroids; ++centroidIdx) {
        if (failed.load()) {
          continue;
        }
        try {
          const std::vector<double>& centroid = centroids[static_cast<size_t>(centroidIdx)];
          const double centroidNorm =
              std::inner_product(centroid.begin(), centroid.end(), centroid.begin(), 0.0);

          std::vector<double> centroidPacked(static_cast<size_t>(slotsPerCiphertext), 0.0);
          for (int q = 0; q < queriesInCt; ++q) {
            const int segmentStart = q * queryDim;
            std::copy_n(
                centroid.begin(),
                queryDim,
                centroidPacked.begin() + segmentStart);
          }

          Plaintext centroidPlain = cc->MakeCKKSPackedPlaintext(centroidPacked);
          Ciphertext<DCRTPoly> product = cc->EvalMult(encQuery, centroidPlain);

          for (int q = 0; q < queriesInCt; ++q) {
            const int segmentStart = q * queryDim;
            const int globalQueryIdx = queryOffset + q;

            std::vector<double> queryMask(static_cast<size_t>(slotsPerCiphertext), 0.0);
            std::fill_n(queryMask.begin() + segmentStart, queryDim, 1.0);
            Plaintext queryMaskPlain = cc->MakeCKKSPackedPlaintext(queryMask);
            Ciphertext<DCRTPoly> maskedProduct = cc->EvalMult(product, queryMaskPlain);
            Ciphertext<DCRTPoly> dot = cc->EvalSum(maskedProduct, queryDim);
            Ciphertext<DCRTPoly> twoDot = cc->EvalAdd(dot, dot);

            std::vector<double> normMask(static_cast<size_t>(slotsPerCiphertext), 0.0);
            normMask[static_cast<size_t>(segmentStart)] = 1.0;
            Plaintext normMaskPlain = cc->MakeCKKSPackedPlaintext(normMask);
            Ciphertext<DCRTPoly> maskedNorm = cc->EvalMult(encNorm, normMaskPlain);
            Ciphertext<DCRTPoly> queryNorm = cc->EvalSum(maskedNorm, slotsPerCiphertext);

            Plaintext centroidNormPlain =
                cc->MakeCKKSPackedPlaintext(std::vector<double>{centroidNorm});
            Ciphertext<DCRTPoly> sumNorms = cc->EvalAdd(queryNorm, centroidNormPlain);
            Ciphertext<DCRTPoly> distance = cc->EvalSub(sumNorms, twoDot);

            const std::string outFile = outputDir + "/distance_" + FormatIndex(centroidIdx) +
                                        "_query_" + FormatIndex(globalQueryIdx) + ".bin";
            SaveCiphertext(outFile, distance);
          }
        } catch (const std::exception& ex) {
          failed.store(true);
          #pragma omp critical
          {
            failureMessage = ex.what();
          }
        }
      }
      if (failed.load()) {
        throw std::runtime_error("Batched distance compute failed: " + failureMessage);
      }
    }

    WriteBatchedDistanceMetadata(
        outputDir + "/distances_metadata.json",
        nCentroids,
        centroidDim,
        nCiphertexts,
        nQueries,
        queriesPerCiphertext);
    std::cout << "Computed batched encrypted distances for " << nQueries << " queries over "
              << nCentroids << " centroids.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_compute_distances_batched error: " << ex.what() << "\n";
    return 1;
  }
}
