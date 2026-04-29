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
#include <stdexcept>
#include <sstream>
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

void WriteCentroidBatchDistanceMetadata(
    const std::string& path,
    int nCentroids,
    int centroidDim,
    int paddedDim,
    int centroidsPerCiphertext,
    int nBatches) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write metadata file: " + path);
  }
  out << "{\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"format_version\": \"openfhe_centroid_batch_distances_v1\",\n";
  out << "  \"n_centroids\": " << nCentroids << ",\n";
  out << "  \"n_distances\": " << nCentroids << ",\n";
  out << "  \"centroid_dim\": " << centroidDim << ",\n";
  out << "  \"padded_dim\": " << paddedDim << ",\n";
  out << "  \"centroids_per_ciphertext\": " << centroidsPerCiphertext << ",\n";
  out << "  \"n_batches\": " << nBatches << "\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string centroidsFile = GetArg(args, "--centroids-file");
    const std::string encryptedQueryPath = GetArg(args, "--encrypted-query");
    const std::string encryptedNormPath = GetArg(args, "--encrypted-norm");
    const std::string outputDir = GetArg(args, "--output-dir");
    const int centroidsPerCiphertext =
        std::stoi(GetArg(args, "--centroids-per-ciphertext", "8"));
    const int paddedDim = std::stoi(GetArg(args, "--padded-dim", "1024"));
    const int batchSize = std::max(1, std::stoi(GetArg(args, "--batch-size", "1")));
    const int numThreads = std::stoi(GetArg(args, "--num-threads", "0"));
    if (contextDir.empty() || centroidsFile.empty() || encryptedQueryPath.empty() ||
        encryptedNormPath.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --centroids-file, --encrypted-query, --encrypted-norm, and --output-dir are required");
    }
    if (centroidsPerCiphertext <= 0 || paddedDim <= 0) {
      throw std::runtime_error("--centroids-per-ciphertext and --padded-dim must be positive");
    }

    std::filesystem::create_directories(outputDir);

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    LoadEvalKeys(paths);
    Ciphertext<DCRTPoly> encQuery = LoadCiphertext(encryptedQueryPath);
    Ciphertext<DCRTPoly> encNorm = LoadCiphertext(encryptedNormPath);
    CryptoContext<DCRTPoly> cc = encQuery->GetCryptoContext();

    const auto centroids = ReadMatrixFile(centroidsFile);
    if (centroids.empty()) {
      throw std::runtime_error("No centroids found");
    }
    const int nCentroids = static_cast<int>(centroids.size());
    const int centroidDim = static_cast<int>(centroids.front().size());
    if (centroidDim <= 0 || centroidDim > paddedDim) {
      throw std::runtime_error("Centroid dimension must be > 0 and <= --padded-dim");
    }
    for (size_t i = 0; i < centroids.size(); ++i) {
      if (static_cast<int>(centroids[i].size()) != centroidDim) {
        throw std::runtime_error("Centroid dimension mismatch in centroids file");
      }
    }
    if (numThreads > 0) {
      omp_set_num_threads(numThreads);
    }

    const int slotsUsed = paddedDim * centroidsPerCiphertext;
    const int nBatches = (nCentroids + centroidsPerCiphertext - 1) / centroidsPerCiphertext;
    std::cout << "Computing centroid-batched distances for " << nCentroids
              << " centroids with centroids-per-ciphertext=" << centroidsPerCiphertext
              << ", padded-dim=" << paddedDim
              << ", threads=" << (numThreads > 0 ? numThreads : omp_get_max_threads())
              << "\n";

    auto wallStart = std::chrono::high_resolution_clock::now();
    std::atomic<bool> failed(false);
    std::string failureMessage;

    #pragma omp parallel for schedule(dynamic, batchSize)
    for (int batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
      if (failed.load()) {
        continue;
      }
      try {
        const int centroidStart = batchIdx * centroidsPerCiphertext;
        const int centroidsInBatch =
            std::min(centroidsPerCiphertext, nCentroids - centroidStart);

        std::vector<double> centroidPacked(static_cast<size_t>(slotsUsed), 0.0);
        std::vector<double> centroidNormPacked(static_cast<size_t>(slotsUsed), 0.0);

        for (int b = 0; b < centroidsInBatch; ++b) {
          const auto& centroid = centroids[static_cast<size_t>(centroidStart + b)];
          const double centroidNorm =
              std::inner_product(centroid.begin(), centroid.end(), centroid.begin(), 0.0);
          centroidNormPacked[static_cast<size_t>(b)] = centroidNorm;
          for (int d = 0; d < centroidDim; ++d) {
            centroidPacked[static_cast<size_t>(d * centroidsPerCiphertext + b)] =
                centroid[static_cast<size_t>(d)];
          }
        }

        Plaintext centroidPlain = cc->MakeCKKSPackedPlaintext(centroidPacked);
        Ciphertext<DCRTPoly> product = cc->EvalMult(encQuery, centroidPlain);
        Ciphertext<DCRTPoly> dot = product;

        for (int step = 1; step < paddedDim; step <<= 1) {
          Ciphertext<DCRTPoly> rotated =
              cc->EvalAtIndex(dot, step * centroidsPerCiphertext);
          dot = cc->EvalAdd(dot, rotated);
        }

        Ciphertext<DCRTPoly> twoDot = cc->EvalAdd(dot, dot);
        Plaintext centroidNormPlain = cc->MakeCKKSPackedPlaintext(centroidNormPacked);
        Ciphertext<DCRTPoly> sumNorms = cc->EvalAdd(encNorm, centroidNormPlain);
        Ciphertext<DCRTPoly> distance = cc->EvalSub(sumNorms, twoDot);

        SaveCiphertext(
            outputDir + "/encrypted_distance_batch_" + FormatIndex(batchIdx) + ".bin",
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

    const double elapsedSec =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - wallStart)
            .count();
    WriteCentroidBatchDistanceMetadata(
        outputDir + "/distances_metadata.json",
        nCentroids,
        centroidDim,
        paddedDim,
        centroidsPerCiphertext,
        nBatches);
    std::cout << "Computed " << nCentroids << " centroid distances in " << nBatches
              << " packed ciphertexts in " << elapsedSec << " s"
              << " (" << (nCentroids / elapsedSec) << " centroids/s).\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_compute_distances_centroid_batched error: " << ex.what() << "\n";
    return 1;
  }
}
