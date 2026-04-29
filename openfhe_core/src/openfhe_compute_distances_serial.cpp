#include "io_utils.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

using namespace lbcrypto;
using namespace openfhe_migration;

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string centroidsFile = GetArg(args, "--centroids-file");
    const std::string encryptedQueryPath = GetArg(args, "--encrypted-query");
    const std::string encryptedNormPath = GetArg(args, "--encrypted-norm");
    const std::string outputDir = GetArg(args, "--output-dir");
    const int batchSize = std::max(1, std::stoi(GetArg(args, "--batch-size", "64")));
    if (contextDir.empty() || centroidsFile.empty() || encryptedQueryPath.empty() ||
        encryptedNormPath.empty() || outputDir.empty()) {
      throw std::runtime_error(
          "--context-dir, --centroids-file, --encrypted-query, --encrypted-norm, and --output-dir are required");
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
    const int dim = static_cast<int>(centroids.front().size());
    if (dim <= 0) {
      throw std::runtime_error("Centroid dimension is invalid");
    }

    auto wallStart = std::chrono::high_resolution_clock::now();
    const int nCentroids = static_cast<int>(centroids.size());
    std::cout << "Computing distances serially for " << nCentroids
              << " centroids with batch-size=" << batchSize << "\n";

    for (int i = 0; i < nCentroids; ++i) {
      const std::vector<double>& centroid = centroids[static_cast<size_t>(i)];
      if (static_cast<int>(centroid.size()) != dim) {
        throw std::runtime_error("Centroid dimension mismatch in centroids file");
      }

      Plaintext centroidPlain = cc->MakeCKKSPackedPlaintext(centroid);
      Ciphertext<DCRTPoly> product = cc->EvalMult(encQuery, centroidPlain);
      Ciphertext<DCRTPoly> dot = cc->EvalSum(product, dim);
      Ciphertext<DCRTPoly> twoDot = cc->EvalAdd(dot, dot);

      const double centroidNorm =
          std::inner_product(centroid.begin(), centroid.end(), centroid.begin(), 0.0);
      Plaintext centroidNormPlain =
          cc->MakeCKKSPackedPlaintext(std::vector<double>{centroidNorm});
      Ciphertext<DCRTPoly> sumNorms = cc->EvalAdd(encNorm, centroidNormPlain);
      Ciphertext<DCRTPoly> distance = cc->EvalSub(sumNorms, twoDot);

      const std::string outFile =
          outputDir + "/encrypted_distance_" +
          (i < 10 ? "000" : i < 100 ? "00" : i < 1000 ? "0" : "") +
          std::to_string(i) + ".bin";
      SaveCiphertext(outFile, distance);
    }

    auto wallEnd = std::chrono::high_resolution_clock::now();
    const double elapsedSec =
        std::chrono::duration<double>(wallEnd - wallStart).count();

    WriteDistanceMetadata(
        outputDir + "/distances_metadata.json",
        nCentroids,
        dim);
    std::cout << "Computed " << nCentroids << " encrypted distances serially in "
              << elapsedSec << " s"
              << " (" << (nCentroids / elapsedSec) << " centroids/s).\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_compute_distances_serial error: " << ex.what() << "\n";
    return 1;
  }
}
