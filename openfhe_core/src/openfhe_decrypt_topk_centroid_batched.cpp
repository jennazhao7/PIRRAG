#include "io_utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    const std::string contextDir = GetArg(args, "--context-dir");
    const std::string encryptedDistancesDir = GetArg(args, "--encrypted-distances-dir");
    const std::string outputJson = GetArg(args, "--output-json");
    const int topK = std::stoi(GetArg(args, "--top-k", "100"));
    if (contextDir.empty() || encryptedDistancesDir.empty() || outputJson.empty()) {
      throw std::runtime_error(
          "--context-dir, --encrypted-distances-dir, --output-json are required");
    }

    const ContextBundlePaths paths = ResolveContextPaths(contextDir);
    PrivateKey<DCRTPoly> sk = LoadSecretKey(paths.secretKeyFile);
    CryptoContext<DCRTPoly> cc = sk->GetCryptoContext();
    LoadEvalKeys(paths);

    const std::string metadataPath = encryptedDistancesDir + "/distances_metadata.json";
    const int nCentroids = ParseRequiredInt(metadataPath, "n_centroids");
    const int centroidsPerCiphertext =
        ParseRequiredInt(metadataPath, "centroids_per_ciphertext");
    const int nBatches = ParseRequiredInt(metadataPath, "n_batches");

    std::vector<std::pair<double, int>> scored;
    scored.reserve(nCentroids);
    for (int batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
      const std::string inFile =
          encryptedDistancesDir + "/encrypted_distance_batch_" +
          FormatIndex(batchIdx) + ".bin";
      if (!std::filesystem::exists(inFile)) {
        continue;
      }
      Ciphertext<DCRTPoly> ct = LoadCiphertext(inFile);
      Plaintext pt;
      cc->Decrypt(sk, ct, &pt);
      const auto values = pt->GetRealPackedValue();
      for (int b = 0; b < centroidsPerCiphertext; ++b) {
        const int centroidIdx = batchIdx * centroidsPerCiphertext + b;
        if (centroidIdx >= nCentroids ||
            static_cast<size_t>(b) >= values.size()) {
          break;
        }
        scored.emplace_back(values[static_cast<size_t>(b)], centroidIdx);
      }
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    const int keep = std::min(topK, static_cast<int>(scored.size()));
    std::vector<double> topDistances;
    std::vector<int> topIndices;
    topDistances.reserve(keep);
    topIndices.reserve(keep);
    for (int i = 0; i < keep; ++i) {
      topDistances.push_back(scored[static_cast<size_t>(i)].first);
      topIndices.push_back(scored[static_cast<size_t>(i)].second);
    }
    WriteJsonTopK(outputJson, topDistances, topIndices);
    std::cout << "Decrypted " << scored.size() << " packed centroid distances and wrote top "
              << keep << " results.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_decrypt_topk_centroid_batched error: " << ex.what() << "\n";
    return 1;
  }
}
