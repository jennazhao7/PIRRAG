#include "io_utils.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <utility>

using namespace lbcrypto;
using namespace openfhe_migration;

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

    const int nDistances = ParseDistanceCount(
        encryptedDistancesDir + "/distances_metadata.json");
    std::vector<std::pair<double, int>> scored;
    scored.reserve(nDistances);
    for (int i = 0; i < nDistances; ++i) {
      const std::string inFile =
          encryptedDistancesDir + "/encrypted_distance_" +
          (i < 10 ? "000" : i < 100 ? "00" : i < 1000 ? "0" : "") +
          std::to_string(i) + ".bin";
      if (!std::filesystem::exists(inFile)) {
        continue;
      }
      Ciphertext<DCRTPoly> ct = LoadCiphertext(inFile);
      Plaintext pt;
      cc->Decrypt(sk, ct, &pt);
      const auto values = pt->GetRealPackedValue();
      if (values.empty()) {
        continue;
      }
      scored.emplace_back(values[0], i);
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
      topDistances.push_back(scored[i].first);
      topIndices.push_back(scored[i].second);
    }
    WriteJsonTopK(outputJson, topDistances, topIndices);
    std::cout << "Decrypted " << scored.size() << " distances and wrote top " << keep
              << " results.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_decrypt_topk error: " << ex.what() << "\n";
    return 1;
  }
}

