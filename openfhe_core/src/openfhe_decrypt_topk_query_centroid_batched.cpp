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

void WriteMultiQueryTopK(
    const std::string& path,
    const std::vector<std::vector<std::pair<double, int>>>& perQueryTopK) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write output JSON: " + path);
  }
  out << "{\n";
  out << "  \"n_queries\": " << perQueryTopK.size() << ",\n";
  out << "  \"results\": [\n";
  for (size_t q = 0; q < perQueryTopK.size(); ++q) {
    out << "    {\n";
    out << "      \"query_index\": " << q << ",\n";
    out << "      \"distances\": [";
    for (size_t i = 0; i < perQueryTopK[q].size(); ++i) {
      out << perQueryTopK[q][i].first;
      if (i + 1 < perQueryTopK[q].size()) {
        out << ", ";
      }
    }
    out << "],\n";
    out << "      \"centroid_indices\": [";
    for (size_t i = 0; i < perQueryTopK[q].size(); ++i) {
      out << perQueryTopK[q][i].second;
      if (i + 1 < perQueryTopK[q].size()) {
        out << ", ";
      }
    }
    out << "]\n";
    out << "    }";
    if (q + 1 < perQueryTopK.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
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
    const int nQueries = ParseRequiredInt(metadataPath, "n_queries");
    const int nCentroids = ParseRequiredInt(metadataPath, "n_centroids");
    const int queriesPerBatch = ParseRequiredInt(metadataPath, "queries_per_batch");
    const int centroidsPerBatch = ParseRequiredInt(metadataPath, "centroids_per_batch");
    const int nQueryBatches = ParseRequiredInt(metadataPath, "n_query_batches");
    const int nCentroidBatches = ParseRequiredInt(metadataPath, "n_centroid_batches");

    std::vector<std::vector<std::pair<double, int>>> scored(
        static_cast<size_t>(nQueries));
    for (auto& row : scored) {
      row.reserve(static_cast<size_t>(nCentroids));
    }

    for (int qb = 0; qb < nQueryBatches; ++qb) {
      const int queryStart = qb * queriesPerBatch;
      const int queriesInBatch = std::min(queriesPerBatch, nQueries - queryStart);
      for (int cb = 0; cb < nCentroidBatches; ++cb) {
        const std::string inFile =
            encryptedDistancesDir + "/encrypted_distance_qbatch_" + FormatIndex(qb) +
            "_cbatch_" + FormatIndex(cb) + ".bin";
        if (!std::filesystem::exists(inFile)) {
          continue;
        }
        Ciphertext<DCRTPoly> ct = LoadCiphertext(inFile);
        Plaintext pt;
        cc->Decrypt(sk, ct, &pt);
        const auto values = pt->GetRealPackedValue();
        const int centroidStart = cb * centroidsPerBatch;
        const int centroidsInBatch =
            std::min(centroidsPerBatch, nCentroids - centroidStart);
        for (int q = 0; q < queriesInBatch; ++q) {
          const int globalQuery = queryStart + q;
          for (int c = 0; c < centroidsInBatch; ++c) {
            const int slot = q * centroidsPerBatch + c;
            if (static_cast<size_t>(slot) >= values.size()) {
              continue;
            }
            scored[static_cast<size_t>(globalQuery)].emplace_back(
                values[static_cast<size_t>(slot)], centroidStart + c);
          }
        }
      }
    }

    std::vector<std::vector<std::pair<double, int>>> topResults(
        static_cast<size_t>(nQueries));
    for (int q = 0; q < nQueries; ++q) {
      auto& row = scored[static_cast<size_t>(q)];
      std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
      });
      const int keep = std::min(topK, static_cast<int>(row.size()));
      topResults[static_cast<size_t>(q)].assign(row.begin(), row.begin() + keep);
    }

    WriteMultiQueryTopK(outputJson, topResults);
    std::cout << "Decrypted packed distances and wrote top " << topK
              << " for " << nQueries << " queries.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_decrypt_topk_query_centroid_batched error: "
              << ex.what() << "\n";
    return 1;
  }
}
