#include "io_utils.h"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace openfhe_migration {

namespace {
template <typename T>
void CheckSerialize(bool ok, const std::string& path, const std::string& op) {
  (void)sizeof(T);
  if (!ok) {
    throw std::runtime_error(op + " failed for path: " + path);
  }
}
}  // namespace

CliArgs ParseArgs(int argc, char** argv) {
  CliArgs out;
  for (int i = 1; i < argc; ++i) {
    std::string token(argv[i]);
    if (token.rfind("--", 0) == 0) {
      if (i + 1 < argc) {
        out.kv.push_back({token, std::string(argv[i + 1])});
        ++i;
      } else {
        out.kv.push_back({token, ""});
      }
    } else {
      out.positional.push_back(token);
    }
  }
  return out;
}

std::string GetArg(const CliArgs& args, const std::string& name, const std::string& defaultValue) {
  for (const auto& kv : args.kv) {
    if (kv.first == name) {
      return kv.second;
    }
  }
  return defaultValue;
}

bool HasArg(const CliArgs& args, const std::string& name) {
  for (const auto& kv : args.kv) {
    if (kv.first == name) {
      return true;
    }
  }
  return false;
}

ContextBundlePaths ResolveContextPaths(const std::string& contextDir) {
  ContextBundlePaths paths;
  paths.contextFile = contextDir + "/context.bin";
  paths.publicKeyFile = contextDir + "/public_key.bin";
  paths.secretKeyFile = contextDir + "/secret_key.bin";
  paths.evalMultKeyFile = contextDir + "/eval_mult_keys.bin";
  paths.evalSumKeyFile = contextDir + "/eval_sum_keys.bin";
  paths.evalAutomorphismKeyFile = contextDir + "/eval_automorphism_keys.bin";
  return paths;
}

std::vector<double> ReadVectorFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open vector file: " + path);
  }
  std::vector<double> out;
  double value = 0.0;
  while (in >> value) {
    out.push_back(value);
  }
  if (out.empty()) {
    throw std::runtime_error("Vector file is empty: " + path);
  }
  return out;
}

std::vector<std::vector<double>> ReadMatrixFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open matrix file: " + path);
  }
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::vector<double> row;
    double value = 0.0;
    while (ss >> value) {
      row.push_back(value);
    }
    if (!row.empty()) {
      rows.push_back(row);
    }
  }
  if (rows.empty()) {
    throw std::runtime_error("Matrix file has no rows: " + path);
  }
  return rows;
}

void WriteJsonTopK(
    const std::string& path,
    const std::vector<double>& distances,
    const std::vector<int>& indices) {
  if (distances.size() != indices.size()) {
    throw std::runtime_error("distances and indices size mismatch");
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write file: " + path);
  }

  double minDistance = distances.empty() ? 0.0 : distances.front();
  double maxDistance = distances.empty() ? 0.0 : distances.front();
  double sum = 0.0;
  for (double d : distances) {
    if (d < minDistance) {
      minDistance = d;
    }
    if (d > maxDistance) {
      maxDistance = d;
    }
    sum += d;
  }
  const double meanDistance = distances.empty() ? 0.0 : (sum / static_cast<double>(distances.size()));

  out << "{\n";
  out << "  \"top_k\": " << distances.size() << ",\n";
  out << "  \"distances\": [";
  for (size_t i = 0; i < distances.size(); ++i) {
    out << distances[i];
    if (i + 1 < distances.size()) {
      out << ", ";
    }
  }
  out << "],\n";
  out << "  \"centroid_indices\": [";
  for (size_t i = 0; i < indices.size(); ++i) {
    out << indices[i];
    if (i + 1 < indices.size()) {
      out << ", ";
    }
  }
  out << "],\n";
  out << "  \"min_distance\": " << minDistance << ",\n";
  out << "  \"max_distance\": " << maxDistance << ",\n";
  out << "  \"mean_distance\": " << meanDistance << "\n";
  out << "}\n";
}

void WriteDistanceMetadata(const std::string& path, int nDistances, int centroidDim) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write metadata file: " + path);
  }
  out << "{\n";
  out << "  \"n_distances\": " << nDistances << ",\n";
  out << "  \"n_centroids\": " << nDistances << ",\n";
  out << "  \"centroid_dim\": " << centroidDim << ",\n";
  out << "  \"backend\": \"openfhe_cpp\"\n";
  out << "}\n";
}

int ParseDistanceCount(const std::string& metadataPath) {
  std::ifstream in(metadataPath);
  if (!in) {
    throw std::runtime_error("Failed to open metadata file: " + metadataPath);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  std::regex rx("\"n_distances\"\\s*:\\s*(\\d+)");
  std::smatch match;
  if (!std::regex_search(text, match, rx)) {
    throw std::runtime_error("n_distances not found in metadata file");
  }
  return std::stoi(match[1].str());
}

void SaveContextAndKeys(
    const ContextBundlePaths& paths,
    const CryptoContext<DCRTPoly>& cc,
    const PublicKey<DCRTPoly>& pk,
    const PrivateKey<DCRTPoly>& sk) {
  CheckSerialize<int>(
      lbcrypto::Serial::SerializeToFile(paths.contextFile, cc, lbcrypto::SerType::BINARY),
      paths.contextFile,
      "Serialize crypto context");
  CheckSerialize<int>(
      lbcrypto::Serial::SerializeToFile(paths.publicKeyFile, pk, lbcrypto::SerType::BINARY),
      paths.publicKeyFile,
      "Serialize public key");
  CheckSerialize<int>(
      lbcrypto::Serial::SerializeToFile(paths.secretKeyFile, sk, lbcrypto::SerType::BINARY),
      paths.secretKeyFile,
      "Serialize secret key");

  std::ofstream evalMultOut(paths.evalMultKeyFile, std::ios::binary);
  if (!evalMultOut) {
    throw std::runtime_error("Failed to open eval mult key file for writing: " + paths.evalMultKeyFile);
  }
  CheckSerialize<int>(
      lbcrypto::CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(
          evalMultOut, lbcrypto::SerType::BINARY, sk->GetKeyTag()),
      paths.evalMultKeyFile,
      "Serialize eval mult keys");

  std::ofstream evalSumOut(paths.evalSumKeyFile, std::ios::binary);
  if (!evalSumOut) {
    throw std::runtime_error("Failed to open eval sum key file for writing: " + paths.evalSumKeyFile);
  }
  CheckSerialize<int>(
      lbcrypto::CryptoContextImpl<DCRTPoly>::SerializeEvalSumKey(
          evalSumOut, lbcrypto::SerType::BINARY, sk->GetKeyTag()),
      paths.evalSumKeyFile,
      "Serialize eval sum keys");
}

CryptoContext<DCRTPoly> LoadContext(const std::string& contextFile) {
  CryptoContext<DCRTPoly> cc;
  CheckSerialize<int>(
      lbcrypto::Serial::DeserializeFromFile(contextFile, cc, lbcrypto::SerType::BINARY),
      contextFile,
      "Deserialize crypto context");
  return cc;
}

PublicKey<DCRTPoly> LoadPublicKey(const std::string& publicKeyFile) {
  PublicKey<DCRTPoly> pk;
  CheckSerialize<int>(
      lbcrypto::Serial::DeserializeFromFile(publicKeyFile, pk, lbcrypto::SerType::BINARY),
      publicKeyFile,
      "Deserialize public key");
  return pk;
}

PrivateKey<DCRTPoly> LoadSecretKey(const std::string& secretKeyFile) {
  PrivateKey<DCRTPoly> sk;
  CheckSerialize<int>(
      lbcrypto::Serial::DeserializeFromFile(secretKeyFile, sk, lbcrypto::SerType::BINARY),
      secretKeyFile,
      "Deserialize secret key");
  return sk;
}

void LoadEvalKeys(const ContextBundlePaths& paths) {
  std::ifstream evalMultIn(paths.evalMultKeyFile, std::ios::binary);
  if (!evalMultIn) {
    throw std::runtime_error("Failed to open eval mult key file for reading: " + paths.evalMultKeyFile);
  }
  CheckSerialize<int>(
      lbcrypto::CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(
          evalMultIn, lbcrypto::SerType::BINARY),
      paths.evalMultKeyFile,
      "Deserialize eval mult keys");

  std::ifstream evalSumIn(paths.evalSumKeyFile, std::ios::binary);
  if (!evalSumIn) {
    throw std::runtime_error("Failed to open eval sum key file for reading: " + paths.evalSumKeyFile);
  }
  CheckSerialize<int>(
      lbcrypto::CryptoContextImpl<DCRTPoly>::DeserializeEvalSumKey(
          evalSumIn, lbcrypto::SerType::BINARY),
      paths.evalSumKeyFile,
      "Deserialize eval sum keys");

  if (std::filesystem::exists(paths.evalAutomorphismKeyFile) &&
      std::filesystem::file_size(paths.evalAutomorphismKeyFile) > 0) {
    std::ifstream evalAutoIn(paths.evalAutomorphismKeyFile, std::ios::binary);
    if (!evalAutoIn) {
      throw std::runtime_error("Failed to open eval automorphism key file for reading: " +
                               paths.evalAutomorphismKeyFile);
    }
    CheckSerialize<int>(
        lbcrypto::CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(
            evalAutoIn, lbcrypto::SerType::BINARY),
        paths.evalAutomorphismKeyFile,
        "Deserialize eval automorphism keys");
  }
}

void SaveEvalAutomorphismKeys(
    const std::string& path,
    const CryptoContext<DCRTPoly>& cc) {
  std::ofstream evalAutoOut(path, std::ios::binary);
  if (!evalAutoOut) {
    throw std::runtime_error("Failed to open eval automorphism key file for writing: " + path);
  }
  CheckSerialize<int>(
      lbcrypto::CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(
          evalAutoOut, lbcrypto::SerType::BINARY, cc),
      path,
      "Serialize eval automorphism keys");
}

void SaveCiphertext(const std::string& path, const Ciphertext<DCRTPoly>& ct) {
  CheckSerialize<int>(
      lbcrypto::Serial::SerializeToFile(path, ct, lbcrypto::SerType::BINARY),
      path,
      "Serialize ciphertext");
}

Ciphertext<DCRTPoly> LoadCiphertext(const std::string& path) {
  Ciphertext<DCRTPoly> ct;
  CheckSerialize<int>(
      lbcrypto::Serial::DeserializeFromFile(path, ct, lbcrypto::SerType::BINARY),
      path,
      "Deserialize ciphertext");
  return ct;
}

std::vector<uint32_t> ParseUintCsv(const std::string& csv) {
  std::vector<uint32_t> out;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      out.push_back(static_cast<uint32_t>(std::stoul(token)));
    }
  }
  return out;
}

}  // namespace openfhe_migration

