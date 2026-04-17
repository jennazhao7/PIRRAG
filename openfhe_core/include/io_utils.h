#pragma once

#if __has_include(<openfhe.h>)
#include <openfhe.h>
#elif __has_include(<pke/openfhe.h>)
#include <pke/openfhe.h>
#elif __has_include(<openfhe/pke/openfhe.h>)
#include <openfhe/pke/openfhe.h>
#else
#error "OpenFHE header not found. Tried <openfhe.h>, <pke/openfhe.h>, <openfhe/pke/openfhe.h>."
#endif

#include <string>
#include <vector>

namespace openfhe_migration {

using lbcrypto::Ciphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;
using lbcrypto::PrivateKey;
using lbcrypto::PublicKey;

struct ContextBundlePaths {
  std::string contextFile;
  std::string publicKeyFile;
  std::string secretKeyFile;
  std::string evalMultKeyFile;
  std::string evalSumKeyFile;
};

struct CliArgs {
  std::vector<std::string> positional;
  std::vector<std::pair<std::string, std::string>> kv;
};

CliArgs ParseArgs(int argc, char** argv);
std::string GetArg(const CliArgs& args, const std::string& name, const std::string& defaultValue = "");
bool HasArg(const CliArgs& args, const std::string& name);

ContextBundlePaths ResolveContextPaths(const std::string& contextDir);

std::vector<double> ReadVectorFile(const std::string& path);
std::vector<std::vector<double>> ReadMatrixFile(const std::string& path);
void WriteJsonTopK(
    const std::string& path,
    const std::vector<double>& distances,
    const std::vector<int>& indices);
void WriteDistanceMetadata(const std::string& path, int nDistances, int centroidDim);
int ParseDistanceCount(const std::string& metadataPath);

void SaveContextAndKeys(
    const ContextBundlePaths& paths,
    const CryptoContext<DCRTPoly>& cc,
    const PublicKey<DCRTPoly>& pk,
    const PrivateKey<DCRTPoly>& sk);
CryptoContext<DCRTPoly> LoadContext(const std::string& contextFile);
PublicKey<DCRTPoly> LoadPublicKey(const std::string& publicKeyFile);
PrivateKey<DCRTPoly> LoadSecretKey(const std::string& secretKeyFile);
void LoadEvalKeys(const ContextBundlePaths& paths);
void SaveCiphertext(const std::string& path, const Ciphertext<DCRTPoly>& ct);
Ciphertext<DCRTPoly> LoadCiphertext(const std::string& path);

std::vector<uint32_t> ParseUintCsv(const std::string& csv);

}  // namespace openfhe_migration

