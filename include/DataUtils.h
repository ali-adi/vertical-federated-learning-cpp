#ifndef DATAUTILS_H
#define DATAUTILS_H

#include <vector>
#include <string>
#include <Eigen/Dense>

// Splits a string by a given delimiter.
std::vector<std::string> split(const std::string &s, char delimiter);

// Loads a CSV file and returns its data as a vector of rows (each row is a vector of strings).
std::vector<std::vector<std::string>> loadCSV(const std::string &filename);

// Splits a matrix along the feature dimension.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> verticalSplit(const Eigen::MatrixXd &full_features, int64_t split_col);

// Shuffles feature rows and labels in unison.
void shuffleData(std::vector<std::vector<float>> &feature_rows, std::vector<float> &labels);

// Converts feature rows and labels to Eigen matrices
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> convertToEigenMatrices(
    const std::vector<std::vector<float>> &feature_rows, 
    const std::vector<float> &labels);

#endif // DATAUTILS_H
