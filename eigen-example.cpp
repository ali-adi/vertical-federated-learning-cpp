#include <iostream>
#include <fstream>      
#include <sstream>
#include <random>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

#define LOG_INFO(x) std::cout << "[INFO] " << x << std::endl
using namespace Eigen;
using namespace std;

MatrixXd randomMatrix(int rows, int cols) {
    MatrixXd m(rows, cols);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, 0.1);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m(i, j) = d(gen);
    return m;
}

MatrixXd sigmoid(const MatrixXd& x) {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
}

MatrixXd sigmoid_derivative(const MatrixXd& x) {
    return (x.array() * (1 - x.array())).matrix();
}

// Fully connected model class
class FCModel {
public:
    MatrixXd W1, b1, W2, b2, W3, b3;
    MatrixXd input, a1, a2, output;

    FCModel(int input_dim) {
        W1 = randomMatrix(128, input_dim);
        b1 = randomMatrix(128, 1);
        W2 = randomMatrix(256, 128);
        b2 = randomMatrix(256, 1);
        W3 = randomMatrix(256, 256);
        b3 = randomMatrix(256, 1);
    }

    MatrixXd forward(const MatrixXd& x) {
        input = x;
        a1 = sigmoid(W1 * input + b1);
        a2 = sigmoid(W2 * a1 + b2);
        output = sigmoid(W3 * a2 + b3);
        return output;
    }

    MatrixXd backward(const MatrixXd& grad_output, double lr) {
        // 注意这里显式转回Matrix
        MatrixXd delta3 = (grad_output.array() * sigmoid_derivative(output).array()).matrix();
        MatrixXd dW3 = delta3 * a2.transpose();
        MatrixXd db3 = delta3;
    
        MatrixXd delta2 = ((W3.transpose() * delta3).array() * sigmoid_derivative(a2).array()).matrix();
        MatrixXd dW2 = delta2 * a1.transpose();
        MatrixXd db2 = delta2;
    
        MatrixXd delta1 = ((W2.transpose() * delta2).array() * sigmoid_derivative(a1).array()).matrix();
        MatrixXd dW1 = delta1 * input.transpose();
        MatrixXd db1 = delta1;
    
        W3 -= lr * dW3;
        b3 -= lr * db3;
        W2 -= lr * dW2;
        b2 -= lr * db2;
        W1 -= lr * dW1;
        b1 -= lr * db1;
    
        return W1.transpose() * delta1;
    }
};

// Central Model with dot product
class CentralDotModel {
    public:
        double logits;
    
        double forward(const MatrixXd& x1, const MatrixXd& x2) {
            logits = 10*(x1.transpose() * x2)(0, 0); // 修正点积
            return 1.0 / (1.0 + exp(-logits));
        }
    
        pair<MatrixXd, MatrixXd> backward(double pred, int label, const MatrixXd& x1, const MatrixXd& x2, double pos_weight = 1.0) {
            double grad_logits = (pred - label) * (label == 1 ? pos_weight : 1.0);
            MatrixXd grad_x1 = grad_logits * x2;
            MatrixXd grad_x2 = grad_logits * x1;
            return {grad_x1, grad_x2};
        }
    };

// Hash encoding categorical features
unordered_map<string, int> feature_index;
int feature_counter = 0;
int get_feature_index(const string& key) {
    if (feature_index.count(key) == 0)
        feature_index[key] = feature_counter++;
    return feature_index[key];
}

void load_data(const string& filename,
    vector<MatrixXd>& XA, vector<MatrixXd>& XB, vector<int>& labels,
    double val_ratio = 0.1,
    vector<MatrixXd>* val_XA = nullptr, vector<MatrixXd>* val_XB = nullptr, vector<int>* val_labels = nullptr) {

    ifstream file(filename);
    string line;
    getline(file, line);
    vector<vector<string>> raw_fields;
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<string> fields;
        while (getline(ss, token, ',')) fields.push_back(token);
        raw_fields.push_back(fields);
    }
    random_device rd;
    mt19937 g(rd());
    shuffle(raw_fields.begin(), raw_fields.end(), g);

    int total = raw_fields.size();
    int val_size = static_cast<int>(val_ratio * total);
    for (int i = 0; i < total; ++i) {
        const auto& fields = raw_fields[i];
        int label = stoi(fields[1]);
        VectorXd guest_vec(8);
        for (int j = 0; j < 8; ++j)
            guest_vec(j) = get_feature_index(fields[16 + j]);
        VectorXd host_vec(14);
        for (int j = 0; j < 14; ++j)
            host_vec(j) = get_feature_index(fields[2 + j]);

        if (val_XA && i < val_size) {
            val_XA->push_back(guest_vec);
            val_XB->push_back(host_vec);
            val_labels->push_back(label);
        } else {
            XA.push_back(guest_vec);
            XB.push_back(host_vec);
            labels.push_back(label);
        }
    }
}


void evaluate(FCModel& modelA, FCModel& modelB, CentralDotModel& modelC,
    const vector<MatrixXd>& XA_val, const vector<MatrixXd>& XB_val,
    const vector<int>& labels_val, const double pos_weight) {
    double total_loss = 0;
    int TP = 0, TN = 0, FP = 0, FN = 0;
    vector<pair<double, int>> scores;  // (pred, label)


    for (size_t i = 0; i < XA_val.size(); i++) {
    auto outA = modelA.forward(XA_val[i]);
    auto outB = modelB.forward(XB_val[i]);
    double pred = modelC.forward(outA, outB);
    pred = max(min(pred, 1.0 - 1e-7), 1e-7); // 防止 log(0)
    scores.emplace_back(pred, labels_val[i]);  // 记录预测分数和真实标签
    // double loss = -(labels_val[i] * log(pred) + (1 - labels_val[i]) * log(1 - pred));
    double loss = -(labels_val[i] * log(pred) * pos_weight + (1 - labels_val[i]) * log(1 - pred));

    total_loss += loss;
    double threshold = 0.55;
    int predicted_label = pred >= threshold ? 1 : 0;
    int true_label = labels_val[i];
    if (predicted_label == 1 && true_label == 1) TP++;
    else if (predicted_label == 0 && true_label == 0) TN++;
    else if (predicted_label == 1 && true_label == 0) FP++;
    else if (predicted_label == 0 && true_label == 1) FN++;
    }

    sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;  // 按预测得分降序排列
    });

    int P = count(labels_val.begin(), labels_val.end(), 1);
    int N = labels_val.size() - P;

    double auc = 0.0;
    int tp = 0, fp = 0;
    double prev_fpr = 0.0, prev_tpr = 0.0;
    for (const auto& [score, label] : scores) {
        if (label == 1) tp++;
        else fp++;

        double tpr = tp / double(P);
        double fpr = fp / double(N);

        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;  // 梯形法
        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    // ---- 指标输出 ----
    double avg_loss = total_loss / XA_val.size();
    double accuracy = double(TP + TN) / XA_val.size();
    double precision = TP + FP > 0 ? double(TP) / (TP + FP) : 0;
    double recall = TP + FN > 0 ? double(TP) / (TP + FN) : 0;
    double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

    cout << "[Validation] Loss: " << avg_loss
         << ", Accuracy: " << accuracy
         << ", Precision: " << precision
         << ", Recall: " << recall
         << ", F1-score: " << f1
         << ", AUC: " << auc << endl;
}

int main() {
    FCModel modelA(8), modelB(14);
    CentralDotModel modelC;
    vector<MatrixXd> XA_train, XB_train, XA_val, XB_val;
    vector<int> labels_train, labels_val;
    load_data("dataset/credit/train_500k.csv", XA_train, XB_train, labels_train, 0.1, &XA_val, &XB_val, &labels_val);

    // label 分布检查
    int positive = count(labels_train.begin(), labels_train.end(), 1);
    int negative = labels_train.size() - positive;
    double pos_weight = negative > 0 ? double(negative) / positive : 1.0;
    cout << "Train set positive rate: " << double(positive) / labels_train.size() << ", pos_weight = " << pos_weight << endl;

    double lr = 0.005;
    int batch_size = 32;
    int max_epoch = 100;
    int patience = 5, stop_counter = 0;
    double best_loss = 1e9;
    // LOG_INFO(XA_train[0]);
    // LOG_INFO(XB_train[0]);
    for (int epoch = 0; epoch < max_epoch; epoch++) {
        double total_loss = 0;
        for (size_t i = 0; i < XA_train.size(); i += batch_size) {
            MatrixXd grad_A = MatrixXd::Zero(256, 1);
            MatrixXd grad_B = MatrixXd::Zero(256, 1);

            for (size_t j = i; j < min(i + batch_size, XA_train.size()); ++j) {
                auto outA = modelA.forward(XA_train[j]);
                auto outB = modelB.forward(XB_train[j]);
                double pred = modelC.forward(outA, outB);
                pred = max(min(pred, 1.0 - 1e-7), 1e-7);
                // double loss = -(labels_train[j] * log(pred) + (1 - labels_train[j]) * log(1 - pred));
                double loss = -(labels_train[j] * log(pred) * pos_weight + (1 - labels_train[j]) * log(1 - pred));
                total_loss += loss;

                auto grads = modelC.backward(pred, labels_train[j], outA, outB, pos_weight);
                grad_A += grads.first;
                grad_B += grads.second;
            }

            grad_A /= batch_size;
            grad_B /= batch_size;
            modelA.backward(grad_A, lr);
            modelB.backward(grad_B, lr);
        }

        cout << "Epoch " << epoch << ", Train Loss: " << total_loss / XA_train.size() << endl;
        evaluate(modelA, modelB, modelC, XA_val, XB_val, labels_val,pos_weight);

    }
    return 0;
}
