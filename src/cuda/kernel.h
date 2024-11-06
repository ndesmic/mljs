extern "C" __declspec(dllexport) float *add_op(const int size, const float *valuesA, const float *valuesB);
extern "C" __declspec(dllexport) void addBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult);
extern "C" __declspec(dllexport) float *sub_op(const int size, const float *valuesA, const float *valuesB);
extern "C" __declspec(dllexport) void subBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult);
extern "C" __declspec(dllexport) float *mul_op(const int size, const float *valuesA, const float *valuesB);
extern "C" __declspec(dllexport) void mulBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB);
extern "C" __declspec(dllexport) float *div_op(const int size, const float *valuesA, const float *valuesB);
extern "C" __declspec(dllexport) void divBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB);
extern "C" __declspec(dllexport) float *pow_op(const int size, const float *valuesA, const float *valuesB);
extern "C" __declspec(dllexport) void powBackprop_op(const int size, float *gradA, float *gradB, const float *gradResult, const float *valuesA, const float *valuesB);

extern "C" __declspec(dllexport) float *neg_op(const int size, const float *values);
extern "C" __declspec(dllexport) void negBackprop_op(const int size, float *grad, const float *gradResult);
extern "C" __declspec(dllexport) float *exp_op(const int size, const float *values);
extern "C" __declspec(dllexport) void expBackprop_op(const int size, float *grad, const float *gradResult, const float *values);
extern "C" __declspec(dllexport) float *tanh_op(const int size, const float *values);
extern "C" __declspec(dllexport) void tanhBackprop_op(const int size, float *grad, const float *gradResult, const float *values);

extern "C" __declspec(dllexport) float *sum_op(const int *shape, const int shapeSize, const int dimToReduce, const float *values);
extern "C" __declspec(dllexport) void sumBackprop_op(const int *shape, const int shapeSize, const int dimToReduce, float *grad, const float *gradResult);