## What is this?
Python scripts in this folder are used to generate lite interpreter models for Android and iOS simulator tests. The goal of these tests is to detect changes that would break existing mobile models used in production (usually they are generated by earlier PyTorch versions). These scripts are based on PyTorch public API (https://pytorch.org/docs/stable/), and are grouped in a similar way:
- math_ops (https://pytorch.org/docs/stable/torch.html#math-operations)
  - pointwise_ops
  - reduction_ops
  - comparison_ops
  - spectral_ops
  - other_math_ops
  - blas_lapack_ops
- sampling_ops (https://pytorch.org/docs/stable/torch.html#random-sampling)
- tensor ops (https://pytorch.org/docs/stable/torch.html#tensors)
  - tensor_general_ops
  - tensor_creation_ops
  - tensor_indexing_ops
  - tensor_typing_ops
  - tensor_view_ops
- nn ops (https://pytorch.org/docs/stable/nn.html)
  - convolution_ops
  - pooling_ops
  - padding_ops
  - activation_ops
  - normalization_ops
  - recurrent_ops
  - transformer_ops
  - linear_ops
  - dropout_ops
  - sparse_ops
  - distance_function_ops
  - loss_function_ops
  - vision_function_ops
  - shuffle_ops
  - nn_utils_ops
- quantization ops (https://pytorch.org/docs/stable/quantization.html)
  - general_quant_ops
  - dynamic_quant_ops
  - static_quant_ops
  - fused_quant_ops
- TorchScript builtin ops (https://pytorch.org/docs/stable/jit_builtin_functions.html)
  - torchscript_builtin_ops
  - torchscript_collection_ops
- torchvision_models (https://pytorch.org/vision/stable/models.html)
  - mobilenet_v2

The generated models are located at
https://github.com/pytorch/pytorch/tree/master/android/pytorch_android/src/androidTest/assets (Android)
https://github.com/pytorch/pytorch/tree/master/ios/TestApp/models/ (iOS) <!-- @lint-ignore -->

These test models will be executed in Android and iOS simulator tests. Note that we only check if there's error in model execution, but don't check the correctness of model output.

## Checked-in models and on-the-fly models
Each test model has a checked-in version and a on-the-fly version. The checked-in versions are stored in this repo (see above model paths) and will only be updated when necessary. The on-the-fly version will be generated during simulator test, with a "_temp" suffix, e.g., "reduction_ops_temp.ptl". Do not commit them.

NOTE: currently Android simulator test does not generate on-the-fly models. Only iOS test does.

## Diagnose failed test
If the simulator test is falling, that means the current change will potentially break a production model. So be careful. The detailed error message can be found in test log. If the change has to be made, make sure it doesn't break existing production models, and update the failed test model as appropriate (see the next section).

You can also run these tests locally, please see the instruction in android and ios folder. Remember to generate on-the-fly test models if you want to test it locally (but don't commit these models with _temp suffix).
```
python test/mobile/model_test/gen_test_model.py ios-test
```

## Update test model
If for any reason a test model needs to be updated, run this script:
```
python test/mobile/model_test/gen_test_model.py <model_name_without_suffix>
```
For example,
```
python test/mobile/model_test/gen_test_model.py reduction_ops
python test/mobile/model_test/gen_test_model.py mobilenet_v2
```

You can also update all test models for android and iOS:
```
python test/mobile/model_test/gen_test_model.py android
python test/mobile/model_test/gen_test_model.py ios
```

## Test Coverage
The test coverage is based on the number of root ops tested in these test models. The full list of generated ops can be found in:
https://github.com/pytorch/pytorch/blob/master/test/mobile/model_test/coverage.yaml

In additional, the simulator tests will also report the percentage of Meta's production ops that are covered. The list of production ops changes overtime, so a Meta employee needs to regularly udpate the list it using
```
python test/mobile/model_test/update_production_ops.py ~/fbsource/xplat/pytorch_models/build/all_mobile_model_configs.yaml
```
