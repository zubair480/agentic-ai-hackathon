# Vitis AI Model Compatibility Report

- Generated: 2026-06-05T16:09:32
- Model input folder: `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original`
- Python files scanned: 17
- Classes discovered: 28
- Operator uses discovered: 379

## Candidate Files

- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/eval.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/main_finetune.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/models/USSFCNet/CMConv.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/models/USSFCNet/MSDConv_SSFC.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/models/USSFCNet/SSFC.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/models/USSFCNet/USSFCNet.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/models/resunet18_pro.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/models/sslchang_net.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/options/test_options.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/options/train_options.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/utils/__init__.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/utils/dataloaders.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/utils/helpers.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/utils/losses.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/utils/metrics.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/utils/parser.py`
- `/Users/meatisdelicious/Documents/Documents/7_projets_perso/Hackaton_AWS_SF/agentic-ai-hackathon/model_input/Transfer-Model_original/utils/transforms.py`

## Classes And Functions

- `FocalLoss` in `Transfer-Model_original/utils/metrics.py:9` bases: nn.Module; methods: __init__, forward
- `TverskyLoss` in `Transfer-Model_original/utils/metrics.py:122` bases: nn.Module; methods: __init__, forward
- `Normalize` in `Transfer-Model_original/utils/transforms.py:8` bases: object; methods: __init__, __call__
- `ToTensor` in `Transfer-Model_original/utils/transforms.py:31` bases: object; methods: __call__
- `RandomHorizontalFlip` in `Transfer-Model_original/utils/transforms.py:53` bases: object; methods: __call__
- `RandomVerticalFlip` in `Transfer-Model_original/utils/transforms.py:66` bases: object; methods: __call__
- `RandomFixRotate` in `Transfer-Model_original/utils/transforms.py:79` bases: object; methods: __init__, __call__
- `RandomRotate` in `Transfer-Model_original/utils/transforms.py:99` bases: object; methods: __init__, __call__
- `RandomGaussianBlur` in `Transfer-Model_original/utils/transforms.py:116` bases: object; methods: __call__
- `RandomScaleCrop` in `Transfer-Model_original/utils/transforms.py:131` bases: object; methods: __init__, __call__
- `FixScaleCrop` in `Transfer-Model_original/utils/transforms.py:168` bases: object; methods: __init__, __call__
- `FixedResize` in `Transfer-Model_original/utils/transforms.py:194` bases: object; methods: __init__, __call__
- `CDDloader` in `Transfer-Model_original/utils/dataloaders.py:111` bases: data.Dataset; methods: __init__, __getitem__, __len__
- `TwoLayerConv` in `Transfer-Model_original/models/sslchang_net.py:10` bases: nn.Module; methods: __init__, forward
- `SSLChangeProjector` in `Transfer-Model_original/models/sslchang_net.py:32` bases: nn.Module; methods: __init__, forward
- `SSLChangePredictor` in `Transfer-Model_original/models/sslchang_net.py:79` bases: nn.Module; methods: __init__, forward
- `ChannelAttention` in `Transfer-Model_original/models/sslchang_net.py:95` bases: nn.Module; methods: __init__, forward
- `SpatialAttention` in `Transfer-Model_original/models/sslchang_net.py:122` bases: nn.Module; methods: __init__, forward
- `ContrastiveNet` in `Transfer-Model_original/models/sslchang_net.py:149` bases: nn.Module; methods: __init__, forward
- `UpSampling2x` in `Transfer-Model_original/models/resunet18_pro.py:6` bases: nn.Module; methods: __init__, forward
- `ResBlock` in `Transfer-Model_original/models/resunet18_pro.py:16` bases: nn.Module; methods: __init__, forward
- `ResUNet18_Pro` in `Transfer-Model_original/models/resunet18_pro.py:35` bases: nn.Module; methods: __init__, forward
- `CMConv` in `Transfer-Model_original/models/USSFCNet/CMConv.py:7` bases: nn.Module; methods: __init__, forward
- `MSDConv_SSFC` in `Transfer-Model_original/models/USSFCNet/MSDConv_SSFC.py:9` bases: nn.Module; methods: __init__, forward
- `SSFC` in `Transfer-Model_original/models/USSFCNet/SSFC.py:6` bases: torch.nn.Module; methods: __init__, forward
- `First_DoubleConv` in `Transfer-Model_original/models/USSFCNet/USSFCNet.py:9` bases: nn.Module; methods: __init__, forward
- `DoubleConv` in `Transfer-Model_original/models/USSFCNet/USSFCNet.py:25` bases: nn.Module; methods: __init__, forward
- `USSFCNet` in `Transfer-Model_original/models/USSFCNet/USSFCNet.py:41` bases: nn.Module; methods: __init__, forward

Top-level functions:

- `load_pretrained_ContrastiveBackbone` in `Transfer-Model_original/main_finetune.py:17`
- `dice_loss` in `Transfer-Model_original/utils/metrics.py:51`
- `jaccard_loss` in `Transfer-Model_original/utils/metrics.py:86`
- `full_path_loader` in `Transfer-Model_original/utils/dataloaders.py:10`
- `full_test_loader` in `Transfer-Model_original/utils/dataloaders.py:58`
- `cdd_loader` in `Transfer-Model_original/utils/dataloaders.py:89`
- `get_parser_with_args` in `Transfer-Model_original/utils/parser.py:5`
- `print_options` in `Transfer-Model_original/utils/parser.py:15`
- `hybrid_loss` in `Transfer-Model_original/utils/losses.py:9`
- `initialize_metrics` in `Transfer-Model_original/utils/helpers.py:12`
- `get_mean_metrics` in `Transfer-Model_original/utils/helpers.py:34`
- `set_metrics` in `Transfer-Model_original/utils/helpers.py:51`
- `set_test_metrics` in `Transfer-Model_original/utils/helpers.py:82`
- `get_loaders` in `Transfer-Model_original/utils/helpers.py:92`
- `get_test_loaders` in `Transfer-Model_original/utils/helpers.py:115`
- `get_criterion` in `Transfer-Model_original/utils/helpers.py:136`

## Operator Inventory

- `Transfer-Model_original/eval.py`: cat (2), getitem (8)
- `Transfer-Model_original/main_finetune.py`: ConvTranspose2d (2), Sequential (2), cat (4), getitem (12), sum (2)
- `Transfer-Model_original/models/USSFCNet/CMConv.py`: Conv2d (3), cat (2), chunk (2), getitem (4)
- `Transfer-Model_original/models/USSFCNet/MSDConv_SSFC.py`: BatchNorm2d (2), CMConv (1), Conv2d (1), ReLU (2), Sequential (2), cat (1), getitem (1)
- `Transfer-Model_original/models/USSFCNet/SSFC.py`: Sigmoid (2), mean (1), pow (1), sum (1)
- `Transfer-Model_original/models/USSFCNet/USSFCNet.py`: BatchNorm2d (4), Conv2d (3), ConvTranspose2d (4), DoubleConv (12), First_DoubleConv (2), MSDConv_SSFC (2), MaxPool2d (1), ReLU (4), Sequential (2), Sigmoid (2), abs (5), cat (4), sub (5)
- `Transfer-Model_original/models/resunet18_pro.py`: BatchNorm2d (7), Conv2d (10), ConvTranspose2d (1), ReLU (10), Sequential (11), cat (4), getitem (41)
- `Transfer-Model_original/models/sslchang_net.py`: AdaptiveAvgPool2d (2), AdaptiveMaxPool2d (1), BatchNorm1d (7), BatchNorm2d (2), Conv2d (5), Flatten (3), Linear (11), ReLU (9), Sequential (5), Sigmoid (2), TwoLayerConv (4), cat (1), mean (1)
- `Transfer-Model_original/utils/dataloaders.py`: getitem (20)
- `Transfer-Model_original/utils/helpers.py`: getitem (16), mean (1)
- `Transfer-Model_original/utils/metrics.py`: cat (6), getitem (15), mean (4), permute (6), sum (8), transpose (1), view (5)
- `Transfer-Model_original/utils/transforms.py`: getitem (41), transpose (11)

## Compatibility Table

| Operator | Uses | Status | Evidence |
| --- | ---: | --- | --- |
| `AdaptiveAvgPool2d` | 2 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `AdaptiveMaxPool2d` | 1 | unsupported | No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment. |
| `BatchNorm1d` | 7 | unsupported | No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment. |
| `BatchNorm2d` | 15 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `CMConv` | 1 | unsupported | No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment. |
| `Conv2d` | 22 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `ConvTranspose2d` | 7 | conditional | Often requires rewrite to resize plus Conv2d for DPU-friendly deployment. |
| `DoubleConv` | 12 | unsupported | No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment. |
| `First_DoubleConv` | 2 | unsupported | No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment. |
| `Flatten` | 3 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `Linear` | 11 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `MSDConv_SSFC` | 2 | unsupported | No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment. |
| `MaxPool2d` | 1 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `ReLU` | 25 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `Sequential` | 22 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `Sigmoid` | 6 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `TwoLayerConv` | 4 | unsupported | No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment. |
| `abs` | 5 | conditional | Elementwise abs may require CPU fallback or graph rewrite. |
| `cat` | 24 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `chunk` | 2 | conditional | Dynamic tensor splitting is risky; use static slicing when possible. |
| `getitem` | 158 | conditional | Slicing must be static and compiler-friendly. |
| `mean` | 7 | conditional | Reductions can fall back to CPU depending on axes and shape constraints. |
| `permute` | 6 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `pow` | 1 | conditional | Elementwise powers are risky for quantized DPU graphs. |
| `sub` | 5 | conditional | Elementwise subtraction is shape-sensitive in quantized graphs. |
| `sum` | 11 | conditional | Reductions can fall back to CPU depending on axes and shape constraints. |
| `transpose` | 12 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |
| `view` | 5 | supported | Common Vitis AI DPU-friendly operator when exported with static shapes and INT8 calibration. |

## DPU Risks

- `AdaptiveMaxPool2d` (1 uses): No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment.
- `BatchNorm1d` (7 uses): No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment.
- `CMConv` (1 uses): No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment.
- `ConvTranspose2d` (7 uses): Often requires rewrite to resize plus Conv2d for DPU-friendly deployment.
- `DoubleConv` (12 uses): No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment.
- `First_DoubleConv` (2 uses): No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment.
- `MSDConv_SSFC` (2 uses): No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment.
- `TwoLayerConv` (4 uses): No built-in local rule matched this operator; verify against AMD Vitis AI docs before deployment.
- `abs` (5 uses): Elementwise abs may require CPU fallback or graph rewrite.
- `chunk` (2 uses): Dynamic tensor splitting is risky; use static slicing when possible.
- `getitem` (158 uses): Slicing must be static and compiler-friendly.
- `mean` (7 uses): Reductions can fall back to CPU depending on axes and shape constraints.
- `pow` (1 uses): Elementwise powers are risky for quantized DPU graphs.
- `sub` (5 uses): Elementwise subtraction is shape-sensitive in quantized graphs.
- `sum` (11 uses): Reductions can fall back to CPU depending on axes and shape constraints.

## Conservative Plan

Rewrite only operators most likely to block Vitis AI DPU compilation; keep validated model structure intact.

- `AdaptiveMaxPool2d`: Keep unchanged until AMD documentation confirms it blocks DPU compilation. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- `BatchNorm1d`: Keep unchanged until AMD documentation confirms it blocks DPU compilation. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- `CMConv`: Keep unchanged until AMD documentation confirms it blocks DPU compilation. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- `ConvTranspose2d`: Upsample/interpolate to a static size followed by Conv2d + BatchNorm2d + ReLU. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- `DoubleConv`: Keep unchanged until AMD documentation confirms it blocks DPU compilation. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- `First_DoubleConv`: Keep unchanged until AMD documentation confirms it blocks DPU compilation. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- `MSDConv_SSFC`: Keep unchanged until AMD documentation confirms it blocks DPU compilation. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- `TwoLayerConv`: Keep unchanged until AMD documentation confirms it blocks DPU compilation. Reason: Minimizes code churn while addressing the highest-risk compatibility issue.
- Estimated compatibility: 61%
- Accuracy risk: Low to medium; validate changed blocks with the original checkpoint and task metrics.
- Effort: Small targeted refactor plus export/quantization test.

## Aggressive Plan

Convert the model to a static-shape, quantization-friendly graph built from DPU-preferred blocks.

- `AdaptiveMaxPool2d`: Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `BatchNorm1d`: Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `CMConv`: Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `ConvTranspose2d`: Upsample/interpolate to a static size followed by Conv2d + BatchNorm2d + ReLU. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `DoubleConv`: Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `First_DoubleConv`: Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `MSDConv_SSFC`: Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `TwoLayerConv`: Confirm support, then rewrite to Conv2d/BatchNorm2d/ReLU/static-shape blocks if needed. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `abs`: Approximate with ReLU(x) + ReLU(-x) only if accuracy validates, otherwise isolate on CPU. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `chunk`: Replace with explicit static slicing in known channel ranges. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `getitem`: Keep only static index/slice expressions with fixed dimensions. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `mean`: Replace dynamic reductions with fixed pooling layers such as AvgPool2d or AdaptiveAvgPool2d. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `pow`: Replace with multiplication for fixed exponents or remove from the DPU subgraph. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `sub`: Prefer add with pre-negated tensors or ensure static broadcast shapes. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- `sum`: Replace dynamic reductions with fixed pooling or static accumulation paths. Reason: Maximizes DPU partition size and reduces CPU fallback risk.
- Estimated compatibility: 95-100%
- Quantization plan: Run post-training INT8 quantization with a representative calibration set, then inspect DPU/CPU graph partitioning.
- Accuracy risk: Medium; requires calibration, regression testing, and possibly retraining.
- Effort: Broader architecture rewrite plus deployment validation.

## Migration Checklist

- Export the model to ONNX with fixed input shapes.
- Replace conditional or unsupported operators before quantization.
- Run Vitis AI quantization with representative calibration data.
- Inspect compiler graph partitioning for CPU fallback.
- Compare task metrics against the original checkpoint.
