function [S, T, fusionInfo] = aligned(Z, c, target_view, fusionOptions)
% ALIGNED 执行跨视图锚点对齐与融合。
% 功能简介：
% 对每个非基准视图求解到基准视图的匹配矩阵，并在对齐后执行融合。
% 默认保持原始 3AMVC 的等权平均；当 fusionOptions.useQualityWeight 为
% true 且提供 qualityScores 时，执行基于视图质量的加权融合。
%
% 输入参数说明：
%   Z            : cell，长度为 v。第 i 个元素为 m_i x n 的锚图。
%   c            : 标量，对齐阶段 DSPFP 的权衡参数。
%   target_view  : 标量，基准视图索引。
%   fusionOptions: 结构体，可选，支持字段：
%                  - useQualityWeight : 是否启用质量加权，默认 false。
%                  - qualityScores    : v x 1 视图质量分数，越小越好。
%                  - tauQ             : softmax 温度参数，默认 1。
%
% 输出参数说明：
%   S          : m_b x n 的融合后锚图。
%   T          : cell，长度为 v。第 i 个元素为对齐矩阵。
%   fusionInfo : 结构体，包含融合模式、视图权重、质量分数和基准视图索引。
%
% 维度说明：
%   若基准视图锚点数为 m_b，则 T{i} 为 m_b x m_i，S 为 m_b x n。
%
% 注意事项：
% 1. 若不传 fusionOptions，本函数行为与原始等权融合保持兼容。
% 2. 质量分数越小表示视图质量越高，因此权重通过 exp(-q_i/tauQ) 计算。
%
% See also algo_qp, DSPFP

numview = length(Z);
if nargin < 4 || isempty(fusionOptions)
    fusionOptions = struct();
end

validate_aligned_inputs(Z, c, target_view, numview);
fusionInfo = parse_fusion_options(fusionOptions, numview, target_view);

S = fusionInfo.weights(target_view) * Z{target_view};
T = cell(numview, 1);
T{target_view} = eye(size(Z{target_view}, 1));

for nv = 1:numview
    if nv ~= target_view
        K = Z{target_view} * Z{nv}';
        S1 = Z{target_view} * Z{target_view}';
        S2 = Z{nv} * Z{nv}';
        T{nv} = DSPFP(S1, S2, K, c);
        S = S + fusionInfo.weights(nv) * (T{nv} * Z{nv});
    end
end
end


function validate_aligned_inputs(Z, c, target_view, numview)
% VALIDATE_ALIGNED_INPUTS 检查对齐阶段关键输入。

if ~iscell(Z) || isempty(Z)
    error('输入 Z 必须是非空 cell 数组。');
end
if ~isscalar(c) || ~isnumeric(c) || ~isfinite(c) || c <= 0
    error('参数 c 必须是有限正数。');
end
if ~isscalar(target_view) || target_view < 1 || target_view > numview || target_view ~= floor(target_view)
    error('target_view 必须是位于 [1, %d] 范围内的整数。', numview);
end

sampleNum = size(Z{target_view}, 2);
for iv = 1:numview
    if isempty(Z{iv}) || ~isnumeric(Z{iv})
        error('Z{%d} 必须是非空数值矩阵。', iv);
    end
    if size(Z{iv}, 2) ~= sampleNum
        error('所有视图锚图的样本数必须一致。Z{%d} 的列数与基准视图不一致。', iv);
    end
    if any(~isfinite(Z{iv}(:)))
        error('Z{%d} 中包含 NaN 或 Inf，无法执行对齐。', iv);
    end
end
end


function fusionInfo = parse_fusion_options(fusionOptions, numview, target_view)
% PARSE_FUSION_OPTIONS 解析并生成融合配置。

fusionInfo = struct();
fusionInfo.targetView = target_view;
fusionInfo.tauQ = 1;
fusionInfo.qualityScores = nan(numview, 1);
fusionInfo.weights = ones(numview, 1) / numview;
fusionInfo.mode = 'uniform';

if ~isstruct(fusionOptions)
    error('fusionOptions 必须是结构体。');
end

if isfield(fusionOptions, 'tauQ') && ~isempty(fusionOptions.tauQ)
    tauQ = fusionOptions.tauQ;
    if ~isscalar(tauQ) || ~isnumeric(tauQ) || ~isfinite(tauQ) || tauQ <= 0
        error('fusionOptions.tauQ 必须是有限正数。');
    end
    fusionInfo.tauQ = tauQ;
end

useQualityWeight = false;
if isfield(fusionOptions, 'useQualityWeight') && ~isempty(fusionOptions.useQualityWeight)
    useQualityWeight = logical(fusionOptions.useQualityWeight);
end

if useQualityWeight
    if ~isfield(fusionOptions, 'qualityScores') || isempty(fusionOptions.qualityScores)
        error('启用质量加权时，必须提供 fusionOptions.qualityScores。');
    end

    qualityScores = double(fusionOptions.qualityScores(:));
    if numel(qualityScores) ~= numview
        error('qualityScores 的长度必须等于视图数 %d。', numview);
    end
    if any(~isfinite(qualityScores))
        error('qualityScores 中包含 NaN 或 Inf，无法计算融合权重。');
    end

    fusionInfo.qualityScores = qualityScores;
    fusionInfo.weights = compute_quality_weights(qualityScores, fusionInfo.tauQ);
    fusionInfo.mode = 'quality_weighted';
end
end


function weights = compute_quality_weights(qualityScores, tauQ)
% COMPUTE_QUALITY_WEIGHTS 根据质量分数计算 softmax 权重。

logits = -qualityScores(:) ./ tauQ;
logits = logits - max(logits);
weights = exp(logits);
weightSum = sum(weights);
if ~(isfinite(weightSum) && weightSum > 0)
    error('质量权重归一化失败，请检查 qualityScores 或 tauQ。');
end
weights = weights / weightSum;
end
