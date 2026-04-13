%% GRID_SEARCH_REUTERS1200 对 Quality-Aware Weighted Anchor Fusion 执行网格搜索
% 功能简介：
% 默认对 dataset/Reuters-1200.mat 数据集搜索 beta、lambda 和 tauQ，并以
% ACC 最大作为最优参数组合选择标准。脚本支持直接改为其他 .mat 数据集，
% 并自动解析常见标签变量名（如 Y、y、label、gt），不会修改数据集内容。
%
% 输入参数说明：
% 本脚本无函数输入。请直接修改“用户可配置参数”区域中的数据集名、搜索网格、
% 随机种子和重复次数。
%
% 输出参数说明：
% 运行结束后，工作区将生成：
%   searchRecords - 每一次参数搜索的详细结果结构体数组
%   searchTable   - searchRecords 对应的结果表
%   comboTable    - 参数组合汇总结果表
%   bestResult    - ACC 最优的搜索结果结构体
%   datasetInfo   - 当前数据集的信息结构体
%
% 维度说明：
% X 为 v x 1 或 1 x v 的 cell，每个视图为 n x d_v；
% Y 为 n x 1 标签向量。
%
% 注意事项：
% 1. 该脚本默认启用 3.X Quality-Aware Weighted Anchor Fusion。
% 2. 单次搜索耗时包含 Neighbor、algo_qp 和聚类评价的总耗时。
% 3. Reuters-1200 上的质量分数量级较大，因此 tauQ 默认搜索范围取较大值。
% 4. 每次搜索都会输出每个视图的质量分数和融合权重。
%
% See also algo_qp, aligned, Neighbor, myNMIACCwithmean

clear;
clc;
warning off;

%% 用户可配置参数
datasetFile = 'Reuters-1200.mat';
betaList = [0.1, 1, 10, 100, 1000];
lambdaList = [1e2, 1e3, 1e4, 1e5];
tauQList = [1e3, 1e4, 1e5, 1e6];
repeatNum = 1;
baseSeed = 1;
labelFieldCandidates = {'Y', 'y', 'gt', 'truth', 'label', 'labels'};

%% 环境初始化
projectRoot = fileparts(mfilename('fullpath'));
datasetDir = fullfile(projectRoot, 'dataset');
addpath(genpath(projectRoot));

validate_search_config(datasetFile, betaList, lambdaList, tauQList, repeatNum, baseSeed, labelFieldCandidates);
[X, Y, datasetInfo] = load_multiview_dataset(datasetDir, datasetFile, labelFieldCandidates);
k = datasetInfo.classNum;
viewNum = datasetInfo.viewNum;
metricNames = {'ACC', 'NMI', 'Purity', 'Fscore', 'Precision', 'Recall', 'AR', 'Entropy'};
totalSearchNum = numel(betaList) * numel(lambdaList) * numel(tauQList) * repeatNum;

fprintf('开始网格搜索：数据集=%s | 标签字段=%s | 样本数=%d | 视图数=%d | 类别数=%d\n', ...
    datasetInfo.datasetFile, datasetInfo.labelField, datasetInfo.sampleNum, ...
    datasetInfo.viewNum, datasetInfo.classNum);
fprintf('搜索网格：beta=%s | lambda=%s | tauQ=%s | repeatNum=%d | baseSeed=%d\n', ...
    mat2str(betaList), mat2str(lambdaList), mat2str(tauQList), repeatNum, baseSeed);

searchRecords(totalSearchNum, 1) = create_empty_record(metricNames, viewNum);
bestResult = create_empty_record(metricNames, viewNum);
bestResult.ACC = -inf;

searchIdx = 0;
comboIdx = 0;
comboSummary = repmat(create_empty_combo_summary(viewNum), ...
    numel(betaList) * numel(lambdaList) * numel(tauQList), 1);

for ib = 1:numel(betaList)
    beta = betaList(ib);
    for il = 1:numel(lambdaList)
        lambda = lambdaList(il);
        for itq = 1:numel(tauQList)
            tauQ = tauQList(itq);
            comboIdx = comboIdx + 1;
            comboACC = nan(repeatNum, 1);
            comboTime = nan(repeatNum, 1);
            comboWeights = nan(repeatNum, viewNum);

            for ir = 1:repeatNum
                currentSeed = baseSeed + ir - 1;
                rng(currentSeed, 'twister');
                tic;

                targetView = NaN;
                iterNum = NaN;
                obj = [];
                neighborTime = nan(viewNum, 1);
                qualityScores = nan(viewNum, 1);
                metricMean = nan(1, numel(metricNames));
                metricStd = nan(1, numel(metricNames));
                fusionInfo = struct('mode', 'quality_weighted', 'weights', nan(viewNum, 1), ...
                    'qualityScores', nan(viewNum, 1), 'tauQ', tauQ, 'targetView', NaN);
                status = 'success';
                errorMessage = '';

                try
                    thetaAll = cell(viewNum, 1);
                    for iv = 1:viewNum
                        [~, timeNeighbor, ~, object, theta] = Neighbor(X{iv}, Y);
                        thetaAll{iv, 1} = theta;
                        qualityScores(iv, 1) = sum(object);
                        neighborTime(iv, 1) = timeNeighbor;
                    end

                    [~, targetView] = min(qualityScores);
                    fusionOptions = struct();
                    fusionOptions.useQualityWeight = true;
                    fusionOptions.qualityScores = qualityScores;
                    fusionOptions.tauQ = tauQ;

                    [U, ~, ~, iterNum, obj, fusionInfo] = algo_qp(X, Y, thetaAll, ...
                        beta, lambda, targetView, fusionOptions);
                    [metricMean, metricStd] = myNMIACCwithmean(U, Y, k);

                    if isempty(obj) || any(~isfinite(obj))
                        error('当前参数组合产生了非法目标函数值，请检查 beta、lambda 或 tauQ。');
                    end
                    if any(~isfinite(metricMean))
                        error('当前参数组合产生了非法评价指标，请检查算法输出是否稳定。');
                    end
                catch ME
                    status = 'failed';
                    errorMessage = ME.message;
                end

                elapsedTime = toc;
                searchIdx = searchIdx + 1;
                searchRecords(searchIdx) = build_search_record(datasetInfo, metricNames, viewNum, ...
                    beta, lambda, tauQ, ir, currentSeed, targetView, iterNum, obj, ...
                    sum(neighborTime(isfinite(neighborTime))), elapsedTime, qualityScores, ...
                    fusionInfo.weights, metricMean, metricStd, status, errorMessage);

                comboACC(ir, 1) = metricMean(1);
                comboTime(ir, 1) = elapsedTime;
                comboWeights(ir, :) = fusionInfo.weights(:)';

                if strcmp(status, 'success')
                    fprintf(['搜索 %3d/%3d | beta=%-8g | lambda=%-8g | tauQ=%-8g | ' ...
                        'repeat=%d/%d | seed=%d | ACC=%.6f | Time=%.2fs\n'], ...
                        searchIdx, totalSearchNum, beta, lambda, tauQ, ir, repeatNum, ...
                        currentSeed, metricMean(1), elapsedTime);
                    fprintf('  质量分数: %s\n', vector_to_text(qualityScores));
                    fprintf('  视图权重: %s\n', vector_to_text(fusionInfo.weights));

                    if metricMean(1) > bestResult.ACC || ...
                            (abs(metricMean(1) - bestResult.ACC) <= eps && elapsedTime < bestResult.elapsedTime)
                        bestResult = searchRecords(searchIdx);
                    end
                else
                    fprintf(['搜索 %3d/%3d | beta=%-8g | lambda=%-8g | tauQ=%-8g | ' ...
                        'repeat=%d/%d | seed=%d | 失败 | Time=%.2fs | 错误=%s\n'], ...
                        searchIdx, totalSearchNum, beta, lambda, tauQ, ir, repeatNum, ...
                        currentSeed, elapsedTime, errorMessage);
                end
            end

            comboSummary(comboIdx) = build_combo_summary(comboSummary(comboIdx), ...
                beta, lambda, tauQ, comboACC, comboTime, comboWeights);

            fprintf(['组合汇总 | beta=%-8g | lambda=%-8g | tauQ=%-8g | ' ...
                '最佳ACC=%.6f | 平均ACC=%.6f | 平均耗时=%.2fs\n'], ...
                beta, lambda, tauQ, comboSummary(comboIdx).bestACC, ...
                comboSummary(comboIdx).meanACC, comboSummary(comboIdx).meanTime);
            fprintf('  平均视图权重: %s\n', comboSummary(comboIdx).meanWeightsText);
        end
    end
end

searchTable = struct2table(searchRecords);
comboTable = struct2table(comboSummary);

if isfinite(bestResult.ACC)
    fprintf(['\n最优参数组合：beta=%g | lambda=%g | tauQ=%g | ACC=%.6f | ' ...
        'NMI=%.6f | Time=%.2fs | seed=%d\n'], ...
        bestResult.beta, bestResult.lambda, bestResult.tauQ, bestResult.ACC, ...
        bestResult.NMI, bestResult.elapsedTime, bestResult.seed);
    fprintf('对应视图质量分数：%s\n', bestResult.qualityScoresText);
    fprintf('对应视图权重：%s\n', bestResult.viewWeightsText);
    fprintf(['其余指标：Purity=%.6f | Fscore=%.6f | Precision=%.6f | ' ...
        'Recall=%.6f | AR=%.6f | Entropy=%.6f\n'], ...
        bestResult.Purity, bestResult.Fscore, bestResult.Precision, ...
        bestResult.Recall, bestResult.AR, bestResult.Entropy);
else
    warning('所有参数组合均搜索失败，请检查数据集、随机种子或搜索网格范围。');
end


function [X, Y, datasetInfo] = load_multiview_dataset(datasetDir, datasetFile, labelFieldCandidates)
% LOAD_MULTIVIEW_DATASET 加载并校验多视图数据集。

datasetPath = fullfile(datasetDir, datasetFile);
if exist(datasetPath, 'file') ~= 2
    error('未找到数据集文件：%s。请确认 datasetFile 是否正确。', datasetPath);
end

datasetStruct = load(datasetPath);
if ~isfield(datasetStruct, 'X')
    error('数据集 %s 中缺少变量 X，无法继续运行。', datasetFile);
end

X = datasetStruct.X;
if ~iscell(X) || isempty(X)
    error('数据集 %s 中的 X 必须是非空 cell 数组。', datasetFile);
end
X = X(:);

labelField = resolve_label_field(datasetStruct, labelFieldCandidates, datasetFile);
Y = datasetStruct.(labelField);
Y = double(Y(:));

[X, Y] = validate_multiview_data(X, Y, datasetFile);

datasetInfo = struct();
datasetInfo.datasetFile = datasetFile;
datasetInfo.datasetPath = datasetPath;
datasetInfo.labelField = labelField;
datasetInfo.sampleNum = numel(Y);
datasetInfo.viewNum = numel(X);
datasetInfo.classNum = numel(unique(Y));
end


function validate_search_config(datasetFile, betaList, lambdaList, tauQList, repeatNum, baseSeed, labelFieldCandidates)
% VALIDATE_SEARCH_CONFIG 校验网格搜索配置参数。

if ~(ischar(datasetFile) || (isstring(datasetFile) && isscalar(datasetFile)))
    error('datasetFile 必须是字符向量或字符串标量。');
end
if isempty(betaList) || ~isnumeric(betaList) || any(~isfinite(betaList)) || any(betaList <= 0)
    error('betaList 必须是非空正数向量。');
end
if isempty(lambdaList) || ~isnumeric(lambdaList) || any(~isfinite(lambdaList)) || any(lambdaList <= 0)
    error('lambdaList 必须是非空正数向量。');
end
if isempty(tauQList) || ~isnumeric(tauQList) || any(~isfinite(tauQList)) || any(tauQList <= 0)
    error('tauQList 必须是非空正数向量。');
end
if ~isscalar(repeatNum) || repeatNum < 1 || repeatNum ~= floor(repeatNum)
    error('repeatNum 必须是大于等于 1 的整数。');
end
if ~isscalar(baseSeed) || ~isfinite(baseSeed) || baseSeed ~= floor(baseSeed)
    error('baseSeed 必须是有限整数。');
end
if ~iscell(labelFieldCandidates) || isempty(labelFieldCandidates)
    error('labelFieldCandidates 必须是非空 cell 数组。');
end
end


function labelField = resolve_label_field(datasetStruct, labelFieldCandidates, datasetFile)
% RESOLVE_LABEL_FIELD 自动解析标签字段名。

for i = 1:numel(labelFieldCandidates)
    candidate = labelFieldCandidates{i};
    if isfield(datasetStruct, candidate)
        value = datasetStruct.(candidate);
        if isnumeric(value) && isvector(value) && ~isempty(value)
            labelField = candidate;
            return;
        end
    end
end

fieldList = fieldnames(datasetStruct);
if isfield(datasetStruct, 'X') && iscell(datasetStruct.X) && ~isempty(datasetStruct.X)
    sampleNum = size(datasetStruct.X{1}, 1);
    for i = 1:numel(fieldList)
        currentField = fieldList{i};
        if strcmp(currentField, 'X')
            continue;
        end
        value = datasetStruct.(currentField);
        if isnumeric(value) && isvector(value) && numel(value) == sampleNum && ~isempty(value)
            labelField = currentField;
            return;
        end
    end
end

error(['数据集 %s 中未找到可识别的标签字段。', ...
    '请检查变量名，或将字段名加入 labelFieldCandidates。'], datasetFile);
end


function [X, Y] = validate_multiview_data(X, Y, datasetFile)
% VALIDATE_MULTIVIEW_DATA 对多视图数据执行基本输入检查。

if isempty(Y) || ~isnumeric(Y) || ~isvector(Y)
    error('数据集 %s 中的标签必须是非空数值向量。', datasetFile);
end
if any(~isfinite(Y))
    error('数据集 %s 中的标签包含 NaN 或 Inf，请先检查数据。', datasetFile);
end

sampleNum = numel(Y);
for iv = 1:numel(X)
    Xi = X{iv};
    if isempty(Xi)
        error('数据集 %s 的第 %d 个视图为空。', datasetFile, iv);
    end
    if ~(isnumeric(Xi) || islogical(Xi))
        error('数据集 %s 的第 %d 个视图必须是数值矩阵。', datasetFile, iv);
    end
    if size(Xi, 1) ~= sampleNum
        error(['数据集 %s 的第 %d 个视图样本数与标签长度不一致：', ...
            'size(X{%d},1)=%d, numel(Y)=%d。'], ...
            datasetFile, iv, iv, size(Xi, 1), sampleNum);
    end
    if any(~isfinite(Xi(:)))
        error('数据集 %s 的第 %d 个视图包含 NaN 或 Inf，请先检查数据。', datasetFile, iv);
    end
    X{iv} = double(Xi);
end

if numel(unique(Y)) < 2
    error('数据集 %s 的标签类别数小于 2，无法进行聚类评价。', datasetFile);
end
end


function record = create_empty_record(metricNames, viewNum)
% CREATE_EMPTY_RECORD 创建单次搜索结果结构体模板。

record = struct( ...
    'datasetFile', '', ...
    'labelField', '', ...
    'sampleNum', NaN, ...
    'viewNum', viewNum, ...
    'classNum', NaN, ...
    'beta', NaN, ...
    'lambda', NaN, ...
    'tauQ', NaN, ...
    'repeatId', NaN, ...
    'seed', NaN, ...
    'targetView', NaN, ...
    'iterNum', NaN, ...
    'finalObj', NaN, ...
    'neighborTime', NaN, ...
    'elapsedTime', NaN, ...
    'status', '', ...
    'errorMessage', '', ...
    'qualityScores', nan(1, viewNum), ...
    'qualityScoresText', '', ...
    'viewWeights', nan(1, viewNum), ...
    'viewWeightsText', '');

for i = 1:numel(metricNames)
    record.(metricNames{i}) = NaN;
    record.([metricNames{i}, 'Std']) = NaN;
end
for iv = 1:viewNum
    record.(sprintf('quality_v%d', iv)) = NaN;
    record.(sprintf('weight_v%d', iv)) = NaN;
end
end


function record = build_search_record(datasetInfo, metricNames, viewNum, beta, lambda, tauQ, ...
    repeatId, seed, targetView, iterNum, obj, neighborTime, elapsedTime, qualityScores, ...
    viewWeights, metricMean, metricStd, status, errorMessage)
% BUILD_SEARCH_RECORD 组织单次搜索结果。

record = create_empty_record(metricNames, viewNum);
record.datasetFile = datasetInfo.datasetFile;
record.labelField = datasetInfo.labelField;
record.sampleNum = datasetInfo.sampleNum;
record.classNum = datasetInfo.classNum;
record.beta = beta;
record.lambda = lambda;
record.tauQ = tauQ;
record.repeatId = repeatId;
record.seed = seed;
record.targetView = targetView;
record.iterNum = iterNum;
if isempty(obj)
    record.finalObj = NaN;
else
    record.finalObj = obj(end);
end
record.neighborTime = neighborTime;
record.elapsedTime = elapsedTime;
record.status = status;
record.errorMessage = errorMessage;
record.qualityScores = qualityScores(:)';
record.qualityScoresText = vector_to_text(qualityScores);
record.viewWeights = viewWeights(:)';
record.viewWeightsText = vector_to_text(viewWeights);

for i = 1:numel(metricNames)
    record.(metricNames{i}) = metricMean(i);
    record.([metricNames{i}, 'Std']) = metricStd(i);
end
for iv = 1:viewNum
    record.(sprintf('quality_v%d', iv)) = qualityScores(iv);
    record.(sprintf('weight_v%d', iv)) = viewWeights(iv);
end
end


function combo = create_empty_combo_summary(viewNum)
% CREATE_EMPTY_COMBO_SUMMARY 创建参数组合汇总模板。

combo = struct( ...
    'beta', NaN, ...
    'lambda', NaN, ...
    'tauQ', NaN, ...
    'bestACC', NaN, ...
    'meanACC', NaN, ...
    'stdACC', NaN, ...
    'bestTime', NaN, ...
    'meanTime', NaN, ...
    'meanWeights', nan(1, viewNum), ...
    'meanWeightsText', '');

for iv = 1:viewNum
    combo.(sprintf('mean_weight_v%d', iv)) = NaN;
end
end


function combo = build_combo_summary(combo, beta, lambda, tauQ, comboACC, comboTime, comboWeights)
% BUILD_COMBO_SUMMARY 组织参数组合汇总结果。

[bestACC, meanACC, stdACC] = compute_valid_statistics(comboACC, 'max');
[bestTime, meanTime] = compute_time_statistics(comboTime);
meanWeights = compute_mean_vector(comboWeights);

combo.beta = beta;
combo.lambda = lambda;
combo.tauQ = tauQ;
combo.bestACC = bestACC;
combo.meanACC = meanACC;
combo.stdACC = stdACC;
combo.bestTime = bestTime;
combo.meanTime = meanTime;
combo.meanWeights = meanWeights;
combo.meanWeightsText = vector_to_text(meanWeights);
for iv = 1:numel(meanWeights)
    combo.(sprintf('mean_weight_v%d', iv)) = meanWeights(iv);
end
end


function [bestValue, meanValue, stdValue] = compute_valid_statistics(values, mode)
% COMPUTE_VALID_STATISTICS 仅基于有限值计算统计量。

validValues = values(isfinite(values));
if isempty(validValues)
    bestValue = NaN;
    meanValue = NaN;
    stdValue = NaN;
    return;
end

if strcmp(mode, 'max')
    bestValue = max(validValues);
elseif strcmp(mode, 'min')
    bestValue = min(validValues);
else
    error('mode 必须为 ''max'' 或 ''min''。');
end
meanValue = mean(validValues);
stdValue = std(validValues, 1);
end


function [bestTime, meanTime] = compute_time_statistics(values)
% COMPUTE_TIME_STATISTICS 计算有限时间值的最小值和均值。

[bestTime, meanTime] = compute_valid_statistics(values, 'min');
end


function meanVector = compute_mean_vector(values)
% COMPUTE_MEAN_VECTOR 对每列有限值求均值。

if isempty(values)
    meanVector = [];
    return;
end

meanVector = nan(1, size(values, 2));
for iv = 1:size(values, 2)
    currentValues = values(:, iv);
    currentValues = currentValues(isfinite(currentValues));
    if ~isempty(currentValues)
        meanVector(iv) = mean(currentValues);
    end
end
end


function text = vector_to_text(values)
% VECTOR_TO_TEXT 将向量格式化为日志字符串。

if isempty(values) || all(~isfinite(values(:)))
    text = '[NaN]';
    return;
end

parts = cell(numel(values), 1);
for i = 1:numel(values)
    if isfinite(values(i))
        parts{i} = sprintf('v%d=%.6f', i, values(i));
    else
        parts{i} = sprintf('v%d=NaN', i);
    end
end
text = strjoin(parts, ', ');
end
