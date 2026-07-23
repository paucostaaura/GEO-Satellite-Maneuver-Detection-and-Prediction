%% A05_validate_supervised_LSTM.m
% Untouched chronological test validation of the supervised LSTM detector.
%
% Run A04_train_supervised_LSTM.m first.
% Test labels are read only here, after architecture and threshold are fixed.

%clear; clc; close all;

%% 1) Load frozen LSTM detector

resultsFile = 'supervised_LSTM_results.mat';

if ~isfile(resultsFile)
    error('Results file not found: %s. Run A04 first.',resultsFile);
end

R = load(resultsFile);

required = {'dates','testSequenceDates','testProbability', ...
    'selectedThreshold','eventGapDays','matchingToleranceDays', ...
    'officialGroupingDays','testStart','testEnd', ...
    'usedFeatureNames','lstmModel','selectedSequenceLength', ...
    'selectedHiddenUnits','maxSequenceGapDays'};

for i = 1:numel(required)
    if ~isfield(R,required{i})
        error('Missing variable %s in %s.',required{i},resultsFile);
    end
end

dates = R.dates;
if isempty(dates.TimeZone)
    dates.TimeZone = 'UTC';
end

testDates = R.testSequenceDates;
testProbability = R.testProbability;

fprintf('Loaded frozen LSTM detector from %s\n',resultsFile);


%% 2) Read and group official events

labelFileName = 'ManoeuvreLabel_QZS3.txt';
labelFile = locateFile(labelFileName);
fprintf('Official label file:\n%s\n',labelFile);

officialBurns = readOfficialManeuvers(labelFile);
officialEventsAll = groupOfficialManeuvers( ...
    officialBurns,R.officialGroupingDays);

officialTestEvents = officialEventsAll( ...
    officialEventsAll.event_date >= R.testStart & ...
    officialEventsAll.event_date <= R.testEnd,:);

%% 3) Rebuild detections from the frozen threshold

testDetectedEvents = buildProbabilityEvents( ...
    testDates,testProbability,R.selectedThreshold,R.eventGapDays);

%% 4) One-to-one event matching

[matchedDetectedIdx,matchedOfficialIdx,timingErrorsDays] = ...
    matchManeuverEvents( ...
        testDetectedEvents.PeakDate, ...
        officialTestEvents.event_date, ...
        R.matchingToleranceDays);

TP = numel(matchedDetectedIdx);
FN = height(officialTestEvents)-TP;
rawFP = height(testDetectedEvents)-TP;

%% 5) Recall by grouped event composition

matchedOfficialMask = false(height(officialTestEvents),1);
matchedOfficialMask(matchedOfficialIdx) = true;

officialTypes = upper(string(officialTestEvents.type));

isEWOnly = officialTypes == "EW";
isNSOnly = officialTypes == "NS";
isCombined = officialTypes == "EW+NS";

TP_EW = sum(matchedOfficialMask & isEWOnly);
FN_EW = sum(~matchedOfficialMask & isEWOnly);
recallEW = safeDivide(TP_EW,TP_EW+FN_EW);

TP_NS = sum(matchedOfficialMask & isNSOnly);
FN_NS = sum(~matchedOfficialMask & isNSOnly);
recallNS = safeDivide(TP_NS,TP_NS+FN_NS);

TP_combined = sum(matchedOfficialMask & isCombined);
FN_combined = sum(~matchedOfficialMask & isCombined);
recallCombined = safeDivide( ...
    TP_combined,TP_combined+FN_combined);

%% 6) Frequency and timing metrics

observationYears = max(years(max(testDates)-min(testDates)),eps);
officialEventsPerYear = height(officialTestEvents)/observationYears;
detectedEventsPerYear = height(testDetectedEvents)/observationYears;
frequencyErrorPerYear = detectedEventsPerYear-officialEventsPerYear;
relativeFrequencyError = safeDivide( ...
    frequencyErrorPerYear,officialEventsPerYear);
rawFalseAlarmsPerYear = rawFP/observationYears;

if isempty(timingErrorsDays)
    meanTimingErrorDays = NaN;
    meanAbsTimingErrorDays = NaN;
    medianAbsTimingErrorDays = NaN;
    rmseTimingDays = NaN;
    maxAbsTimingErrorDays = NaN;
else
    meanTimingErrorDays = mean(timingErrorsDays,'omitnan');
    meanAbsTimingErrorDays = mean(abs(timingErrorsDays),'omitnan');
    medianAbsTimingErrorDays = median(abs(timingErrorsDays),'omitnan');
    rmseTimingDays = sqrt(mean(timingErrorsDays.^2,'omitnan'));
    maxAbsTimingErrorDays = max(abs(timingErrorsDays));
end

%% 7) Comparison tables

detectedComparison = testDetectedEvents;
detectedComparison.IsTruePositive = false(height(testDetectedEvents),1);
detectedComparison.MatchedOfficialEventID = NaN(height(testDetectedEvents),1);
detectedComparison.TimingErrorDays = NaN(height(testDetectedEvents),1);

for i = 1:numel(matchedDetectedIdx)
    d = matchedDetectedIdx(i);
    o = matchedOfficialIdx(i);
    detectedComparison.IsTruePositive(d) = true;
    detectedComparison.MatchedOfficialEventID(d) = ...
        officialTestEvents.EventID(o);
    detectedComparison.TimingErrorDays(d) = timingErrorsDays(i);
end

officialComparison = officialTestEvents;
officialComparison.WasDetected = matchedOfficialMask;
officialComparison.MatchedDetectedEventID = ...
    NaN(height(officialTestEvents),1);

for i = 1:numel(matchedDetectedIdx)
    d = matchedDetectedIdx(i);
    o = matchedOfficialIdx(i);
    officialComparison.MatchedDetectedEventID(o) = ...
        testDetectedEvents.EventID(d);
end

missedOfficialEvents = officialComparison(~officialComparison.WasDetected,:);
%% 7b) Separate duplicate detections from genuine false positives

unmatchedDetectedMask = ~detectedComparison.IsTruePositive;

duplicateDetectedMask = false(height(detectedComparison),1);
nearestOfficialEventID = NaN(height(detectedComparison),1);
nearestOfficialDate = NaT(height(detectedComparison),1,'TimeZone','UTC');
distanceToNearestOfficialDays = NaN(height(detectedComparison),1);

for d = find(unmatchedDetectedMask)'

    deltaDays = days( ...
        detectedComparison.PeakDate(d) - officialTestEvents.event_date);

    [minimumDistance,nearestOfficialIdx] = min(abs(deltaDays));

    nearestOfficialEventID(d) = ...
        officialTestEvents.EventID(nearestOfficialIdx);

    nearestOfficialDate(d) = ...
        officialTestEvents.event_date(nearestOfficialIdx);

    distanceToNearestOfficialDays(d) = ...
        deltaDays(nearestOfficialIdx);

    % An unmatched detection inside the matching tolerance is a duplicate
    % when that official event has already been assigned to another detection.
    if minimumDistance <= R.matchingToleranceDays && ...
            matchedOfficialMask(nearestOfficialIdx)

        duplicateDetectedMask(d) = true;
    end
end

detectedComparison.IsDuplicateDetection = duplicateDetectedMask;
detectedComparison.NearestOfficialEventID = nearestOfficialEventID;
detectedComparison.NearestOfficialDate = nearestOfficialDate;
detectedComparison.DistanceToNearestOfficialDays = ...
    distanceToNearestOfficialDays;

duplicateDetectionEvents = detectedComparison( ...
    duplicateDetectedMask,:);

genuineFalsePositiveEvents = detectedComparison( ...
    unmatchedDetectedMask & ~duplicateDetectedMask,:);

% Keep this name for compatibility with the rest of the script
falsePositiveEvents = genuineFalsePositiveEvents;

numDuplicateDetections = height(duplicateDetectionEvents);
FP = height(genuineFalsePositiveEvents);
falseAlarmsPerYear = FP/observationYears;

precision = safeDivide(TP,TP+FP);
recall = safeDivide(TP,TP+FN);
f1Score = safeDivide(2*precision*recall,precision+recall);
eventAccuracy = safeDivide(TP,TP+FP+FN);
falseDiscoveryRate = safeDivide(FP,TP+FP);
missRate = safeDivide(FN,TP+FN);
%% 8) Print report

fprintf('\n============================================================\n');
fprintf(' SUPERVISED LSTM TEST VALIDATION\n');
fprintf('============================================================\n');
fprintf('Test period represented by valid sequence endpoints:\n');
fprintf('  Start: %s\n',char(min(testDates)));
fprintf('  End:   %s\n',char(max(testDates)));
fprintf('  Duration: %.3f years\n',observationYears);

fprintf('\nFrozen model configuration:\n');
fprintf('  Sequence length:       %d epochs\n',R.selectedSequenceLength);
fprintf('  Hidden units:          %d\n',R.selectedHiddenUnits);
fprintf('  Maximum sequence gap:  %.2f days\n',R.maxSequenceGapDays);
fprintf('  Probability threshold: %.3f\n',R.selectedThreshold);
fprintf('  Detection grouping gap: %.2f days\n',R.eventGapDays);
fprintf('  Matching tolerance:     %.2f days\n',R.matchingToleranceDays);
fprintf('  Official grouping gap:  %.2f days\n',R.officialGroupingDays);
fprintf('  Number of features:     %d\n',numel(R.usedFeatureNames));

fprintf('\nEvent counts:\n');
fprintf('  Official grouped events: %d\n',height(officialTestEvents));
fprintf('  Detected grouped events: %d\n',height(testDetectedEvents));

fprintf('\nConfusion counts:\n');
fprintf('  TP: %d\n',TP);
fprintf('  FP: %d\n',FP);
fprintf('  FN: %d\n',FN);

fprintf('\nEvent-level metrics:\n');
fprintf('  Precision:            %.4f\n',precision);
fprintf('  Recall:               %.4f\n',recall);
fprintf('  F1 score:             %.4f\n',f1Score);
fprintf('  Event accuracy:       %.4f\n',eventAccuracy);
fprintf('  False discovery rate: %.4f\n',falseDiscoveryRate);
fprintf('  Miss rate:            %.4f\n',missRate);

fprintf('\nRecall by grouped event composition:\n');
fprintf('  EW-only recall: %.4f (%d TP, %d FN)\n', ...
    recallEW,TP_EW,FN_EW);
fprintf('  NS-only recall: %.4f (%d TP, %d FN)\n', ...
    recallNS,TP_NS,FN_NS);
fprintf('  EW+NS recall:   %.4f (%d TP, %d FN)\n', ...
    recallCombined,TP_combined,FN_combined);

fprintf('\nFrequency metrics:\n');
fprintf('  Official frequency:       %.2f events/year\n', ...
    officialEventsPerYear);
fprintf('  Detected frequency:       %.2f events/year\n', ...
    detectedEventsPerYear);
fprintf('  Frequency error:          %+.2f events/year\n', ...
    frequencyErrorPerYear);
fprintf('  Relative frequency error: %+.2f %%\n', ...
    100*relativeFrequencyError);
fprintf('  False alarms/year:        %.2f\n',falseAlarmsPerYear);

fprintf('\nTiming metrics:\n');
fprintf('  Mean signed error:     %+.3f days\n',meanTimingErrorDays);
fprintf('  Mean absolute error:    %.3f days\n',meanAbsTimingErrorDays);
fprintf('  Median absolute error:  %.3f days\n',medianAbsTimingErrorDays);
fprintf('  RMSE:                   %.3f days\n',rmseTimingDays);
fprintf('  Maximum absolute error: %.3f days\n',maxAbsTimingErrorDays);
fprintf('============================================================\n');


testPredictions = table( ...
    testDates, ...
    testProbability, ...
    testProbability >= R.selectedThreshold, ...
    'VariableNames',{'Epoch','Probability','PredictedLabel'});

predictedManoeuvres = testPredictions(testPredictions.PredictedLabel,:);

fprintf('\n====================================================\n');
fprintf('ALL TEST EPOCHS PREDICTED AS MANOEUVRE\n');
fprintf('====================================================\n');

disp(predictedManoeuvres)

fprintf('\n====================================================\n');
fprintf('FALSE POSITIVE EVENTS\n');
fprintf('====================================================\n');

if isempty(falsePositiveEvents)
    fprintf('No false positives.\n');
else
    disp(falsePositiveEvents(:,{ ...
        'EventID','StartDate','EndDate','PeakDate', ...
        'NumPositiveEpochs','PeakProbability'}))
end

fprintf('\n====================================================\n');
fprintf('FALSE POSITIVES AND NEAREST OFFICIAL EVENT\n');
fprintf('====================================================\n');

fpNearestOfficial = falsePositiveEvents;

fpNearestOfficial.NearestOfficialDate = ...
    NaT(height(falsePositiveEvents),1,'TimeZone','UTC');

fpNearestOfficial.DistanceToOfficialDays = ...
    NaN(height(falsePositiveEvents),1);

fpNearestOfficial.NearestOfficialType = ...
    strings(height(falsePositiveEvents),1);

fpNearestOfficial.OfficialWasMatched = ...
    false(height(falsePositiveEvents),1);

for k = 1:height(falsePositiveEvents)

    fpDate = falsePositiveEvents.PeakDate(k);

    deltaDays = days(fpDate-officialTestEvents.event_date);

    [minDistance,idxOfficial] = min(abs(deltaDays));

    fpNearestOfficial.NearestOfficialDate(k) = ...
        officialTestEvents.event_date(idxOfficial);

    fpNearestOfficial.DistanceToOfficialDays(k) = ...
        deltaDays(idxOfficial);

    fpNearestOfficial.NearestOfficialType(k) = ...
        string(officialTestEvents.type(idxOfficial));

    fpNearestOfficial.OfficialWasMatched(k) = ...
        matchedOfficialMask(idxOfficial);

end

disp(fpNearestOfficial(:,{ ...
    'EventID', ...
    'StartDate', ...
    'EndDate', ...
    'PeakDate', ...
    'PeakProbability', ...
    'NearestOfficialDate', ...
    'DistanceToOfficialDays', ...
    'NearestOfficialType', ...
    'OfficialWasMatched'}));

writetable(predictedManoeuvres,...
    'LSTM_test_predicted_manoeuvre_epochs.csv');

writetable(testPredictions,'LSTM_test_predictions.csv');

%% 9) Save outputs

validationMetrics = table( ...
    TP,FP,FN,precision,recall,f1Score,eventAccuracy, ...
    falseDiscoveryRate,missRate, ...
    TP_EW,FN_EW,recallEW, ...
    TP_NS,FN_NS,recallNS, ...
    TP_combined,FN_combined,recallCombined, ...
    officialEventsPerYear,detectedEventsPerYear, ...
    frequencyErrorPerYear,relativeFrequencyError,falseAlarmsPerYear, ...
    meanTimingErrorDays,meanAbsTimingErrorDays, ...
    medianAbsTimingErrorDays,rmseTimingDays,maxAbsTimingErrorDays);

disp(validationMetrics);

save('supervised_LSTM_validation_results.mat', ...
    'validationMetrics','officialTestEvents','testDetectedEvents', ...
    'detectedComparison','officialComparison', ...
    'missedOfficialEvents','falsePositiveEvents', ...
    'matchedDetectedIdx','matchedOfficialIdx','timingErrorsDays');

writetable(validationMetrics,'LSTM_test_validation_metrics.csv');
writetable(detectedComparison,'LSTM_test_detection_comparison.csv');
writetable(officialComparison,'LSTM_test_official_comparison.csv');
writetable(missedOfficialEvents,'LSTM_test_missed_events.csv');
writetable(falsePositiveEvents,'LSTM_test_false_positive_events.csv');

%% 10) Plots

figure('Color','w','Name','LSTM test probabilities');
plot(testDates,testProbability,'k','LineWidth',1);
hold on;
yline(R.selectedThreshold,'--','Threshold','LineWidth',1.1);

if ~isempty(testDetectedEvents)
    scatter(testDetectedEvents.PeakDate, ...
        testDetectedEvents.PeakProbability,35,'filled');
end

for i = 1:height(officialTestEvents)
    xline(officialTestEvents.event_date(i),':');
end

grid on; box on;
xlabel('Epoch');
ylabel('P(manoeuvre)');
title('LSTM untouched 2024-2025 test period');

figure('Color','w','Name','LSTM test timing errors');

if isempty(timingErrorsDays)
    text(0.5,0.5,'No matched events','HorizontalAlignment','center');
    axis off;
else
    histogram(timingErrorsDays);
    grid on; box on;
    xlabel('Detected date - official date [days]');
    ylabel('Number of matched events');
    title('LSTM test timing error distribution');
end

fprintf('\nLSTM validation results saved.\n');

%% =========================================================
% LOCAL FUNCTIONS
% ==========================================================

function filePath = locateFile(fileName)
    filePath = which(fileName);
    if isempty(filePath)
        matches = dir(fullfile(pwd,'**',fileName));
        if isempty(matches)
            error('File not found: %s',fileName);
        end
        filePath = fullfile(matches(1).folder,matches(1).name);
    end
end

function burns = readOfficialManeuvers(filePath)
    lines = readlines(filePath);
    startDate = NaT(0,1,'TimeZone','UTC');
    endDate = NaT(0,1,'TimeZone','UTC');
    manoeuvreType = strings(0,1);

    for i = 1:numel(lines)
        line = strtrim(lines(i));
        if strlength(line)==0 || startsWith(line,"#")
            continue
        end

        fields = split(line,",");
        if numel(fields) < 4
            continue
        end

        try
            t0 = datetime(strtrim(fields(1)), ...
                'InputFormat','yyyy-MM-dd HH:mm:ss','TimeZone','UTC');
            t1 = datetime(strtrim(fields(2)), ...
                'InputFormat','yyyy-MM-dd HH:mm:ss','TimeZone','UTC');
        catch
            continue
        end

        rawType = upper(strtrim(fields(4)));
        if startsWith(rawType,"EW")
            type = "EW";
        elseif startsWith(rawType,"NS")
            type = "NS";
        else
            continue
        end

        startDate(end+1,1) = t0; %#ok<AGROW>
        endDate(end+1,1) = t1; %#ok<AGROW>
        manoeuvreType(end+1,1) = type; %#ok<AGROW>
    end

    burns = table(startDate,endDate,manoeuvreType, ...
        'VariableNames',{'start_date','end_date','type'});
    burns = sortrows(burns,'start_date');
end

function events = groupOfficialManeuvers(burns,gapDays)
    if isempty(burns)
        events = table();
        return
    end

    burns = sortrows(burns,'start_date');
    eventID = zeros(height(burns),1);
    currentEvent = 1;
    eventID(1) = currentEvent;

    for i = 2:height(burns)
        temporalGapDays = days( ...
            burns.start_date(i)-burns.end_date(i-1));

        if temporalGapDays > gapDays
            currentEvent = currentEvent+1;
        end

        eventID(i) = currentEvent;
    end

    nEvents = max(eventID);
    EventID = (1:nEvents)';
    event_date = NaT(nEvents,1,'TimeZone','UTC');
    start_date = NaT(nEvents,1,'TimeZone','UTC');
    end_date = NaT(nEvents,1,'TimeZone','UTC');
    type = strings(nEvents,1);
    NumBurns = zeros(nEvents,1);
    NumEW = zeros(nEvents,1);
    NumNS = zeros(nEvents,1);

    for e = 1:nEvents
        idx = eventID == e;
        start_date(e) = min(burns.start_date(idx));
        end_date(e) = max(burns.end_date(idx));
        event_date(e) = start_date(e);
        NumBurns(e) = sum(idx);
        NumEW(e) = sum(burns.type(idx) == "EW");
        NumNS(e) = sum(burns.type(idx) == "NS");

        if NumEW(e)>0 && NumNS(e)>0
            type(e) = "EW+NS";
        elseif NumEW(e)>0
            type(e) = "EW";
        else
            type(e) = "NS";
        end
    end

    events = table(EventID,event_date,start_date,end_date,type, ...
        NumBurns,NumEW,NumNS);
end

function eventTable = buildProbabilityEvents(dates,p,threshold,gapDays)
    idx = find(isfinite(p) & p >= threshold);

    if isempty(idx)
        eventTable = table( ...
            zeros(0,1),NaT(0,1,'TimeZone','UTC'), ...
            NaT(0,1,'TimeZone','UTC'),NaT(0,1,'TimeZone','UTC'), ...
            zeros(0,1),zeros(0,1), ...
            'VariableNames',{'EventID','StartDate','EndDate','PeakDate', ...
            'NumPositiveEpochs','PeakProbability'});
        return
    end

    candidateDates = dates(idx);
    newEvent = [true; days(diff(candidateDates)) > gapDays];
    pointEventID = cumsum(newEvent);
    nEvents = max(pointEventID);

    EventID = (1:nEvents)';
    StartDate = NaT(nEvents,1,'TimeZone','UTC');
    EndDate = NaT(nEvents,1,'TimeZone','UTC');
    PeakDate = NaT(nEvents,1,'TimeZone','UTC');
    NumPositiveEpochs = zeros(nEvents,1);
    PeakProbability = zeros(nEvents,1);

    for e = 1:nEvents
        local = pointEventID == e;
        globalIdx = idx(local);
        localP = p(globalIdx);

        StartDate(e) = dates(globalIdx(1));
        EndDate(e) = dates(globalIdx(end));
        NumPositiveEpochs(e) = numel(globalIdx);
        [PeakProbability(e),k] = max(localP);
        PeakDate(e) = dates(globalIdx(k));
    end

    eventTable = table(EventID,StartDate,EndDate,PeakDate, ...
        NumPositiveEpochs,PeakProbability);
end

function [matchedDetected,matchedOfficial,timingErrors] = ...
        matchManeuverEvents(detectedDates,officialDates,toleranceDays)

    matchedDetected = zeros(0,1);
    matchedOfficial = zeros(0,1);
    timingErrors = zeros(0,1);

    if isempty(detectedDates) || isempty(officialDates)
        return
    end

    pairs = zeros(0,3);

    for d = 1:numel(detectedDates)
        delta = days(detectedDates(d)-officialDates);
        valid = find(abs(delta) <= toleranceDays);

        for j = 1:numel(valid)
            pairs(end+1,:) = [abs(delta(valid(j))),d,valid(j)]; %#ok<AGROW>
        end
    end

    if isempty(pairs)
        return
    end

    pairs = sortrows(pairs,1);
    usedD = false(numel(detectedDates),1);
    usedO = false(numel(officialDates),1);

    for i = 1:size(pairs,1)
        d = pairs(i,2);
        o = pairs(i,3);

        if ~usedD(d) && ~usedO(o)
            usedD(d) = true;
            usedO(o) = true;
            matchedDetected(end+1,1) = d; %#ok<AGROW>
            matchedOfficial(end+1,1) = o; %#ok<AGROW>
            timingErrors(end+1,1) = ...
                days(detectedDates(d)-officialDates(o)); %#ok<AGROW>
        end
    end
end

function value = safeDivide(a,b)
    if b == 0
        value = NaN;
    else
        value = a/b;
    end
end
