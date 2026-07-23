clear; clc; close all;

inputFile = 'tle_raw_variables_ID_42917.mat';

load(inputFile);

%% Extract NORAD ID from filename

tokens = regexp(inputFile,'\d+','match');

if isempty(tokens)
    error('No NORAD ID found in input filename.');
end

NORAD_ID = tokens{1};

mu = 398600.4418; % km^3/s^2

nTLE = length(epoch_datetime);

%% Basic features

n_rad_s = n_rev_day * 2*pi / 86400;
a_km = (mu ./ n_rad_s.^2).^(1/3); % Semimajor axis (a)
T_orbit_day = 1 ./ n_rev_day;          % orbital period [days]
T_orbit_h   = T_orbit_day * 24;        % orbital period [hours]
dT_orbit_h  = [NaN; abs(diff(T_orbit_h))]; % Quizás después de hzcer matriz de correlación lo descarto por similitud con n

dinc_deg = [NaN; diff(inc_deg)];
dRAAN_deg = [NaN; angleDiffDeg(RAAN_deg)];
decc = [NaN; diff(ecc)];
dn_rev_day = [NaN; diff(n_rev_day)];
da_km = [NaN; diff(a_km)];

n_geo_rev_day = 1.002737909;   % Earth sidereal rotation [rev/day]

lambda_drift_deg_day = 360 * (n_rev_day - n_geo_rev_day);
dlambda_drift_deg_day = [NaN; diff(lambda_drift_deg_day)];

abs_lambda_drift_deg_day = abs(lambda_drift_deg_day); % Los abs son los que guardo (al modelo le interesa si hay un cambio de tendencia/valor abs)
abs_dlambda_drift_deg_day = abs(dlambda_drift_deg_day);


% Absolute basic variations (las que guardaremos)

dinc_deg = abs(dinc_deg);
dRAAN_deg = abs(dRAAN_deg);
decc = abs(decc);
dn_rev_day = abs(dn_rev_day);
da_km = abs(da_km);

% Second-order differences. Cuanto cambia el cambio (aceleración). Muy útil
% para maniobras impulsivas/bruscas
d2a_km = [NaN; abs(diff(da_km))];
d2inc_deg = [NaN; abs(diff(dinc_deg))];
d2n_rev_day = [NaN; abs(diff(dn_rev_day))];
d2lambda_drift_deg_day = [NaN; abs(diff(abs_dlambda_drift_deg_day))];
d2ecc = [NaN; abs(diff(decc))];


%% Derivadas temporales

min_dt_days = 0.05;    % only for derivatives (~1.2 h)

safe_dt = dt_days;
safe_dt(dt_days < min_dt_days) = NaN;

ma_km_day   = da_km ./ safe_dt;      % da/dt [km/day]
mi_deg_day  = dinc_deg ./ safe_dt;   % di/dt [deg/day]
mn_rev_day2 = dn_rev_day ./ safe_dt; % dn/dt [(rev/day)/day]
mecc_day = decc ./ safe_dt;   
%% SDP4 Propagation (deep-space propagator, not sgp4, which is more for LEO, not GEO)

opsmode = 'a';      % AFSPC mode
whichconst = 72;    % WGS-72 constants, standard for TLE/SGP4


res_pos_norm_km = NaN(nTLE,1);
res_vel_norm_kms = NaN(nTLE,1);

res_R_km = NaN(nTLE,1);
res_T_km = NaN(nTLE,1);
res_N_km = NaN(nTLE,1);

res_a_km = NaN(nTLE,1);
res_e = NaN(nTLE,1);
res_i_deg = NaN(nTLE,1);
res_RAAN_deg = NaN(nTLE,1);
res_argp_deg = NaN(nTLE,1);
res_n_rev_day = NaN(nTLE,1);

for k = 1:nTLE-1


    L1_k = char(tle_line1(k));
    L2_k = char(tle_line2(k));

    L1_next = char(tle_line1(k+1));
    L2_next = char(tle_line2(k+1));

    dt_min = minutes(epoch_datetime(k+1) - epoch_datetime(k));

    % Initialise Vallado SGP4/SDP4 structures
    satrec_k    = twoline2rv(L1_k,    L2_k,    opsmode, whichconst);
    satrec_next = twoline2rv(L1_next, L2_next, opsmode, whichconst);

    % Propagate TLE k to epoch of TLE k+1
    [satrec_k, r_prop, v_prop] = sgp4(satrec_k, dt_min); % r y v en k

    % Evaluate TLE k+1 at its own epoch
    [satrec_next, r_obs, v_obs] = sgp4(satrec_next, 0); % r y v en k+1

    % Skip failed propagations
    if satrec_k.error ~= 0 || satrec_next.error ~= 0
        continue
    end

    r_prop = r_prop(:)';
    v_prop = v_prop(:)';

    r_obs = r_obs(:)';
    v_obs = v_obs(:)';

    dr = r_obs - r_prop;
    dv = v_obs - v_prop;

    res_pos_norm_km(k+1) = norm(dr);
    res_vel_norm_kms(k+1) = norm(dv);

    % RTN residuals
    dr_rtn = eci2rtn_residual(r_obs, v_obs, dr);

    res_R_km(k+1) = dr_rtn(1); % Radial residual
    res_T_km(k+1) = dr_rtn(2); % Along-track residual
    res_N_km(k+1) = dr_rtn(3); % Cross-track residual

    % Orbital-element residuals
    coe_prop = rv2coe_simple(r_prop, v_prop, mu);
    coe_obs  = rv2coe_simple(r_obs,  v_obs,  mu);

    res_a_km(k+1) = coe_obs.a_km - coe_prop.a_km;
    res_e(k+1) = coe_obs.ecc - coe_prop.ecc;
    res_i_deg(k+1) = coe_obs.inc_deg - coe_prop.inc_deg;
    res_RAAN_deg(k+1) = wrapTo180(coe_obs.RAAN_deg - coe_prop.RAAN_deg);
    res_argp_deg(k+1) = wrapTo180(coe_obs.argp_deg - coe_prop.argp_deg);

    n_obs_rev_day  = sqrt(mu / coe_obs.a_km^3)  * 86400/(2*pi);
    n_prop_rev_day = sqrt(mu / coe_prop.a_km^3) * 86400/(2*pi);

    res_n_rev_day(k+1) = n_obs_rev_day - n_prop_rev_day;

end

% Time-normalised residual
res_norm_dt_km_day = res_pos_norm_km ./ safe_dt;
res_R_dt_km_day = abs(res_R_km) ./ safe_dt;
res_T_dt_km_day = abs(res_T_km) ./ safe_dt;
res_N_dt_km_day = abs(res_N_km) ./ safe_dt;

%% Descomentar y ejecutar para explicar en el report si los picos de res_T
% coinciden con una variacion de angulo n (o algo así)

% idxT = find(abs(res_T_km) > 500);
% 
% debug_T = table( ...
%     epoch_datetime(idxT), ...
%     dt_days(idxT), ...
%     res_T_km(idxT), ...
%     res_R_km(idxT), ...
%     res_N_km(idxT), ...
%     res_a_km(idxT), ...
%     res_n_rev_day(idxT), ...
%     da_km(idxT), ...
%     n_rev_day(idxT), ...
%     'VariableNames', {'epoch','dt_days','res_T','res_R','res_N','res_a','res_n','da','n_rev_day'});
% 
% disp(debug_T)

%% Derived residual features

res_energy = sqrt(res_R_km.^2 + res_T_km.^2 + res_N_km.^2); % magnitud del error de posición (Residual norm). Se calcula para luego calcular res_energy_roll

W = 40; % window length. Optimizar entrenando el modelo y seleccionar el W que maximice las metricas del isolation forest (mayor F1)

res_energy_roll = movsum(res_energy,[W-1 0],'omitnan');
res_cumulative_roll = movsum(res_pos_norm_km,[W-1 0],'omitnan'); % Cumulative residual magnitude within a moving window

res_slope_km_day = [NaN; diff(res_pos_norm_km)] ./ safe_dt; % mide la velocidad a la que crece el error (low-thrust normalmente produce error creciente)
res_T_slope_km_day = abs([NaN; diff(res_T_km)] ./ safe_dt); % dT/dt en valor abs
res_N_slope_km_day = abs([NaN; diff(res_N_km)] ./ safe_dt); % dN/dt en valor abs

lowthrust_persistence = movmean(abs(res_slope_km_day),[W-1 0],'omitnan');

%% Rolling statistics and local anomaly indicators

roll_mean_resT = movmean(res_T_km,[W-1 0],'omitnan');
roll_std_resT  = movstd(res_T_km,[W-1 0],'omitnan');

roll_mean_resN = movmean(res_N_km,[W-1 0],'omitnan');
roll_std_resN  = movstd(res_N_km,[W-1 0],'omitnan');

roll_mean_pos = movmean(res_pos_norm_km,[W-1 0],'omitnan');
roll_std_pos  = movstd(res_pos_norm_km,[W-1 0],'omitnan');

z_resT = (res_T_km - roll_mean_resT) ./ roll_std_resT;
z_resN = (res_N_km - roll_mean_resN) ./ roll_std_resN;
z_pos  = (res_pos_norm_km - roll_mean_pos) ./ roll_std_pos;

z_resT(~isfinite(z_resT)) = NaN;
z_resN(~isfinite(z_resN)) = NaN;
z_pos(~isfinite(z_pos)) = NaN;

%% Chi-square-like normalized residual indicator

roll_mean_resR = movmean(res_R_km,[W-1 0],'omitnan');
roll_std_resR  = movstd(res_R_km,[W-1 0],'omitnan');

z_resR = (res_R_km - roll_mean_resR) ./ roll_std_resR;

z_resR(~isfinite(z_resR)) = NaN;

chi2_RTN = z_resR.^2 + z_resT.^2 + z_resN.^2; % útil porque combina anomalías en R/T/N en un solo indicador
chi2_RTN(~isfinite(chi2_RTN)) = NaN;

%% Residual direction change angle
% - Valor bajo: el residual sigue apuntando en dirección parecida
% - Valor alto: el residual cambia bruscamente de dirección
% Para low-thrust suele ser más suave; para impulsivas puede cambiar mucho

res_direction_change_deg = NaN(nTLE,1);
WindowAngle = 20;

for k = 2:nTLE

    r_prev = [res_R_km(k-1), res_T_km(k-1), res_N_km(k-1)];
    r_curr = [res_R_km(k),   res_T_km(k),   res_N_km(k)];

    if all(isfinite(r_prev)) && all(isfinite(r_curr)) && norm(r_prev) > 0 && norm(r_curr) > 0

        cos_theta = dot(r_curr,r_prev) / (norm(r_curr)*norm(r_prev));
        cos_theta = max(min(cos_theta,1),-1);

        res_direction_change_deg(k) = acosd(cos_theta);
        direction_change_roll = movmedian(res_direction_change_deg,[WindowAngle-1 0],'omitnan');

    end

end

%% Indicators (EW indicator function and NS indicator function)
EW_indicator = abs(res_T_km) + 100*abs(res_a_km) + 1e6*abs(res_n_rev_day) + 1000*abs_dlambda_drift_deg_day; %actividad East-West, dominada por along-track, semi-major axis, mean motion y drift longitudinal

NS_indicator = abs(res_N_km) + 1000*abs(res_i_deg) + 1000*dinc_deg + 100*dRAAN_deg; %actividad North-South, dominada por cross-track, inclinación y RAAN

control_asymmetry = EW_indicator ./ NS_indicator;
control_asymmetry(~isfinite(control_asymmetry)) = NaN;

EW_NS_ratio = EW_indicator ./ (NS_indicator+eps);
TN_ratio = abs(res_T_km)./(abs(res_N_km)+eps); % ayuda a distinguir entre NS o EW junto con EW_NS ratio

%% FFT-based rolling descriptors
% These features capture periodic/local frequency content
% They are exploratory and should not necessarily be used in the first ML model

fftWindow = 64;        % number of TLE samples in each rolling FFT window
fftMinSamples = 32;    % minimum valid samples required inside window

% Main residual frequency descriptors
[fft_dom_freq_res_pos, fft_dom_period_res_pos, fft_dom_amp_res_pos, ...
    fft_highE_ratio_res_pos, fft_entropy_res_pos] = ...
    rollingFFTdescriptors(res_pos_norm_km, dt_days, fftWindow, fftMinSamples);

[fft_dom_freq_res_T, fft_dom_period_res_T, fft_dom_amp_res_T, ...
    fft_highE_ratio_res_T, fft_entropy_res_T] = ...
    rollingFFTdescriptors(res_T_km, dt_days, fftWindow, fftMinSamples);

[fft_dom_freq_res_N, fft_dom_period_res_N, fft_dom_amp_res_N, ...
    fft_highE_ratio_res_N, fft_entropy_res_N] = ...
    rollingFFTdescriptors(res_N_km, dt_days, fftWindow, fftMinSamples);

% Orbital-element exploratory descriptors
[fft_dom_freq_a, fft_dom_period_a, fft_dom_amp_a, ...
    fft_highE_ratio_a, fft_entropy_a] = ...
    rollingFFTdescriptors(a_km, dt_days, fftWindow, fftMinSamples);

[fft_dom_freq_ecc, fft_dom_period_ecc, fft_dom_amp_ecc, ...
    fft_highE_ratio_ecc, fft_entropy_ecc] = ...
    rollingFFTdescriptors(ecc, dt_days, fftWindow, fftMinSamples);
%% Sanity check

% valid_for_ml = valid_sgp4_step;
% 
% valid_for_ml = valid_for_ml & ...
%     abs(res_pos_norm_km) < 5000 & ...
%     abs(res_R_km) < 1000 & ...
%     abs(res_T_km) < 3000 & ...
%     abs(res_N_km) < 100 & ...
%     abs(res_a_km) < 100 & ...
%     abs(res_i_deg) < 0.1 & ...
%     abs(res_n_rev_day) < 0.01;

%% Valid rows for ML

%% Valid rows for ML

valid_for_ml = ...
    isfinite(a_km) & ...
    isfinite(dT_orbit_h) & ...
    isfinite(dinc_deg) & ...
    isfinite(dRAAN_deg) & ...
    isfinite(decc) & ...
    isfinite(ecc) & ...
    isfinite(d2ecc) & ...
    isfinite(mecc_day) & ...
    isfinite(dn_rev_day) & ...
    isfinite(da_km) & ...
    isfinite(d2a_km) & ...
    isfinite(d2inc_deg) & ...
    isfinite(d2n_rev_day) & ...
    isfinite(abs_lambda_drift_deg_day) & ...
    isfinite(abs_dlambda_drift_deg_day) & ...
    isfinite(d2lambda_drift_deg_day) & ...
    isfinite(ma_km_day) & ...
    isfinite(mi_deg_day) & ...
    isfinite(mn_rev_day2) & ...
    isfinite(res_pos_norm_km) & ...
    isfinite(res_vel_norm_kms) & ...
    isfinite(res_R_km) & ...
    isfinite(res_T_km) & ...
    isfinite(res_N_km) & ...
    isfinite(res_norm_dt_km_day) & ...
    isfinite(res_R_dt_km_day) & ...
    isfinite(res_T_dt_km_day) & ...
    isfinite(res_N_dt_km_day) & ...
    isfinite(res_a_km) & ...
    isfinite(res_e) & ...
    isfinite(res_i_deg) & ...
    isfinite(res_RAAN_deg) & ...
    isfinite(res_argp_deg) & ...
    isfinite(res_n_rev_day) & ...
    isfinite(res_energy) & ...
    isfinite(res_energy_roll) & ...
    isfinite(res_cumulative_roll) & ...
    isfinite(res_slope_km_day) & ...
    isfinite(res_T_slope_km_day) & ...
    isfinite(res_N_slope_km_day) & ...
    isfinite(lowthrust_persistence) & ...
    isfinite(roll_std_resR) & ...
    isfinite(roll_std_resT) & ...
    isfinite(roll_std_resN) & ...
    isfinite(z_resR) & ...
    isfinite(z_resT) & ...
    isfinite(z_resN) & ...
    isfinite(z_pos) & ...
    isfinite(chi2_RTN) & ...
    isfinite(direction_change_roll) & ...
    isfinite(EW_indicator) & ...
    isfinite(NS_indicator) & ...
    isfinite(EW_NS_ratio) & ...
    isfinite(TN_ratio) & ...
    isfinite(fft_dom_freq_res_pos) & ...
    isfinite(fft_dom_period_res_pos) & ...
    isfinite(fft_dom_amp_res_pos) & ...
    isfinite(fft_highE_ratio_res_pos) & ...
    isfinite(fft_entropy_res_pos) & ...
    isfinite(fft_dom_freq_res_T) & ...
    isfinite(fft_dom_period_res_T) & ...
    isfinite(fft_dom_amp_res_T) & ...
    isfinite(fft_highE_ratio_res_T) & ...
    isfinite(fft_entropy_res_T) & ...
    isfinite(fft_dom_freq_res_N) & ...
    isfinite(fft_dom_period_res_N) & ...
    isfinite(fft_dom_amp_res_N) & ...
    isfinite(fft_highE_ratio_res_N) & ...
    isfinite(fft_entropy_res_N) & ...
    isfinite(fft_dom_freq_a) & ...
    isfinite(fft_dom_period_a) & ...
    isfinite(fft_dom_amp_a) & ...
    isfinite(fft_highE_ratio_a) & ...
    isfinite(fft_entropy_a) & ...
    isfinite(fft_dom_freq_ecc) & ...
    isfinite(fft_dom_period_ecc) & ...
    isfinite(fft_dom_amp_ecc) & ...
    isfinite(fft_highE_ratio_ecc) & ...
    isfinite(fft_entropy_ecc);

fprintf('Total rows: %d\n', nTLE);
fprintf('Valid ML rows: %d\n', sum(valid_for_ml));

outputFile = sprintf('tle_features_variables_ID_%s.mat', NORAD_ID);

save(outputFile);

fprintf('Saved %s\n', outputFile);

%% Plots de features

plotFeatureGroups = { ...
    {'a_km','da_km','ma_km_day'}, ...
    {'inc_deg','dinc_deg','mi_deg_day'}, ...
    {'ecc','decc','mecc_day'}, ...
    {'n_rev_day','dn_rev_day','mn_rev_day2'}, ...
    {'lambda_drift_deg_day','abs_lambda_drift_deg_day','abs_dlambda_drift_deg_day'}, ...
    {'res_R_km','res_T_km','res_N_km'}, ...
    {'res_pos_norm_km','res_vel_norm_kms','res_energy'}, ...
    {'res_norm_dt_km_day','res_T_dt_km_day','res_N_dt_km_day'}, ...
    {'res_slope_km_day','res_T_slope_km_day','res_N_slope_km_day'}, ...
    {'res_cumulative_roll','res_energy_roll','lowthrust_persistence'}, ...
    {'roll_std_resR','roll_std_resT','roll_std_resN'}, ...
    {'z_resR','z_resT','z_resN'}, ...
    {'z_pos','chi2_RTN','direction_change_roll'}, ...
    {'EW_indicator','NS_indicator','EW_NS_ratio'}, ...
    {'TN_ratio','control_asymmetry','res_a_km'} ...
    {'fft_dom_period_res_T','fft_dom_amp_res_T','fft_entropy_res_T'}, ...
    {'fft_highE_ratio_res_pos','fft_highE_ratio_res_T','fft_highE_ratio_res_N'}, ...
    {'fft_dom_period_a','fft_dom_amp_a','fft_entropy_a'}, ...
    {'fft_dom_period_ecc','fft_dom_amp_ecc','fft_entropy_ecc'}, ...
};

for g = 1:length(plotFeatureGroups)

    figure('Color','w','Name',sprintf('Feature group %d',g));

    featureGroup = plotFeatureGroups{g};

    for j = 1:length(featureGroup)

        featureName = featureGroup{j};

        subplot(3,1,j)

        if exist(featureName,'var')
            y = eval(featureName);
            plot(epoch_datetime,y,'LineWidth',1.0)
            grid on
            ylabel(strrep(featureName,'_','\_'))
            title(strrep(featureName,'_','\_'))
        else
            text(0.1,0.5,['Missing: ',featureName])
            axis off
        end

    end

    xlabel('Date')

end


%% Example FFT plot for one selected signal

exampleSignal = res_T_km;
exampleName = 'res_T_km';

validExample = isfinite(exampleSignal) & isfinite(t_days);

t_example = t_days(validExample);
x_example = exampleSignal(validExample);

% Interpolate to regular sampling for standard FFT
dt_uniform = median(diff(t_example),'omitnan');
t_uniform = (t_example(1):dt_uniform:t_example(end))';

x_uniform = interp1(t_example, x_example, t_uniform, 'linear', 'extrap');

% Remove mean/trend to avoid DC dominance
x_uniform = detrend(x_uniform);

Nfft = length(x_uniform);
Fs = 1/dt_uniform;          % samples/day

Y = fft(x_uniform);
P2 = abs(Y/Nfft);
P1 = P2(1:floor(Nfft/2)+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:floor(Nfft/2))/Nfft;   % cycles/day

% Avoid zero frequency in period plot
f_plot = f(2:end);
P_plot = P1(2:end);
period_days = 1 ./ f_plot;

figure('Color','w','Name',['FFT example - ',exampleName]);

subplot(2,1,1)
plot(t_uniform,x_uniform,'LineWidth',1.0)
grid on
xlabel('Days')
ylabel(exampleName)
title(['Detrended uniformly sampled signal: ',strrep(exampleName,'_','\_')])

subplot(2,1,2)
plot(period_days,P_plot,'LineWidth',1.0)
grid on
xlabel('Period [days]')
ylabel('Amplitude')
title(['FFT amplitude spectrum: ',strrep(exampleName,'_','\_')])
xlim([0 365])

%% Local functions

function d = angleDiffDeg(angle_deg)
    d_raw = diff(angle_deg);
    d = mod(d_raw + 180,360) - 180;
end


function dr_rtn = eci2rtn_residual(r,v,dr)

    r = r(:);
    v = v(:);
    dr = dr(:);

    Rhat = r / norm(r);

    h = cross(r,v);
    Nhat = h / norm(h);

    That = cross(Nhat,Rhat);
    That = That / norm(That);

    Q = [Rhat, That, Nhat];

    dr_rtn = (Q' * dr)';

end

function coe = rv2coe_simple(r,v,mu)

    r = r(:);
    v = v(:);

    R = norm(r);
    V = norm(v);

    h = cross(r,v);
    H = norm(h);

    khat = [0;0;1];

    n = cross(khat,h);
    N = norm(n);

    e_vec = ((V^2 - mu/R)*r - dot(r,v)*v) / mu;
    ecc = norm(e_vec);

    energy = V^2/2 - mu/R;
    a = -mu/(2*energy);

    inc = acosd(h(3)/H);

    if N > 1e-12
        RAAN = atan2d(n(2),n(1));
    else
        RAAN = 0;
    end

    if N > 1e-12 && ecc > 1e-12
        argp = atan2d(dot(cross(n,e_vec),h)/H,dot(n,e_vec));
    else
        argp = 0;
    end

    if ecc > 1e-12
        nu = atan2d(dot(cross(e_vec,r),h)/H,dot(e_vec,r));
    else
        nu = 0;
    end

    coe.a_km = a;
    coe.ecc = ecc;
    coe.inc_deg = inc;
    coe.RAAN_deg = wrapTo360(RAAN);
    coe.argp_deg = wrapTo360(argp);
    coe.nu_deg = wrapTo360(nu);

end

function [domFreq, domPeriod, domAmp, highERatio, specEntropy] = ...
    rollingFFTdescriptors(x, dt_days, W, minSamples)

    n = length(x);

    domFreq = NaN(n,1);        % dominant frequency [cycles/day]
    domPeriod = NaN(n,1);      % dominant period [days]
    domAmp = NaN(n,1);         % dominant amplitude
    highERatio = NaN(n,1);     % high-frequency energy ratio
    specEntropy = NaN(n,1);    % spectral entropy [0-1]

    for k = W:n

        idx = (k-W+1):k;

        xw = x(idx);
        dtw = dt_days(idx);

        valid = isfinite(xw) & isfinite(dtw);

        if sum(valid) < minSamples
            continue
        end

        xw = xw(valid);
        dtw = dtw(valid);

        % Approximate uniform sampling using median dt
        dt_med = median(dtw,'omitnan');

        if ~isfinite(dt_med) || dt_med <= 0
            continue
        end

        % Remove mean and linear trend
        xw = detrend(xw);

        N = length(xw);
        Fs = 1/dt_med;   % samples per day

        Y = fft(xw);
        P2 = abs(Y/N).^2;              % power spectrum
        P1 = P2(1:floor(N/2)+1);
        P1(2:end-1) = 2*P1(2:end-1);

        f = Fs*(0:floor(N/2))/N;       % cycles/day

        % Remove DC component
        f = f(2:end);
        P1 = P1(2:end);

        if isempty(P1) || sum(P1,'omitnan') <= 0
            continue
        end

        [peakPower, imax] = max(P1);

        domFreq(k) = f(imax);
        domPeriod(k) = 1 / f(imax);
        domAmp(k) = sqrt(peakPower);

        totalPower = sum(P1,'omitnan');

        % Define high frequency as above median frequency of current window
        fCut = median(f,'omitnan');
        highERatio(k) = sum(P1(f > fCut),'omitnan') / totalPower;

        p = P1 / totalPower;
        p = p(p > 0);

        specEntropy(k) = -sum(p .* log(p)) / log(length(P1));

    end

end