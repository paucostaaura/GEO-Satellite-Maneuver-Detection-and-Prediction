clear; clc;

TLEarx = 'TLE42917.txt';
% TLEarx = 'TLE41866.txt';

NORAD = [];
Year = [];
DOY = [];
epoch_datetime = datetime.empty(0,1);


% Extract NORAD ID from filename

tokens = regexp(TLEarx,'\d+','match');

if isempty(tokens)
    error('No NORAD ID found in filename.');
end

NORAD_ID = tokens{1};


inc_deg = [];
RAAN_deg = [];
ecc = [];
argp_deg = [];
M_deg = [];
n_rev_day = [];

tle_line1 = strings(0,1);
tle_line2 = strings(0,1);

fid = fopen(TLEarx,'r');
if fid == -1
    error("Cannot open TLE file: %s",TLEarx);
end

while true

    L1 = fgetl(fid);
    if L1 == -1
        break
    end

    L2 = fgetl(fid);
    if L2 == -1
        break
    end

    norad_i = str2double(strtrim(L1(3:7)));

    year_i = str2double(L1(19:20));
    if year_i < 57
        year_i = 2000 + year_i;
    else
        year_i = 1900 + year_i;
    end

    doy_i = str2double(L1(21:23)) + str2double(L1(24:32));
    epoch_i = datetime(year_i,1,1) + days(doy_i - 1);

    inc_i  = str2double(L2(9:16));
    raan_i = str2double(L2(18:25));
    ecc_i  = str2double(['0.' L2(27:33)]);
    argp_i = str2double(L2(35:42));
    M_i    = str2double(L2(44:51));
    n_i    = str2double(L2(53:63));

    NORAD(end+1,1) = norad_i;
    Year(end+1,1) = year_i;
    DOY(end+1,1) = doy_i;
    epoch_datetime(end+1,1) = epoch_i;

    inc_deg(end+1,1) = inc_i;
    RAAN_deg(end+1,1) = raan_i;
    ecc(end+1,1) = ecc_i;
    argp_deg(end+1,1) = argp_i;
    M_deg(end+1,1) = M_i;
    n_rev_day(end+1,1) = n_i;

    tle_line1(end+1,1) = string(L1);
    tle_line2(end+1,1) = string(L2);

end

fclose(fid);

%% Sort chronologically

[epoch_datetime,idx] = sort(epoch_datetime);

NORAD = NORAD(idx);
Year = Year(idx);
DOY = DOY(idx);

inc_deg = inc_deg(idx);
RAAN_deg = RAAN_deg(idx);
ecc = ecc(idx);
argp_deg = argp_deg(idx);
M_deg = M_deg(idx);
n_rev_day = n_rev_day(idx);

tle_line1 = tle_line1(idx);
tle_line2 = tle_line2(idx);

% Remove duplicated or near-duplicated epochs

min_separation_days = 1/1440;   % 1 minute

keep = true(length(epoch_datetime),1);

for k = 2:length(epoch_datetime)

    dt_k = days(epoch_datetime(k) - epoch_datetime(k-1));

    if dt_k < min_separation_days
        keep(k) = false;
    end

end

NORAD = NORAD(keep);
Year = Year(keep);
DOY = DOY(keep);

epoch_datetime = epoch_datetime(keep);

inc_deg = inc_deg(keep);
RAAN_deg = RAAN_deg(keep);
ecc = ecc(keep);
argp_deg = argp_deg(keep);
M_deg = M_deg(keep);
n_rev_day = n_rev_day(keep);

tle_line1 = tle_line1(keep);
tle_line2 = tle_line2(keep);

% Remove initial transfer / orbit acquisition phase

firstValid = 60;   % start from TLE number 60

NORAD = NORAD(firstValid:end);
Year = Year(firstValid:end);
DOY = DOY(firstValid:end);

epoch_datetime = epoch_datetime(firstValid:end);

inc_deg = inc_deg(firstValid:end);
RAAN_deg = RAAN_deg(firstValid:end);
ecc = ecc(firstValid:end);
argp_deg = argp_deg(firstValid:end);
M_deg = M_deg(firstValid:end);
n_rev_day = n_rev_day(firstValid:end);

tle_line1 = tle_line1(firstValid:end);
tle_line2 = tle_line2(firstValid:end);

% Time variables after cleaning and cutting

t_days = days(epoch_datetime - epoch_datetime(1));
dt_days = [NaN; days(diff(epoch_datetime))];
dt_s = dt_days * 86400;

%% Check chronological consistency

if any(dt_days(2:end) <= 0)
    warning('Non-positive time step detected.');
end

if any(dt_days > 30)
    warning('Very large gap (>30 days) between consecutive TLEs.');
end

fprintf('Minimum dt = %.3f days\n',min(dt_days(2:end)));
fprintf('Maximum dt = %.3f days\n',max(dt_days(2:end)));
fprintf('Median  dt = %.3f days\n',median(dt_days(2:end)));
%% Save

save(sprintf('tle_raw_variables_ID_%s.mat',NORAD_ID));

fprintf('Saved tle_raw_variables_ID_%s.mat\n',NORAD_ID);