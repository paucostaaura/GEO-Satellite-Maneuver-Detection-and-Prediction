%% GEO_full_perturbation_frequency_ode45.m
%
% Full Cartesian GEO propagation for expected station-keeping frequency
%
% Perturbations:
%   - EGM2008 Earth gravity
%   - Solar third-body gravity
%   - Lunar third-body gravity
%   - Solar radiation pressure
%   - Cylindrical Earth eclipse
%
% Manoeuvre logic:
%   - Stop propagation when EW or NS deadband is exceeded
%   - Count the event
%   - Apply an idealised station-keeping reset
%   - Continue until the end of the simulation
%
% Requirements:
%   - Aerospace Toolbox
%   - Ephemeris Data for Aerospace Toolbox
%
% Important:
% gravitysphericalharmonic already includes central gravity.
% Do NOT add -mu*r/r^3 separately when this function is used.

clear;
clc;
close all;

%% =========================================================
% USER SETTINGS
% ==========================================================

startEpoch = datetime(2025,1,1,0,0,0,'TimeZone','UTC');

simYears = 1;

% Use a coarse grid first. Change to 1 deg after checking the model.
longitudeGridDeg = 0:5:355;

% Station-keeping deadbands
longitudeBoxDeg = 0.05;
latitudeBoxDeg  = 0.05;

% Earth gravity model
gravityDegree = 20;

% Spacecraft optical properties
spacecraftMassKg = 2000;
solarAreaM2      = 40;
Cr               = 1.3;

% Enable/disable perturbations
useEarthGravity = true;
useSunGravity   = true;
useMoonGravity  = true;
useSRP          = true;
useEclipse      = true;

% Small residual errors after an idealised manoeuvre
postManoeuvreLonErrorDeg = 0.005;
postManoeuvreLatErrorDeg = 0.005;

% Numerical settings
relativeTolerance = 1e-9;
absoluteTolerance = 1e-6;
maximumStepSec    = 1800;       % 30 min

% Prevent infinite restart loops
maximumEventsPerLongitude = 200;

%% =========================================================
% CONSTANTS
% ==========================================================

P.muEarth = 3.986004418e14;       % [m^3/s^2]
P.muSun   = 1.32712440018e20;     % [m^3/s^2]
P.muMoon  = 4.9048695e12;         % [m^3/s^2]

P.RE = 6378.1363e3;               % [m]
P.AU = 149597870700;              % [m]

P.omegaEarth = 7.2921150e-5;      % [rad/s]

% Solar radiation pressure at 1 AU
P.solarPressure1AU = 4.56e-6;     % [N/m^2]

P.mass = spacecraftMassKg;
P.area = solarAreaM2;
P.Cr   = Cr;

P.startEpoch = startEpoch;

P.gravityDegree = gravityDegree;

P.useEarthGravity = useEarthGravity;
P.useSunGravity   = useSunGravity;
P.useMoonGravity  = useMoonGravity;
P.useSRP          = useSRP;
P.useEclipse      = useEclipse;

P.longitudeBoxRad = deg2rad(longitudeBoxDeg);
P.latitudeBoxRad  = deg2rad(latitudeBoxDeg);

P.postLonErrorRad = deg2rad(postManoeuvreLonErrorDeg);
P.postLatErrorRad = deg2rad(postManoeuvreLatErrorDeg);

P.aGEO = (P.muEarth/P.omegaEarth^2)^(1/3);

simulationDurationSec = simYears*365.25*86400;

fprintf('GEO radius: %.3f km\n',P.aGEO/1000);
fprintf('Simulation duration: %.2f years\n',simYears);

%% =========================================================
% CHECK REQUIRED FUNCTIONS
% ==========================================================

requiredFunctions = {
    'gravitysphericalharmonic'
    'planetEphemeris'
    };

for k = 1:numel(requiredFunctions)

    if exist(requiredFunctions{k},'file') ~= 2
        error(['Required function not found: ',requiredFunctions{k}, ...
            '. Check Aerospace Toolbox and ephemeris data installation.']);
    end

end

%% =========================================================
% STORAGE
% ==========================================================

nLongitudes = numel(longitudeGridDeg);

EWFrequency = zeros(nLongitudes,1);
NSFrequency = zeros(nLongitudes,1);
TotalFrequency = zeros(nLongitudes,1);

EWCount = zeros(nLongitudes,1);
NSCount = zeros(nLongitudes,1);

meanEWIntervalDays = NaN(nLongitudes,1);
meanNSIntervalDays = NaN(nLongitudes,1);

allEventTables = cell(nLongitudes,1);

%% =========================================================
% MAIN LONGITUDE LOOP
% ==========================================================

for iLon = 1:nLongitudes

    slotLongitudeDeg = longitudeGridDeg(iLon);
    slotLongitudeRad = deg2rad(slotLongitudeDeg);

    P.slotLongitudeRad = slotLongitudeRad;

    fprintf('\nLongitude %.1f deg\n',slotLongitudeDeg);

    % Initial nominal GEO state with small post-manoeuvre residual errors
    state0 = createNominalGEOState( ...
        0, ...
        slotLongitudeRad, ...
        P.postLonErrorRad, ...
        P.postLatErrorRad, ...
        P);

    currentTime = 0;

    ewEventTimes = [];
    nsEventTimes = [];

    eventTypeList = strings(0,1);
    eventTimeList = [];
    eventLonList  = [];
    eventLatList  = [];

    eventCounter = 0;

    while currentTime < simulationDurationSec

        eventCounter = eventCounter + 1;

        if eventCounter > maximumEventsPerLongitude
            warning('Maximum event count reached at %.1f deg.', ...
                slotLongitudeDeg);
            break
        end

        odeOptions = odeset( ...
            'RelTol',relativeTolerance, ...
            'AbsTol',absoluteTolerance, ...
            'MaxStep',maximumStepSec, ...
            'Events',@(t,x) stationKeepingEvents(t,x,P));

        [~,~,te,xe,ie] = ode45( ...
            @(t,x) orbitalDynamics(t,x,P), ...
            [currentTime simulationDurationSec], ...
            state0, ...
            odeOptions);

        % No box exit before the end
        if isempty(te)
            break
        end

        eventTime = te(end);
        eventState = xe(end,:)';

        % Multiple event surfaces may be reached simultaneously
        eventIndices = unique(ie);

        [longitudeRad,latitudeRad] = ...
            stateToEarthFixedAngles(eventTime,eventState,P);

        if any(eventIndices == 1)
            ewEventTimes(end+1,1) = eventTime;
            eventTypeList(end+1,1) = "EW";
        end

        if any(eventIndices == 2)
            nsEventTimes(end+1,1) = eventTime;
            eventTypeList(end+1,1) = "NS";
        end

        eventTimeList(end+1,1) = eventTime;
        eventLonList(end+1,1)  = rad2deg(longitudeRad);
        eventLatList(end+1,1)  = rad2deg(latitudeRad);

        % Idealised station-keeping correction
        %
        % The spacecraft is returned close to:
        %   assigned longitude
        %   zero latitude
        %   geostationary angular velocity
        %
        % A small residual error prevents exact equilibrium locking.

        state0 = createNominalGEOState( ...
            eventTime, ...
            slotLongitudeRad, ...
            P.postLonErrorRad, ...
            P.postLatErrorRad, ...
            P);

        % Move very slightly after the event to avoid detecting the same
        % zero crossing repeatedly.
        currentTime = eventTime + 1;

    end

    EWCount(iLon) = numel(ewEventTimes);
    NSCount(iLon) = numel(nsEventTimes);

    EWFrequency(iLon) = EWCount(iLon)/simYears;
    NSFrequency(iLon) = NSCount(iLon)/simYears;
    TotalFrequency(iLon) = EWFrequency(iLon) + NSFrequency(iLon);

    if ~isempty(ewEventTimes)
        meanEWIntervalDays(iLon) = ...
            mean(diff([0;ewEventTimes]))/86400;
    end

    if ~isempty(nsEventTimes)
        meanNSIntervalDays(iLon) = ...
            mean(diff([0;nsEventTimes]))/86400;
    end

    if ~isempty(eventTimeList)

        eventDate = startEpoch + seconds(eventTimeList);

        allEventTables{iLon} = table( ...
            eventDate, ...
            eventTypeList, ...
            eventLonList, ...
            eventLatList, ...
            'VariableNames',{ ...
            'event_date', ...
            'event_type', ...
            'longitude_deg', ...
            'latitude_deg'});

    else

        allEventTables{iLon} = table();

    end

    fprintf('  EW events: %d\n',EWCount(iLon));
    fprintf('  NS events: %d\n',NSCount(iLon));

end

%% =========================================================
% RESULTS
% ==========================================================

results = table( ...
    longitudeGridDeg(:), ...
    EWCount, ...
    NSCount, ...
    EWFrequency, ...
    NSFrequency, ...
    TotalFrequency, ...
    meanEWIntervalDays, ...
    meanNSIntervalDays, ...
    'VariableNames',{ ...
    'Longitude_deg', ...
    'EW_event_count', ...
    'NS_event_count', ...
    'EW_frequency_events_per_year', ...
    'NS_frequency_events_per_year', ...
    'Total_frequency_events_per_year', ...
    'Mean_EW_interval_days', ...
    'Mean_NS_interval_days'});

disp(results);

writetable(results, ...
    'GEO_full_perturbation_frequency_results.csv');

save('GEO_full_perturbation_frequency_results.mat', ...
    'results', ...
    'allEventTables', ...
    'P', ...
    'longitudeGridDeg');

%% =========================================================
% PLOTS
% ==========================================================

figure('Color','w');

plot(longitudeGridDeg,EWFrequency, ...
    '-o','LineWidth',0.9,'MarkerSize',3);
hold on;

plot(longitudeGridDeg,NSFrequency, ...
    '--o','LineWidth',0.9,'MarkerSize',3);

plot(longitudeGridDeg,TotalFrequency, ...
    '-s','LineWidth',0.9,'MarkerSize',3);

grid on;

xlabel('GEO longitude [deg]');
ylabel('Expected manoeuvre frequency [events/year]');

title('Full-force GEO station-keeping frequency');

legend('East-West','North-South','Total', ...
    'Location','best');

xlim([0 360]);

figure('Color','w');

plot(longitudeGridDeg,meanEWIntervalDays, ...
    '-o','LineWidth',0.9,'MarkerSize',3);

hold on;

plot(longitudeGridDeg,meanNSIntervalDays, ...
    '--o','LineWidth',0.9,'MarkerSize',3);

grid on;

xlabel('GEO longitude [deg]');
ylabel('Mean interval [days]');

title('Mean GEO station-keeping interval');

legend('East-West','North-South', ...
    'Location','best');

xlim([0 360]);

disp('Full-force propagation completed.');

%% =========================================================
% LOCAL FUNCTION: ORBITAL DYNAMICS
% ==========================================================

function dx = orbitalDynamics(t,x,P)

    rECI = x(1:3);
    vECI = x(4:6);

    accelerationECI = zeros(3,1);

    %% Earth gravity using EGM2008

    if P.useEarthGravity

        R_ECI_to_ECEF = simpleECItoECEFMatrix(t,P);

        rECEF = R_ECI_to_ECEF*rECI;

        [gx,gy,gz] = gravitysphericalharmonic( ...
            rECEF', ...
            'EGM2008', ...
            P.gravityDegree);

        aEarthECEF = [gx;gy;gz];

        % Rotate acceleration back to ECI
        aEarthECI = R_ECI_to_ECEF'*aEarthECEF;

        accelerationECI = accelerationECI + aEarthECI;

    else

        accelerationECI = accelerationECI ...
            - P.muEarth*rECI/norm(rECI)^3;

    end

    %% Solar and lunar ephemerides

    currentDate = P.startEpoch + seconds(t);
    jd = juliandate(currentDate);

    if P.useSunGravity || P.useSRP

        rSunECI_km = planetEphemeris( ...
            jd,'Earth','Sun');

        rSunECI = rSunECI_km(:)*1000;

    else

        rSunECI = zeros(3,1);

    end

    if P.useMoonGravity

        rMoonECI_km = planetEphemeris( ...
            jd,'Earth','Moon');

        rMoonECI = rMoonECI_km(:)*1000;

    else

        rMoonECI = zeros(3,1);

    end

    %% Solar third-body gravity

    if P.useSunGravity

        aSun = thirdBodyAcceleration( ...
            rECI,rSunECI,P.muSun);

        accelerationECI = accelerationECI + aSun;

    end

    %% Lunar third-body gravity

    if P.useMoonGravity

        aMoon = thirdBodyAcceleration( ...
            rECI,rMoonECI,P.muMoon);

        accelerationECI = accelerationECI + aMoon;

    end

    %% Solar radiation pressure

    if P.useSRP

        if P.useEclipse
            illumination = cylindricalShadowFactor( ...
                rECI,rSunECI,P.RE);
        else
            illumination = 1;
        end

        sunToSatellite = rECI - rSunECI;
        distanceSunSatellite = norm(sunToSatellite);

        srpDirection = sunToSatellite/distanceSunSatellite;

        solarPressure = P.solarPressure1AU ...
            *(P.AU/distanceSunSatellite)^2;

        aSRP = illumination ...
            *solarPressure ...
            *P.Cr ...
            *(P.area/P.mass) ...
            *srpDirection;

        accelerationECI = accelerationECI + aSRP;

    end

    dx = [
        vECI
        accelerationECI
        ];

end

%% =========================================================
% LOCAL FUNCTION: THIRD-BODY ACCELERATION
% ==========================================================

function acceleration = thirdBodyAcceleration( ...
    rSatellite,rBody,muBody)

    relativeVector = rBody - rSatellite;

    acceleration = muBody*( ...
        relativeVector/norm(relativeVector)^3 ...
        - rBody/norm(rBody)^3);

end

%% =========================================================
% LOCAL FUNCTION: ECLIPSE
% ==========================================================

function illumination = cylindricalShadowFactor( ...
    rSatellite,rSun,earthRadius)

    sunDirection = rSun/norm(rSun);

    % Satellite is behind Earth relative to the Sun
    behindEarth = dot(rSatellite,sunDirection) < 0;

    distanceFromSunEarthAxis = norm( ...
        rSatellite ...
        - dot(rSatellite,sunDirection)*sunDirection);

    insideCylinder = ...
        distanceFromSunEarthAxis < earthRadius;

    if behindEarth && insideCylinder
        illumination = 0;
    else
        illumination = 1;
    end

end

%% =========================================================
% LOCAL FUNCTION: STATION-KEEPING EVENTS
% ==========================================================

function [value,isterminal,direction] = ...
    stationKeepingEvents(t,x,P)

    [longitudeRad,latitudeRad] = ...
        stateToEarthFixedAngles(t,x,P);

    longitudeError = ...
        wrapToPiLocal(longitudeRad - P.slotLongitudeRad);

    % Event 1: East-West box exit
    ewValue = P.longitudeBoxRad - abs(longitudeError);

    % Event 2: North-South box exit
    nsValue = P.latitudeBoxRad - abs(latitudeRad);

    value = [
        ewValue
        nsValue
        ];

    % Stop ode45 at either event
    isterminal = [1;1];

    % Detect crossing from inside to outside
    direction = [-1;-1];

end

%% =========================================================
% LOCAL FUNCTION: EARTH-FIXED LONGITUDE AND LATITUDE
% ==========================================================

function [longitudeRad,latitudeRad] = ...
    stateToEarthFixedAngles(t,x,P)

    rECI = x(1:3);

    R = simpleECItoECEFMatrix(t,P);
    rECEF = R*rECI;

    longitudeRad = atan2(rECEF(2),rECEF(1));

    latitudeRad = atan2( ...
        rECEF(3), ...
        hypot(rECEF(1),rECEF(2)));

end

%% =========================================================
% LOCAL FUNCTION: NOMINAL GEO STATE
% ==========================================================

function stateECI = createNominalGEOState( ...
    t,slotLongitudeRad,lonResidualRad,latResidualRad,P)

    longitude = slotLongitudeRad + lonResidualRad;
    latitude  = latResidualRad;

    rECEF = P.aGEO*[
        cos(latitude)*cos(longitude)
        cos(latitude)*sin(longitude)
        sin(latitude)
        ];

    % A nominal geostationary spacecraft is stationary in ECEF
    vECEF = zeros(3,1);

    R = simpleECItoECEFMatrix(t,P);

    omegaCrossR = cross( ...
        [0;0;P.omegaEarth], ...
        rECEF);

    % From ECEF to ECI:
    % v_ECI = R'*(v_ECEF + omega x r_ECEF)

    rECI = R'*rECEF;
    vECI = R'*(vECEF + omegaCrossR);

    stateECI = [
        rECI
        vECI
        ];

end

%% =========================================================
% LOCAL FUNCTION: SIMPLE ECI TO ECEF ROTATION
% ==========================================================

function R = simpleECItoECEFMatrix(t,P)

    % Simplified Earth rotation.
    %
    % This is adequate for a first medium-fidelity model but does not
    % include precession, nutation, polar motion or UT1-UTC corrections.

    theta = P.omegaEarth*t;

    R = [
         cos(theta)  sin(theta)  0
        -sin(theta)  cos(theta)  0
         0           0           1
        ];

end

%% =========================================================
% LOCAL FUNCTION
% ==========================================================

function angle = wrapToPiLocal(angle)

    angle = mod(angle + pi,2*pi) - pi;

end