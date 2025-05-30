% Base value
Sb = 10e6;                % VA
Vb = 12.66e3;             % V
Zb = (Vb)^2/Sb;           % O
Ib = Sb*sqrt(3)/Vb;   % A
N_Bus = 33;
NT = 24;
% Option_Price = 1; % TOU 1, Peak-Vally 2, 3 Random, 4 LMP
% Cgrid1 = [0.08*ones(1,8) 0.11*ones(1,6) 0.15*ones(1,8) 0.08*ones(1,2)]; % $/(kW.h)TOU
% Cgrid2 = [0.3/6*ones(1,7) 0.6/6*ones(1,14) 0.3/6*ones(1,3)]; % peak-valley
% Cgrid3 = [0.033 0.027 0.020 0.017 0.017 0.029 0.033 0.054 ...
%           0.215 0.572 0.572 0.572 0.033 0.027 0.020 0.017 ...
%           0.017 0.029 0.033 0.054 0.215 0.572 0.572 0.572];
% switch Option_Price
%     case 1 
%         Cgrid = 0.45*Cgrid1*1e3; % $/(MW.h)
%     case 2
%         Cgrid = 0.65*Cgrid2*1e3;
%     case 3
%         Cgrid = 0.15*Cgrid3*1e3;
%     case 4
%         load LMP_base.mat
%         Cgrid = LMP_base;
%     end
Cgrid = [208.7*ones(1,7) 294.8*ones(1,14) 208.7*ones(1,3)]; % $/(p.u)/10^7
% % Line
% No.(1)|From(2)|To(3)|Pd(kW)(4)|Qd(5)|r(6)(ohms)|x(7)|rpu(8)|xpu(9)|R/X(10)|Pup(11)
linedata = [
        1	1	2	100	60	0.0922	0.047	0.0058	0.0029	2       9900
        2	2	3	90	40	0.493	0.2511	0.0308	0.0157	1.9618	9900
        3	3	4	120	80	0.366	0.1864	0.0228	0.0116	1.9655	9900
        4	4	5	60	30	0.3811	0.1941	0.0238	0.0121	1.9669	9900
        5	5	6	60	20	0.819	0.707	0.0511	0.0441	1.1587	9900
        6	6	7	200	100	0.1872	0.6188	0.0117	0.0386	0.3031	9900
        7	7	8	200	100	1.7114	1.2351	0.1068	0.0771	1.3852	9900
        8	8	9	60	20	1.03	0.74	0.0643	0.0462	1.3918	9900
        9	9	10	60	20	1.044	0.74	0.0651	0.0462	1.4091	9900
        10	10	11	45	30	0.1966	0.065	0.0123	0.0041	3       9900
        11	11	12	60	35	0.3744	0.1238	0.0234	0.0077	3.039	9900
        12	12	13	60	35	1.468	1.155	0.0916	0.0721	1.2705	9900
        13	13	14	120	80	0.5416	0.7129	0.0338	0.0445	0.7596	9900
        14	14	15	60	10	0.591	0.526	0.0369	0.0328	1.125	9900
        15	15	16	60	20	0.7463	0.545	0.0466	0.034	1.3706	9900
        16	16	17	60	20	1.289	1.721	0.0804	0.1074	0.7486	9900
        17	17	18	90	40	0.732	0.574	0.0457	0.0358	1.2765	9900
        18	2	19	90	40	0.164	0.1565	0.0102	0.0098	1.0408	9900
        19	19	20	90	40	1.5042	1.3554	0.0939	0.0846	1.1099	9900
        20	20	21	90	40	0.4095	0.4784	0.0255	0.0298	0.8557	9900
        21	21	22	90	40	0.7089	0.9373	0.0442	0.0585	0.7556	9900
        22	3	23	90	50	0.4512	0.3083	0.0282	0.0192	1.4688	9900
        23	23	24	420	200	0.898	0.7091	0.056	0.0442	1.267	9900
        24	24	25	420	200	0.896	0.7011	0.0559	0.0437	1.2792	9900
        25	6	26	60	25	0.203	0.1034	0.0127	0.0065	1.9538	9900
        26	26	27	60	25	0.2842	0.1447	0.0177	0.009	1.9667	9900
        27	27	28	60	20	1.059	0.9337	0.0661	0.0583	1.1338	9900
        28	28	29	120	70	0.8042	0.7006	0.0502	0.0437	1.1487	9900
        29	29	30	200	600	0.5075	0.2585	0.0317	0.0161	1.9689	9900
        30	30	31	150	70	0.9744	0.963	0.0608	0.0601	1.0116	9900
        31	31	32	210	100	0.3105	0.3619	0.0194	0.0226	0.8584	9900
        32	32	33	60	40	0.341	0.5302	0.0213	0.0331	0.6435	9900
        ];  
N_line = size(linedata,1);
line_i = linedata(:,2);
line_j = linedata(:,3);
r = linedata(:,6)/Zb;
x = linedata(:,7)/Zb;
b = zeros(N_line,1);

%% Generation
% Gas turbine
% Loc.(1)|P_min MW (2)|P_max MW (3)|Q_min MVar (4)|Q_max(5) MVar|a
% $/MW^2(6)|b $/MW (7) | alpha $/MW^2 (8) | beta $/MW (9)
GTdata = [
       7    0   2000*1e3   0   0.5*1e6   0.12   20   5   6
       21   0   1500*1e3   0   1e6       0.09   15   4   5
       % 2   0   2*1e6    -10*1e6   10*1e6   0.1   20   5   6
%    13   0   1.0   0   1.0   0  15   4   5
%    7    0   1.0   0   0.5   0.12   40  5   6
%    13   0   1.0   0   1.0   0.09   30   4   5
];
IndGT= GTdata(:,1);  
Pg_GT_l = zeros(N_Bus,NT);
Pg_GT_u = zeros(N_Bus,NT);
Qg_GT_l = zeros(N_Bus,NT);
Qg_GT_u = zeros(N_Bus,NT);
Pg_GT_l(IndGT,:) = GTdata(:,2)/Sb*ones(1,NT); % p.u.
Pg_GT_u(IndGT,:) = GTdata(:,3)/Sb*ones(1,NT); % p.u.
Qg_GT_l(IndGT,:) = GTdata(:,4)/Sb*ones(1,NT); % p.u.
Qg_GT_u(IndGT,:) = GTdata(:,5)/Sb*ones(1,NT); % p.u.
Pg_GT_max = Pg_GT_u; % p.u.
Qg_GT_max = Qg_GT_u; % p.u.

ag = zeros(N_Bus,1); 
bg = zeros(N_Bus,1); 
ag(IndGT,:) = GTdata(:,6);
bg(IndGT,:) = GTdata(:,7);
ag(IndGT,:) = ag(IndGT,:); % p.u./(Sb^2)
bg(IndGT,:) = bg(IndGT,:); % p.u./Sb

% PV Gen %loc(1)|RatedPower(2)(MW)
PVdata = [
         17 0;
         32 0
         ];
IndGenPV = PVdata(:,1);
% relative value of average and st deviation 
PVxishu0=[0.00,0.00,0.00,0.00,0.00,0.12,0.21,0.35,0.91,0.96,0.95,0.95,0.90,0.70,0.52,0.3,0.26,0.11,0.11,0.00,0.00,0.00,0.00,0.00]*800*0.9/1000;
PVxishu1=[0.00,0.00,0.00,0.00,0.00,0.12,0.21,0.35,0.91,0.96,0.95,0.95,0.90,0.70,0.52,0.3,0.26,0.11,0.11,0.00,0.00,0.00,0.00,0.00]*1620*0.9/1000;
R_Pwav = [0.63 0.73 0.79 0.75 0.77 0.51 0.35 0.17 0.09 0.18 0.34 0.4 ...
          0.21 0.39 0.67 0.74 0.76 0.73 0.36 0.31 0.36 0.32 0.42 0.5]*1.25; % Pwav/Pwr
PV_pre = [PVxishu0(1:NT)*1e6/Sb; PVxishu1(1:NT)*1e6/Sb]; % p.u
PV_l = zeros(N_Bus,NT);
PV_u = zeros(N_Bus,NT);
PV_l(IndGenPV,:) = 0;
PV_u(IndGenPV,:) = PV_pre; % p.u

% SVG % Mvar 
% Location(1)|Max(2)var|Min(3)
SVGdata = [ 
        22  200*1e3 -200*1e3; % Var
        26  400*1e3 0;
        27  200*1e3 -200*1e3; % Var
    ];
IndSVG = SVGdata(:,1);
Qg_SVG_l = zeros(N_Bus,NT);
Qg_SVG_u = zeros(N_Bus,NT);
Qg_SVG_l(IndSVG,:) = ones(length(IndSVG),NT).*repmat(SVGdata(:,3)/Sb,1,NT); % p.u.
Qg_SVG_u(IndSVG,:) = ones(length(IndSVG),NT).*repmat(SVGdata(:,2)/Sb,1,NT); % p.u.
Qg_SVG_max = Qg_SVG_u;% p.u.

%ESS
% Location(1)|Max(2)Cha|Max(3)Dis|Max capicaty(4)|Min(5)
% ESSdata = [ 
%         18  1e6     1e6     1.2*1e6 0; % w
%         33  0.8*1e6 0.8*1e6 1.2*1e6 0; % w
%     ];
% IndESS = [];
% 
% epsilon = 0.01;  % Tolerance value, adjust as needed
% IndESS = find(abs(u - 1) <= epsilon);
%charging
ESS_cha_l = zeros(N_Bus,NT);
% ESS_cha_u = R_ESS*ones(1,NT);
% ESS_cha_l(IndESS,:) = ESSdata(:,5)/Sb*ones(1,NT);
% ESS_cha_u(IndESS,:) = ESSdata(:,2)/Sb*ones(1,NT);
%discharging
ESS_dis_l = zeros(N_Bus,NT);
% ESS_dis_u = R_ESS*ones(1,NT);
% ESS_dis_l(IndESS,:) = ESSdata(:,5)/Sb*ones(1,NT);
% ESS_dis_u(IndESS,:) = ESSdata(:,2)/Sb*ones(1,NT);
%soc
% ESS_soc0 = 0.2*1e5*ones(33,1);
ESS_soc_l = zeros(N_Bus,NT);
% ESS_soc_u = C_ESS*ones(1,NT);
% ESS_soc_l(IndESS,:) = ESSdata(:,5)/Sb*ones(1,NT);
% ESS_soc_u(IndESS,:) = ESSdata(:,4)/Sb*ones(1,NT);
%% PDN Bus
Pd_ratio = [0;linedata(:,4)]/sum(linedata(:,4));% pd factor
Qd_ratio = [0;linedata(:,5)]/sum(linedata(:,5));% qd factor
PFactor = 1;   % adjust(MW)
SeasonP =  1;
switch SeasonP
    case 0 % normal
        Pd_Season = (PFactor*[63 62 60 58 59 65 72 85 95 99 100 99 93 92 90 88 90 92 96 98 96 90 80 70]/15-1.5)*1e6;  % MW
    case {1,3} % spring and fall(MW)
        Pd_Season = (PFactor*[63 62 60 58 59 65 72 85 95 99 100 99 93 92 90 88 90 92 96 98 96 90 80 70]/15-1.5)*1e6; 
    case 2 % summer(MW)
        Pd_Season = (PFactor*[64 60 58 56 56 58 64 76 87 95 99 100 99 100 100 97 96 96 93 92 92 93 87 72]/15-1.5)*1e6;
    case 4  % winter(MW)
        Pd_Season = (PFactor*[67 63 60 59 59 60 74 86 95 96 96 95 95 95 93 94 99 100 100 96 91 83 73 63]/15-1.5)*1e6;
end
Qd_Season = ([18 16 15 14 15.5 15 16 17 18 19 20 20.5 21 20.5 21 19.5 20 20 19.5 19.5 18.5 18.5 18 18]/10)*1e6;% MVar
Pd = Pd_ratio*Pd_Season(1:NT); 
Qd = Qd_ratio*Qd_Season(1:NT);
Pd = Pd/Sb; % p.u 
Qd = Qd/Sb; % p.u
% Pd = [zeros(1,NT);Pd]; % load on root node
% Qd = [zeros(1,NT);Qd];
U2_l = 0.9^2*ones(N_Bus,NT); % p.u.
U2_u = 1.1^2*ones(N_Bus,NT); % p.u.
U2_max = U2_u; % p.u.

% figure
% % Pd_Season_0 = PFactor*[63 62 60 58 59 65 72 85 95 99 100 99 93 92 90 88 90 92 96 98 96 90 80 70]/15-1.5;  % MW
% Pd_Season_1 = PFactor*[63 62 60 58 59 65 72 85 95 99 100 99 93 92 90 88 90 92 96 98 96 90 80 70]/15-1.5; 
% Pd_Season_2 = PFactor*[64 60 58 56 56 58 64 76 87 95 99 100 99 100 100 97 96 96 93 92 92 93 87 72]/15-1.5;
% Pd_Season_3 = PFactor*[67 63 60 59 59 60 74 86 95 96 96 95 95 95 93 94 99 100 100 96 91 83 73 63]/15-1.5;
% plot(1:NT,Sb*Pd_Season_1,'-go')
% hold on
% plot(1:NT,Sb*Pd_Season_2,'-bs')
% plot(1:NT,Sb*Pd_Season_3,'-r*')
% xlabel('time (h)')
% ylabel('electric load (MW))')
% hh = legend('spring-fall', 'summer', 'winter');%
% hh.Orientation = 'horizontal';

DataPDN.IndGT = IndGT; % struct 
DataPDN.IndGenPV = IndGenPV; 
DataPDN.IndSVG = IndSVG; % struct 
% DataPDN.IndESS = IndESS;