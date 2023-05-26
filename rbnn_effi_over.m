close all;
clc;
clear all;

%Input Data
turbin1_ML = readtable('turbin1-machinelearning2.csv');
turbin1_ML = table2array(turbin1_ML);
turbinParam = turbin1_ML(:, 1:7);

%Data Division
pt1r = turbin1_ML(:,1);
pt2r = turbin1_ML(:,2);
pt3 = turbin1_ML(:,3);
pt4 = turbin1_ML(:,4);
pt5 = turbin1_ML(:,5);
pt6 = turbin1_ML(:,6);
pt7 = turbin1_ML(:,7);
%effi = turbin1_ML(:,9);
effi = turbin1_ML(:,10);

%Transpose Data
turbinParam = transpose(turbinParam);
effi = transpose(effi);


%rbnn param
eg = 0.1;
sc = 100;

%build model(rbnn)
model_rbnn = newrb(turbinParam, effi, eg, sc);

%create Sample Data (lhs)
%sample_pt1r = lhsdesign_modified(48000, 0.014757, 0.018037)*1000;
%sample_pt2r = lhsdesign_modified(48000, 0.033448, 0.04088)*1000;
%sample_pt3 = lhsdesign_modified(48000, 3.6162, 4.4198)*10;
%sample_pt4 = lhsdesign_modified(48000, -60.4043, -49.4217);
%sample_pt5 = lhsdesign_modified(48000, 7.3989, 9.0431)*10;
%sample_pt6 = lhsdesign_modified(48000, -6.3228, -5.1732)*10;
%sample_pt7 = lhsdesign_modified(48000, -76.9626, -62.9694);

sample_pt1r = lhsdesign_modified(48000, 14.757, 18.037);
sample_pt2r = lhsdesign_modified(48000, 33.448, 40.88);
sample_pt3 = lhsdesign_modified(48000, 36.162, 44.198);
sample_pt4 = lhsdesign_modified(48000, -60.4043, -49.4217);
sample_pt5 = lhsdesign_modified(48000, 73.989, 90.431);
sample_pt6 = lhsdesign_modified(48000, -63.228, -51.732);
sample_pt7 = lhsdesign_modified(48000, -76.9626, -62.9694);

%sample_pt3 = lhsdesign_modified(48000, -28.2832, -23.1408);
%sample_pt4 = lhsdesign_modified(48000, -73.3799, -60.0381);
%sample_pt5 = lhsdesign_modified(48000, -28.2843, -23.1417);
%sample_pt6 = lhsdesign_modified(48000, -41.2148, -33.7212);
%sample_pt7 = lhsdesign_modified(48000, -73.4173, -60.0687);


sample_turbinParam = horzcat(sample_pt1r, sample_pt2r, sample_pt3, sample_pt4, sample_pt5, sample_pt6, sample_pt7);
sample_turbinParam = transpose(sample_turbinParam);

%run model
result_power = model_rbnn(sample_turbinParam);

%find max power and turbin parameter
[max_effi, max_index] = max(result_power);
max_turbinParam = sample_turbinParam(:, max_index)
max_effi

%find all parameter over input_max_effi

input_max_effi=max(effi);
%입력값중 최고 효율
input_over_index =find(result_power > max(input_max_effi));
turbin_para_effi_pred = vertcat(sample_turbinParam(:,input_over_index),result_power(input_over_index))
turbin_para_effi_pred = transpose(turbin_para_effi_pred)
turbin_para_effi_pred = sortrows(turbin_para_effi_pred,8) 

%Opt Paramter by reference(paper)
%Turbin_Param_Opt = [0.0156;0.0364;3.8798;-52.8123;8.9330;-5.7180;-70.0365]
%Turbin_Param_Opt = [0.014757;0.04088;4.4198;-60.4043;9.0431;-6.3228;-76.9626]
Turbin_Param_Opt = [18.037;33.448;36.162;-60.4043;90.431;-63.228;-62.9694]

%Turbin_Param_Opt = [0.014757;0.04088;-28.2832;-60.0381;-23.1417;-41.2148;-73.4173]

eff_Opt_ref = model_rbnn(Turbin_Param_Opt)