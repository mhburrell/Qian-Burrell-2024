addpath('D:/PhD/Photometry/Analysis/pipeline')
mouse = 'FgDA_W1';
protocol = 'Selina_C5D5R3E5R3';
round = 'water_vs_sucrose';
implant_side = 'L';
filedir = ['D:/PhD/Photometry/DATA/' round '/' mouse '/' protocol '/Session Data/'] ;
DoricStudioVersion = '5.4.1.23';
cd(filedir)
files=uigetfile('*.mat','Select the INPUT DATA FILE(s) from BPOD','MultiSelect','on');
for i = 1:length(files)
    filename = files(i);
    if implant_side == 'L'

        sync_bpod_doric_data(filedir, filename,DoricStudioVersion)
    else 
        sync_bpod_doric_data_for_20221006andafter(filedir, filename,DoricStudioVersion)
    end
end