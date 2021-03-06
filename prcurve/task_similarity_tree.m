clear,clc

load('affinity_depara.mat')
task_list = {'Autoencoder', 'Curvature', 'Denoise', 'Edge 2D', 'Edge 3D', ...
'Keypoint 2D','Keypoint 3D', ...
'Reshade' ,'Rgb2depth' ,'Rgb2mist','Rgb2sfnorm', ...
'Room Layout', 'Segment 25D', 'Segment 2D', 'Vanishing Point', ...
'Segment Semantic' ,'Class 1000' ,'Class Places'};

plot_dendrogram(affinity_depara./max(max(affinity_depara)), task_list);
plot_dendrogram(affinity_depara_coco./max(max(affinity_depara_coco)), task_list);
plot_dendrogram(affinity_depara_indoor./max(max(affinity_depara_indoor)), task_list);
