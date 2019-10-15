function plot_dendrogram(affinity_matrix,task_list_labels)
    semantic = {'Class 1000', 'Class Places','Segment Semantic'};
    geometrix = {'Vanishing Point','Room Layout'};
    dimension2 = {'Edge 2D','Keypoint 2D','Colorization','Inpainting Whole','Autoencoder','Segment 2D','Denoise'};
    dimension3 = {'Curvature','Edge 3D','Keypoint 3D','Reshade','Rgb2depth','Rgb2sfnorm','Segment 25D','Rgb2mist'};
    
    figure
    Z = linkage(affinity_matrix);
    D = pdist(affinity_matrix);
    leafOrder = optimalleaforder(Z,D);
    
    for i = 1:length(task_list_labels)
        if ismember(task_list_labels{i}, semantic)
            task_list_labels{i} = ['\color{magenta} ' task_list_labels{i}];
        elseif ismember(task_list_labels{i}, geometrix)
            task_list_labels{i} = ['\color{red} ' task_list_labels{i}];
        elseif ismember(task_list_labels{i}, dimension2)
            task_list_labels{i} = ['\color{blue} ' task_list_labels{i}];
        elseif ismember(task_list_labels{i}, dimension3)
            task_list_labels{i} = ['\color{green} ' task_list_labels{i}];
        else
            task_list_labels{i} = ['\color{black} ' task_list_labels{i}];
        end
    end
    dendrogram(Z,20,'Orientation','left', 'Labels',task_list_labels,'Reorder',leafOrder)
end