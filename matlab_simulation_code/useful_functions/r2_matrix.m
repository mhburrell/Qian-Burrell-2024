function R2_values  = r2_matrix(Y_actual,Y_predicted)
[~, nCols] = size(Y_actual); % Assuming Y_actual and Y_predicted have the same size
R2_values = zeros(1, nCols);

for col = 1:nCols
    y_actual_col = Y_actual(:, col);
    y_predicted_col = Y_predicted(:, col);
    
    residuals = y_actual_col - y_predicted_col;
    SSR = sum(residuals.^2);
    
    mean_actual = mean(y_actual_col);
    SST = sum((y_actual_col - mean_actual).^2);
    
    R2_values(col) = 1 - (SSR/SST);
end
