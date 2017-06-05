function s = compute_similarity(Y, y)
    % Compute similarity between y and Y
    % Input:
    % Y: a dxN matrix, d is the dimension of feature, N is number of particle
    % Y contains features for each particle
    % y: feature of the last tracked rect
    % Ouptput:
    % s: a vector indicates the similarity between each column of Y and y
    [m, n] = size(Y);
    s=[];
    for i = 1:n
        difference = 0;
        for j = 1:m
            difference = difference + sqrt((Y(j,i) - y(j))^2);
        end
        s=[s, difference];
    end
    
end