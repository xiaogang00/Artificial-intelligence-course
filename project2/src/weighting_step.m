function weights = weighting_step(img, particles, sz_I, y, feature_type)
    % This function first compute feature for each particle, then compute
    % similarity between features of particles and y
    
    % Input:
    % img: input image
    % particles£ºa 4xN matrix, each col corresponds to a particle state
    % sz_I: base size that a rect should be resized to
    % y: the feature of the last tracked rect.
    % feature_type: the type of feature
    % Oputput:
    % weights: a vector, each element corresponds to a particle
    
    Y =[];
    [~, n] = size(particles);
    for i = 1 : n
        rect = [particles(1,i), particles(2,i),...
                particles(3,i),particles(4,i)];
        temp = feature_extract(img, rect, sz_I, feature_type);
        Y = [Y, temp];
    end
    s = compute_similarity(Y, y);
    
    weights = [];
    for i = 1 : n
        weights(i) = s(i)/sum(s);
    end
    
end