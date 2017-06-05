function particles = resample_step(particles, weights)
    % resaple particles according to weights
    % Input: 
    % particles: a 4xN matrix, each column is a particle
    % weights: a vector, each element corresponds to a particle
    % Output:
    % particles: resampled particles
    [~, n] = size(particles);
    weights1 = sort(weights);
    number = weights1(ceil(n)*0.98);
    notremove =[];
    length = 0;
    
    %需要我们先除去权重比较小的那些点
    for i = 1 : n
        if weights(i) >= number
            notremove = [notremove ,[i; weights(i)]];
            length = length + 1;
        end
    end
    
    %做一步归一化
    for i = 1 : length
         notremove(2,i) = notremove(2,i)/sum(notremove(2,:));
    end
    
    %进行重采样
    new_particles = [];
    count = 0;
    for i = 1 : length
        if i == length
            num = n - count;
        else
            num = floor(n * notremove(2,i));
            count = count + num;
        end
        
        for j = 1 : num
            result = [];
            mu = [particles(1,i), particles(2,i), particles(3,i),particles(4,i)];
            Sigma = [0.0001, 0, 0, 0;
                     0, 0.0001, 0, 0;
                     0, 0, 0.0001, 0;
                     0, 0, 0, 0.0001];
             temp = mvnrnd(mu,Sigma);
             result = [temp(1);temp(2);temp(3);temp(4)];
             new_particles = [new_particles, result];
        end
    end
    particles = new_particles;
end