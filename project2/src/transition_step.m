function particles = transition_step(particles, stds)
    % Sample particles from gaussian distribution N(particles, stds)
    % Input:
    % particles:  a matrix of 4 rows and n_particles cols
    % stds: a 4 dimention vector, each is a standard deviation for a
    % dimension of particle
    % Ouput:
    % particles: output particles
    [~,n] = size(particles);
%     for i = 1 : m
%         for j = 1 : n
%             particles(i,j) = normrnd(particles(i,j), stds(i));
%         end
%     end
     for i = 1 : n
         mu = [particles(1,i), particles(2,i), particles(3,i),particles(4,i)];
         Sigma = [stds(1), 0, 0, 0;
                  0, stds(2), 0, 0;
                  0, 0, stds(3), 0;
                  0, 0, 0, stds(4)];
         temp = mvnrnd(mu,Sigma);
         particles(1,i) = temp(1);particles(2,i) = temp(2);
         particles(3,i) = temp(3);particles(4,i) = temp(4);
     end
end