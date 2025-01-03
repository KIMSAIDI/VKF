function [predictions, signals] = vkf(outcomes,lambda,v0,sigma2)
% Volatile Kalman Filter (VKF) for continuous outcomes
% [predictions, signals] = vkf(lambda,sigma2,v0,outcomes)
% Inputs: 
%       outcomes: column-vector of outcomes
%       0<lambda<1, volatility learning rate 
%       v0>0, initial volatility
%       sigma2>0, outcome noise
% Outputs:
%       predictions: predicted state
%       signals: a struct that contains signals that might be useful:
%               predictions
%               volatility
%               learning rate
%               prediction error
%               volatility prediction error
% 
% Note: outputs of VKF also depends on initial variance (w0), which is
% assumed here w0 = sigma2
% 
% See the following paper and equations therein
% Piray and Daw, "A simple model for learning in volatile environments"
% https://doi.org/10.1101/701466

if lambda<=0 || lambda>=1
    error('lambda should be in the unit range');
end
if sigma2<=0
    error('sigma2 should be positive');
end
if v0<=0
    error('v0 should be positive');
end
    

w0      = sigma2;
[T,C] = size(outcomes);
% T: number of trials
% C: number of cues

m       = zeros(1,C);
w       = w0*ones(1,C);
v       = v0*ones(1,C);

predictions = nan(T,C);
learning_rate = nan(T,C);
volatility = nan(T,C);
prediction_error = nan(T,C);
volatility_error = nan(T,C);

for t  = 1:T      
    o = outcomes(t,:);
    predictions(t,:) = m;    
    volatility(t,:) = v;
        
    mpre        = m;
    wpre        = w;
    
    delta_m     = o - m;
    k           = (w+v)./(w+v + sigma2);                            % Eq 9
    m           = m + k.*delta_m;                                   % Eq 10
    w           = (1-k).*(w+v);                                     % Eq 11
    
    wcov        = (1-k).*wpre;                                      % Eq 12
    delta_v     = (m-mpre).^2 + w + wpre - 2*wcov - v;
    v           = v +lambda.*delta_v;                               % Eq 13
    
    learning_rate(t,:) = k;
    prediction_error(t,:) = delta_m;
    volatility_error(t,:) = delta_v;
end

signals = struct('predictions',predictions,'volatility',volatility,'learning_rate',learning_rate,...
                 'prediction_error',prediction_error,'volatility_prediction_error',volatility_error);

end

% Script pour comparer le VKF et un filtre particulaire dans un environnement binaire

% parameters
lambda = 0.1;   % volatility learning rate
sigma2 = 0.1;   % vbservation noise variance
T = 200;        % nombre d'essais
num_particles = 10000; % nombre de particules

% simuler un état caché qui alterne entre des périodes stables et volatiles
hidden_state = zeros(T, 1);
volatility_blocks = [50, 50, 50, 50]; % longueurs des blocs pour les conditions stables et volatiles
switch_points = cumsum(volatility_blocks);

% on génère des états cachés entre 0 et 1
for t = 1:T
    if t <= switch_points(1)
        hidden_state(t) = 0; % Bloc stable 1
    elseif t <= switch_points(2)
        hidden_state(t) = 1; % Bloc stable 2
    elseif t <= switch_points(3)
        hidden_state(t) = 0; % Bloc volatile 1
    else
        hidden_state(t) = 1; % Bloc volatile 2
    end
end

% on génère des observations avec du bruit
observations = hidden_state + sqrt(sigma2) * randn(T, 1);
observations = double(observations > 0.5);% on convertit en valeurs binaires (0 ou 1)

% VKF
[predictions_vkf, signals_vkf] = vkf(observations, lambda, 0.1, sigma2);

% application : filtre particulaire
particles = rand(num_particles, 1); % on initialise des particules entre 0 et 1
weights = ones(num_particles, 1) / num_particles; % poids égaux au début
predictions_pf = nan(T, 1);
volatility_pf = nan(T, 1);

for t = 1:T
    % update des particules
    particles = particles + sqrt(sigma2) * randn(num_particles, 1);
    particles = max(0, min(1, particles)); % pour contraindre les particules entre [0, 1]
    
    % on calcul les poids basés sur la vraisemblance des observations
    likelihoods = particles.^observations(t) .* (1 - particles).^(1 - observations(t));
    weights = likelihoods .* weights;
    weights = weights / sum(weights); % normalisation des poids
    
    % resampling des particules selon les poids
    indices = randsample(1:num_particles, num_particles, true, weights);
    particles = particles(indices);
    weights = ones(num_particles, 1) / num_particles; % réinitialisation des poids
    
    % estimation de la prédiction et de la volatilite
    predictions_pf(t) = mean(particles);
    volatility_pf(t) = var(particles);
end

% ---- plot -----
figure('Position', [100, 100, 1200, 600]); 
sgtitle('Comparison of VKF with Benchmark Sampling Method  in Binary Environement', 'FontSize', 14, 'FontWeight', 'Bold');

% VKF: volatilité et prédictions
subplot(2, 2, 1);
plot(1:T, signals_vkf.volatility, '-', 'LineWidth', 1., 'Color', [1, 0.5, 0]);
title('VKF Volatility');
xlabel('Trials');
ylabel('Volatility');
grid on;

subplot(2, 2, 2);
plot(1:T, predictions_vkf, '-', 'LineWidth', 1., 'Color', [1, 0.5, 0]);
hold on;
plot(1:T, hidden_state, 'b--', 'LineWidth', 1.);
title('VKF Predictions');
xlabel('Trials');
ylabel('Prediction');
ylim([-0.3, 1.3]); yticks([0 1]);
legend('VKF Prediction', 'True State');
grid on;

% filtre particulaire : volatilité et prédictions 
subplot(2, 2, 3);
plot(1:T, volatility_pf, '-', 'LineWidth', 1., 'Color', [1, 0.5, 0]);
title('Benchmark Volatility');
xlabel('Trials');
ylabel('Volatility');
grid on;

subplot(2, 2, 4);
plot(1:T, predictions_pf, '-', 'LineWidth', 1., 'Color', [1, 0.5, 0]);
hold on;
plot(1:T, hidden_state, 'b--', 'LineWidth', 1);
title('Benchmark Predictions');
xlabel('Trials');
ylabel('Prediction');
ylim([-0.3, 1.3]); yticks([0 1]);
legend('PF Prediction', 'True State');
grid on;