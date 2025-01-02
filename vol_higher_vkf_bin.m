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


% Script pour simuler le VKF dans un environnement binaire avec une forte volatilité

% Paramètres
lambda = 0.2;   % Taux d'apprentissage de la volatilité (augmenté)
v0 = 0.2;       % Volatilité initiale (augmentée)
sigma2 = 0.1;   % Variance du bruit sur les observations
T = 200;        % Nombre d'essais

% Générer un état caché binaire avec des transitions fréquentes
hidden_state = zeros(T, 1);
for t = 1:T
    if mod(t, 20) < 10
        hidden_state(t) = 0; % Bloc 0
    else
        hidden_state(t) = 1; % Bloc 1
    end
end

% Générer des observations bruitées
observations = hidden_state + sqrt(sigma2) * randn(T, 1);
observations = double(observations > 0.5); % Conversion en valeurs binaires (0 ou 1)

% Application du VKF
[predictions, signals] = vkf(observations, lambda, v0, sigma2);

% Visualization of the results
figure('Position', [100, 100, 1200, 600]);

% hidden state and VKF predictions
subplot(3, 1, 1);
plot(1:T, hidden_state, 'b--', 'LineWidth', 1, 'DisplayName', 'True State'); hold on;
plot(1:T, predictions, '-', 'LineWidth', 1, 'DisplayName', 'VKF Predictions', 'Color', [1, 0.5, 0]);
xlabel('Trial');
ylabel('State');
ylim([-0.3, 1.3]); yticks([0 1]);
yticklabels({'0', '1'}); % Binary state labels
legend('Location', 'Best');
title('Hidden State and VKF Predictions');
grid on;

% VKF learning rate
subplot(3, 1, 2);
plot(1:T, signals.learning_rate, '-', 'LineWidth', 1.5, 'Color', [1, 0.5, 0]);
xlabel('Trial');
ylabel('Learning Rate');
ylim([0.6, 0.9]); % Adjusted limits for high volatility
title('VKF Learning Rate');
grid on;

% volatility estimated by VKF
subplot(3, 1, 3);
plot(1:T, signals.volatility, '-', 'LineWidth', 1.5, 'Color', [1, 0.5, 0]);
xlabel('Trial');
ylabel('Volatility');
ylim([0, 0.6]); % Adjusted limits for high volatility
title('Volatility Estimated by VKF');
grid on;

sgtitle('Performance Analysis of the VKF in a Binary Environment with High Volatility', 'FontSize', 14, 'FontWeight', 'Bold');
