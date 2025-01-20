function main_mlb_kf()
    block_size = 100;
    iterations = 10;
    noise_std = 0.2;
   
    sequence = generate_continuous_sequence(block_size, iterations, noise_std); % sequence où la récompense de base change tous les block_size pas de temps -> introduit de la variabilité
   
    % ----- MLB_KF -----

    % agent Main2 (multi-armed bandit)
    agent = Main2(2, 0.14, 1/15, 1/350, 0.44, 1.5, 0.5, 0.5, 0.05, 0.1); % changement des variables var_ob et var_tr

    % initialisation des tableaux pour Main2
    T = length(sequence);
    predictions = zeros(1, T);
    volatilities = zeros(1, T);
    learning_rates = zeros(1, T);

    % simulation pour Main2
    for t = 1:T
        agent.decide();
        predictions(t) = agent.mu(1); % prédiction pour le premier bras
        volatilities(t) = agent.var(1); % volatilité associée au premier bras
        learning_rates(t) = agent.var(1) / (agent.var(1) + agent.var_ob);
        % update de l'agent avec la récompense
        agent.update(sequence(t));
    end

    % ----- VKF -----

    % paramètres 
    lambda = 0.1; % taux d'apprentissage de la volatilité
    v0 = 0.1;     % volatilité initiale
    sigma2 = 0.1; % variance du bruit

    sequence = sequence(:); % Convertir en vecteur colonne
    % Simulation pour le VKF
    [vkf_predictions, vkf_signals] = vkf(sequence, lambda, v0, sigma2);

    % ----- plot -----

    figure('Position', [100, 100, 1600, 1000]);
    
    % plot : prédictions VKF
    subplot(3, 2, 1);
    plot(sequence, 'k-', 'LineWidth', 1.5); hold on;
    plot(vkf_predictions, 'r-', 'LineWidth', 1.5);
    title('Prédictions VKF vs Séquence réelle', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Séquence réelle', 'VKF');
    xlabel('Timestep'); ylabel('Valeur');
    grid on;
    
    % plot : prédictions MLB_KF
    subplot(3, 2, 2);
    plot(sequence, 'k-', 'LineWidth', 1.5); hold on;
    plot(predictions, 'b-', 'LineWidth', 1.5);
    title('Prédictions MLB-KF vs Séquence réelle', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Séquence réelle', 'MLB-KF');
    xlabel('Timestep'); ylabel('Valeur');
    grid on;
    
    % plot : volatilité VKF
    subplot(3, 2, 3);
    plot(vkf_signals.volatility, 'r-', 'LineWidth', 1.5);
    title('Volatilité VKF', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Volatilité');
    grid on;
    
    % plot : volatilité MLB_KF
    subplot(3, 2, 4);
    plot(volatilities, 'b-', 'LineWidth', 1.5);
    title('Volatilité MLB-KF', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Volatilité');
    grid on;
    
    % plot : taux d'apprentissage VKF
    subplot(3, 2, 5);
    plot(vkf_signals.learning_rate, 'r-', 'LineWidth', 1.5);
    title('Taux d apprentissage VKF', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Taux d apprentissage');
    grid on;
    
    % plot : taux d'apprentissage MLB_KF
    subplot(3, 2, 6);
    plot(learning_rates, 'b-', 'LineWidth', 1.5);
    title('Taux d apprentissage MLB-KF', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Taux d apprentissage');
    grid on;
    
    
    set(gcf, 'Color', 'w');

end



function [predictions, signals] = vkf(outcomes,lambda,v0,sigma2) % codé tiré de Piray & Daw, comme tel
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


function sequence = generate_continuous_sequence(block_size, iterations, noise_std)
    total_steps = block_size * iterations;
    sequence = zeros(1, total_steps);
    
    for i = 0:iterations-1
        start_idx = i * block_size + 1;
        end_idx = (i + 1) * block_size;
        base_value = 3 * rand() - 1.5;  % uniforme entre -1.5 et 1.5
        sequence(start_idx:end_idx) = base_value + noise_std * randn(1, block_size);
    end
end