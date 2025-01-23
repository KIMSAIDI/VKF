function simulation()
    
    % Génération de la séquence
    block_size = 100;
    iterations = 10;
    noise_std = 0.1;
    sequence = generate_continuous_sequence(block_size, iterations, noise_std);
    
    % Création de l'agent
    lambda = 0.95;
    k = 0.5;
    agent = Main21(2, 0.14, 1/15, 1/350, 0.3, 1.5, 0.5, 0.5, 0.005, 0.1, lambda, k);
    
    % Initialisation des tableaux pour stocker les résultats
    predictions = zeros(size(sequence));
    volatilities = zeros(size(sequence));
    learning_rates = zeros(size(sequence));
    observations = sequence;
    
    % Initialisation de la volatilité
    current_volatility = 0.1;
    
    
    % Simulation
    for t = 1:length(sequence)
        agent.decide();
        predictions(t) = agent.mu(1);
        surprise = abs(sequence(t) - predictions(t));
        
        % Mise à jour de la volatilité basée sur la surprise
        current_volatility = lambda * current_volatility + (1 - lambda) * k * surprise^2;
        volatilities(t) = current_volatility;
        
        % Mise à jour du bruit de processus basé sur la volatilité
        agent.var_tr = current_volatility;
        
        % Calcul du taux d'apprentissage adaptatif
        learning_rate = current_volatility / (current_volatility + agent.var_ob);
        learning_rates(t) = learning_rate;
        
        % Mise à jour
        agent.update(sequence(t));
    end
    
  
    
    % ---- plot ----
    figure('Position', [100, 100, 1200, 800]);
    
    % plot 1: Prédictions vs Observations
    subplot(3,1,1);
    p = plot(observations, 'k-', 'LineWidth', 1);
    set(p, 'Color', [0 0 0 0.7]);
    hold on;
    plot(predictions, 'r-', 'LineWidth', 1.5);
    title('Prédictions et Observations', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Valeur');
    legend('Observations', 'Prédictions (VKF)');
    grid on;
    
    % plot 2: Volatilité
    subplot(3,1,2);
    plot(volatilities, 'b-', 'LineWidth', 2);
    title('Volatilité Adaptative', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Volatilité');
    legend('Volatilité');
    grid on;
    
    % plot 3: Taux d'apprentissage
    subplot(3,1,3);
    plot(learning_rates, 'b-', 'LineWidth', 2);
    title('Taux d''Apprentissage Adaptatif', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep');
    ylabel('Taux');
    legend('Taux d''Apprentissage');
    grid on;
    
    
    set(gcf, 'Color', 'w');
end
   

function sequence = generate_continuous_sequence(block_size, iterations, noise_std)
    total_steps = block_size * iterations;
    sequence = zeros(1, total_steps);
    
    for i = 0:iterations-1
        start_idx = i * block_size + 1;
        end_idx = (i + 1) * block_size;
        base_value = 3 * rand() - 1.5; % Uniforme entre -1.5 et 1.5
        sequence(start_idx:end_idx) = base_value + noise_std * randn(1, block_size);
    end
end
