function simulate()
    block_size = 100;
    iterations = 10;
    noise_std = 0.1;
   
    
    sequence = generate_continuous_sequence(block_size, iterations, noise_std); % séquence avec changements tous les block_size pas de temps
    
    % ----- initialisation des agents -----
    
    % agent basé sur le filtre de Kalman classique
    agent1 = bandits(2, 0.14, 1/15, 1/350, 0.44, 1.5, 0.5, 0.5, 0.05, 0.1);
    
    % agent avec volatilité adaptative inspirée du VKF
    lambda = 0.05; 
    k = 0.1;       
    agent2 = Main21(2, 0.14, 1/15, 1/350, 0.3, 1.5, 0.5, 0.5, 0.05, 0.1, lambda, k);

    % ----- initialisation des tableaux pour stocker les résultats -----
    T = length(sequence);
    
    % résultats pour bandits
    predictions1 = zeros(1, T);
    volatilities1 = zeros(1, T);
    learning_rates1 = zeros(1, T);
    
    %rRésultats pour Main21
    predictions2 = zeros(1, T);
    volatilities2 = zeros(1, T);
    learning_rates2 = zeros(1, T);

    % ----- Simulation des deux agents -----
    for t = 1:T
        % agent 1 : bandits (classique)
        agent1.decide();
        predictions1(t) = agent1.mu(1); % estimation de la récompense pour le bras 1
        volatilities1(t) = agent1.var(1); % incertitude pour le bras 1
        learning_rates1(t) = agent1.var(1) / (agent1.var(1) + agent1.var_ob); % taux d'apprentissage
        
        agent1.update(sequence(t));

        % agent 2 : Main21 (volatilité adaptative)
        agent2.decide();
        predictions2(t) = agent2.mu(1); 
        volatilities2(t) = agent2.var(1); 
        learning_rates2(t) = agent2.var(1) / (agent2.var(1) + agent2.var_ob); 
        
        agent2.update(sequence(t)); 
    end

    % ----- Visualisation des résultats -----
    figure('Position', [100, 100, 2000, 800]);

    % Résultats de l'agent 1 (bandits)
    subplot(3, 2, 1);
    plot(sequence, 'k-', 'LineWidth', 1.5); hold on;
    plot(predictions1, 'b-', 'LineWidth', 1.5);
    title('MLB-KF : Prédictions vs Séquence réelle', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Valeur');
    legend('Séquence réelle', 'Prédictions');
    grid on;

    subplot(3, 2, 3);
    plot(volatilities1, 'b-', 'LineWidth', 1.5);
    title('MLB-KF : Volatilité', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Volatilité');
    grid on;

    subplot(3, 2, 5);
    plot(learning_rates1, 'b-', 'LineWidth', 1.5);
    title('MLB-KF: Taux d apprentissage', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Taux d apprentissage');
    grid on;

    % Résultats de l'agent 2 (Main21)
    subplot(3, 2, 2);
    plot(sequence, 'k-', 'LineWidth', 1.5); hold on;
    plot(predictions2, 'r-', 'LineWidth', 1.5);
    title('2 MLB-KF : Prédictions vs Séquence réelle', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Valeur');
    legend('Séquence réelle', 'Prédictions');
    grid on;

    subplot(3, 2, 4);
    plot(volatilities2, 'r-', 'LineWidth', 1.5);
    title('2 MLB-KF : Volatilité', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Volatilité');
    grid on;

    subplot(3, 2, 6);
    plot(learning_rates2, 'r-', 'LineWidth', 1.5);
    title('2 MLB-KF : Taux d apprentissage', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Timestep'); ylabel('Taux d apprentissage');
    grid on;

    % Ajustement de la mise en page
    set(gcf, 'Color', 'w');
end

function sequence = generate_continuous_sequence(block_size, iterations, noise_std)
    total_steps = block_size * iterations;
    sequence = zeros(1, total_steps);
    
    for i = 0:iterations-1
        start_idx = i * block_size + 1;
        end_idx = (i + 1) * block_size;
        base_value = 3 * rand() - 1.5;  % Uniforme entre -1.5 et 1.5
        sequence(start_idx:end_idx) = base_value + noise_std * randn(1, block_size);
    end
end
