function simulation_tmp()
    % Start performance monitoring
    tic;
    
    % Initial CPU measurement
    if ispc() % Windows
        [~, initial_cpu_str] = system('wmic cpu get loadpercentage');
        initial_cpu = str2double(regexp(initial_cpu_str, '\d+', 'match'));
    elseif isunix() % Linux/Mac
        [~, initial_cpu_str] = system('top -n 1 | grep "Cpu(s)" | awk ''{print $2}''');
        initial_cpu = str2double(initial_cpu_str);
    end
    
    % Initial memory measurement
    [usedMemory1,~] = memory;
    initial_memory = usedMemory1.MemUsedMATLAB;
    
    % Initialize parallel pool if available
    try
        if isempty(gcp('nocreate'))
            parpool('local');
        end
        parallel_available = true;
        pool = gcp;
        num_workers = pool.NumWorkers;
    catch
        parallel_available = false;
        num_workers = 0;
    end
    
    % Génération de la séquence
    block_size = 100;
    iterations = 10;
    noise_std = 0.2;
    sequence = generate_continuous_sequence(block_size, iterations, noise_std);
    
    % Création de l'agent
    lambda = 0.98;
    k = 0.5;
    agent = Main21(2, 0.14, 1/15, 1/350, 0.44, 1.5, 0.5, 0.5, 0.005, 0.1, lambda, k);
    
    % Initialisation des tableaux pour stocker les résultats
    predictions = zeros(size(sequence));
    volatilities = zeros(size(sequence));
    learning_rates = zeros(size(sequence));
    observations = sequence;
    
    % Initialisation de la volatilité
    current_volatility = 0.1;
    
    % mesure main loop time
    main_loop_timer = tic;
    
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
    
    % Calculate main loop time
    main_loop_time = toc(main_loop_timer);
    
    % Final measurements
    execution_time = toc;
    [usedMemory2,~] = memory;
    final_memory = usedMemory2.MemUsedMATLAB;
    memory_used = final_memory - initial_memory;
    
    % Final CPU measurement
    if ispc()
        [~, final_cpu_str] = system('wmic cpu get loadpercentage');
        final_cpu = str2double(regexp(final_cpu_str, '\d+', 'match'));
    elseif isunix()
        [~, final_cpu_str] = system('top -n 1 | grep "Cpu(s)" | awk ''{print $2}''');
        final_cpu = str2double(final_cpu_str);
    end
    cpu_usage = final_cpu - initial_cpu;
    
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
    
    % Create performance metrics figure
    figure('Position', [100, 100, 400, 300]);
    annotation('textbox', [0.1, 0.1, 0.8, 0.8], ...
        'String', {...
            'Performance Metrics:', ...
            sprintf('Total execution time: %.2f s', execution_time), ...
            sprintf('Main loop time: %.2f s', main_loop_time), ...
            sprintf('Memory used: %.2f MB', memory_used/1024/1024), ...
            sprintf('CPU usage change: %.2f%%', cpu_usage), ...
            sprintf('Parallel Workers: %d', num_workers) ...
        }, ...
        'FontSize', 12, ...
        'EdgeColor', 'none');
    
    % Display console metrics
    fprintf('\nPerformance Metrics:\n');
    fprintf('Total execution time: %.2f seconds\n', execution_time);
    fprintf('Main loop time: %.2f seconds\n', main_loop_time);
    fprintf('Memory used: %.2f MB\n', memory_used/1024/1024);
    fprintf('CPU usage change: %.2f%%\n', cpu_usage);
    if parallel_available
        fprintf('Number of parallel workers: %d\n', num_workers);
    end
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
