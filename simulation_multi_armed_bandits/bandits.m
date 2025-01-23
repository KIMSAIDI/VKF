classdef bandits < handle
    properties
        narms   % nombre de bras
        aq      % pour ajuster la vitesse d'apprentissage des moyennes
        am      % poids pour calculer la récompense moyenne
        al      % poids pour calculer récompense moyenne lissée
        eta     % taux de mise à jour de beta (inverse température pour softmax)
        phi     % facteur d'exploration pour pondérer l'incertitude
        mu      % initialisation moyenne
        var     % initialisation variance
        var_ob  % variance d'observation (incertitude des récompenses)
        var_tr  % variance de transition (incertitude ajoutée aux bras non choisis)
        beta
        rbar
        rbbar
        action
        q
        probs
        
    end
    
    methods
        function obj = bandits(narms, aq, am, al, eta, phi, mu0, var0, var_ob, var_tr)
            obj.narms = narms;
            obj.aq = aq;
            obj.am = am;
            obj.al = al;
            obj.eta = eta;
            obj.phi = phi;
            obj.mu = mu0 * ones(1, narms);
            obj.var = var0 * ones(1, narms);
            obj.beta = 0;
            obj.rbar = 0;
            obj.rbbar = 0;
            obj.action = 1;
            obj.q = zeros(1, narms);
            obj.probs = ones(1, narms) / narms;
            obj.var_ob = var_ob; 
            obj.var_tr = var_tr;  
            
            
        end
        
        function action = decide(obj)
            % calcul des valeurs d'action (Q-values)
            obj.q = obj.mu + obj.phi * sqrt(obj.var);
            
            % calcul des probabilités softmax 
            logits = obj.beta * obj.q;
            logits = logits - max(logits);  
            
            exp_logits = exp(logits);
            obj.probs = exp_logits / sum(exp_logits);
            
            % on sélectionne l'action de manière probabiliste
            cumulative_probs = cumsum(obj.probs);                       % probabilités cumulées
            random_number = rand();                                     % on génère un nombre aléatoire entre 0 et 1
            obj.action = find(cumulative_probs >= random_number, 1);    % la première action dépassant le nombre aléatoire

            action = obj.action;
        end
        
         function update(obj, reward)
            % update des moyennes de récompense
            obj.rbar = obj.am * reward + (1 - obj.am) * obj.rbar;
            obj.rbbar = obj.al * obj.rbar + (1 - obj.al) * obj.rbbar;
            
            % update pour le bras choisi
            a = obj.action;
            prediction_error = reward - obj.mu(a);
      

            % on ajuste la variance de transition en fonction de l'erreur de prédiction
            obj.var_tr = max(0.001, min(0.5, abs(prediction_error) * 0.1));
            
            % update Kalman avec variance adaptative
            kalman_gain = (obj.var(a) + obj.var_tr) / (obj.var(a) + obj.var_tr + obj.var_ob);
            obj.mu(a) = obj.mu(a) + kalman_gain * prediction_error;
            obj.var(a) = (1 - kalman_gain) * (obj.var(a) + obj.var_tr);
            
            % augmentation de l'incertitude pour les bras non choisis
            non_chosen = true(1, obj.narms);
            non_chosen(a) = false;
            obj.var(non_chosen) = obj.var(non_chosen) + obj.var_tr;
            
            % mise à jour de beta
            obj.beta = max(obj.beta + obj.eta * (obj.rbar - obj.rbbar), 0);
         end

    end
end

