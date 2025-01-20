classdef Main2 < handle
    properties
        narms
        aq
        am
        al
        eta
        phi
        mu
        var
        var_ob
        var_tr
        beta
        rbar
        rbbar
        action
        q
        probs
        
    end
    
    methods
        function obj = Main2(narms, aq, am, al, eta, phi, mu0, var0, var_ob, var_tr)
            obj.narms = narms;
            obj.aq = aq;
            obj.am = am;
            obj.al = al;
            obj.eta = eta;
            obj.phi = phi;
            obj.mu = mu0 * ones(1, narms);
            obj.var = var0 * ones(1, narms);
            %obj.var_ob = var_ob;
            %obj.var_tr = var_tr;
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
            % Calcul des valeurs d'action (Q-values)
            obj.q = obj.mu + obj.phi * sqrt(obj.var);
            
            % Calcul des probabilités softmax avec stabilité numérique
            logits = obj.beta * obj.q;
            logits = logits - max(logits);  % Pour la stabilité numérique
            exp_logits = exp(logits);
            obj.probs = exp_logits / sum(exp_logits);
            
            % Sélection de l'action
            [~, obj.action] = max(obj.probs);
            action = obj.action;
        end
        
         function update(obj, reward)
            % Mise à jour des moyennes de récompense
            obj.rbar = obj.am * reward + (1 - obj.am) * obj.rbar;
            obj.rbbar = obj.al * obj.rbar + (1 - obj.al) * obj.rbbar;
            
            % Mise à jour pour le bras choisi
            a = obj.action;
            prediction_error = reward - obj.mu(a);
      

            % Ajuster la variance de transition en fonction de l'erreur de prédiction
            obj.var_tr = max(0.001, min(0.5, abs(prediction_error) * 0.1));
            
            % Mise à jour Kalman avec variance adaptative
            kalman_gain = (obj.var(a) + obj.var_tr) / (obj.var(a) + obj.var_tr + obj.var_ob);
            obj.mu(a) = obj.mu(a) + kalman_gain * prediction_error;
            obj.var(a) = (1 - kalman_gain) * (obj.var(a) + obj.var_tr);
            
            % Augmentation de l'incertitude pour les bras non choisis
            non_chosen = true(1, obj.narms);
            non_chosen(a) = false;
            obj.var(non_chosen) = obj.var(non_chosen) + obj.var_tr;
            
            % Mise à jour de beta
            obj.beta = max(obj.beta + obj.eta * (obj.rbar - obj.rbbar), 0);
         end

    end
end

