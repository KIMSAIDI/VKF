classdef Main21 < handle
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
        % ajout des paramètres suivants
        
        lambda              % lissage pour la volatilité
        k                   % pondération de la surprise
        current_vol         % volatilité adaptative
        surprise            % écart entre la surprise la récompense observée et la récompense attendue
        action_index
    end
    
    methods
        function obj = Main21(narms, aq, am, al, eta, phi, mu0, var0, var_ob, var_tr, lambda, k)
            obj.narms = narms;
            obj.aq = aq;
            obj.am = am;
            obj.al = al;
            obj.eta = eta;
            obj.phi = phi;
            obj.mu = mu0 * ones(narms, 1);      
            obj.var = var0 * ones(narms, 1);    
            obj.beta = 0;
            obj.rbar = 0;
            obj.rbbar = 0;
            obj.action = 1;
            obj.action_index = 0;
            obj.q = zeros(narms, 1);            
            obj.probs = ones(narms, 1) / narms; 
            obj.var_ob = var_ob;
            obj.var_tr = var_tr;
            
            % ajout des paramètres suivants
            obj.lambda = lambda;
            obj.k = k;
            obj.current_vol = var0;
            obj.surprise = 0;
        end
        
        function action = decide(obj)
            % calcul des valeurs d'action (Q-values)
            obj.q = obj.mu + obj.phi * (obj.var .^ 0.5); 
            
            logits = obj.beta * obj.q;
            logits = logits - max(logits);
            obj.probs = exp(logits) / sum(exp(logits));
            
            obj.action_index = randsample(1:obj.narms, 1, true, obj.probs);
            
            action = obj.action_index;
        end
        
        function update(obj, reward)
            % update des moyennes de récompense
            obj.rbar = obj.am * reward + (1 - obj.am) * obj.rbar;
            obj.rbbar = obj.al * obj.rbar + (1 - obj.al) * obj.rbbar;
            
            % update pour le bras choisi
            a = obj.action_index;
            
            % calcul de la surprise
            obj.surprise = abs(reward - obj.mu(a));
            
            % on update la volatilité adaptative basé sur la surprise
            obj.current_vol = obj.lambda * obj.current_vol + (1 - obj.lambda) * obj.k * obj.surprise^2;
            
            % on update la variance basé sur la volatilité
            obj.var_tr = obj.current_vol;
            
            % update du filtre de Kalman avec la volatilité adaptative
            a_not = setdiff(1:obj.narms, a);
            
            % calcul du gain de Kalman avec la nouvelle volatilité
            kalman_gain = (obj.var(a) + obj.current_vol) / (obj.var(a) + obj.current_vol + obj.var_ob);
            
            % update de la moyenne et de la variance pour les actions choisies 
            obj.mu(a) = obj.mu(a) + kalman_gain * (reward - obj.mu(a));
            obj.var(a) = (1 - kalman_gain) * (obj.var(a) + obj.current_vol);
            
            % augmentation de l'incertitude pour les bras non choisis
            obj.var(a_not) = obj.var(a_not) + obj.current_vol;
            
            % mise à jour de beta
            obj.beta = max(obj.beta + obj.eta * (obj.rbar - obj.rbbar), 0);
        end
        
       
    end
end