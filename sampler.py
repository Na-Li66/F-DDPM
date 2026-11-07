import torch

class load_sampler:
    def __init__(self, algorithm_name, order, nfe, model, use_ldm, device):
        self.algorithm_name = algorithm_name
        self.order = order
        self.nfe = nfe
        self.model = model
        self.use_ldm = use_ldm
        self.device = device
        self.get_schedule(nfe+1, device, use_ldm, model)
    
    def get_sampler(self):
        if self.algorithm_name == "F-DDPM":
            self.sampler = self.F_DDPM
        elif self.algorithm_name == "F-DDIM":
            self.sampler = self.F_DDIM
        elif self.algorithm_name == "SDE-DPMSolver++":
            self.sampler = self.SDE_DPMSolver_plus
        elif self.algorithm_name == "DDIM":
            self.order = 1
            self.sampler = self.ODESolver
        elif self.algorithm_name == "DPMSolver":
            self.sampler = self.ODESolver
        elif self.algorithm_name == "UniPC":
            self.sampler = self.UniPC
        else:
            raise ValueError(f"No existing algorithm {self.algorithm_name}!")
        return self.sampler

    def F_DDPM(self, xt, class_labels,step_noise):
        device = xt.device
        step_noise = step_noise.reshape(step_noise.shape[0],step_noise.shape[1],-1)

        with torch.no_grad():
            img_size = xt.shape[2]
            num_ch = xt.shape[1]
            x = xt.reshape(xt.shape[0],-1).clone()
            eps_init = x.clone()
            eps_tilde = eps_init.clone()

            for step_idx in range(self.nfe):
                abar = self.alphas_bar[step_idx]
                abar_next = self.alphas_bar[step_idx+1]
                sigma_prime = self.sigma_prime[step_idx]

                lambd_t = 0.5*torch.log(abar/(1-abar))
                lambd_next = 0.5*torch.log(abar_next/(1-abar_next))
                lambd_prime = 0.5*torch.log(abar_next/sigma_prime**2)
                delta = lambd_next - lambd_t
                
                xhat = (abar_next/abar)**(1/2)*x + sigma_prime*(1-torch.exp(lambd_prime-lambd_t))*eps_tilde
                
                t_batch = self.look_ts[step_idx] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                log_snr_p = torch.log(abar_next) - 2*torch.log(sigma_prime)
                at_prime = torch.sigmoid(log_snr_p)
                xprime = xhat/(abar_next+sigma_prime**2)**(1/2)
                eps_hat = self.model.apply_model(xprime.reshape(xhat.shape[0], num_ch, img_size, img_size), t_batch, class_labels, log_snr_p)
                mu_hat = (xhat - sigma_prime*eps_hat)/(abar_next)**(1/2)

                coeff_x = (sigma_prime/(1-abar)**(1/2))
                eta_t = abar_next**(1/2)*(1-torch.exp(lambd_t-lambd_prime))
                coeff_z = ((1-abar_next)-sigma_prime**2)**(1/2)
                x = coeff_x*x + eta_t*mu_hat + coeff_z*step_noise[step_idx]

                eps_tilde = (sigma_prime*eps_hat + (1-abar_next-sigma_prime**2)**(1/2)*step_noise[step_idx])/(1-abar_next)**(1/2)
            return x.reshape(x.shape[0], num_ch, img_size, img_size)
        
    def F_DDIM(self, xt, class_labels):
        device = xt.device

        # load model and evaluate
        with torch.no_grad():
            img_size = xt.shape[2]
            num_ch = xt.shape[1]
            x = xt.reshape(xt.shape[0],-1).clone()
            eps_hist = torch.stack([x.clone() for _ in range(self.order)], dim=0)

            for step_idx in range(self.nfe):
                hist_start = max(step_idx-self.order+1, 0)
                abar_hist = self.alphas_bar[hist_start:step_idx+1].flip(0)
                abar_next = self.alphas_bar[step_idx+1]
                abar = self.alphas_bar[step_idx]
                xhat = self.ODESampler_onestep(x, min(step_idx+1, self.order), abar_next, abar_hist, eps_hist[0:step_idx-hist_start+1])
                coeff_x = ((1-abar_next) / (1-abar))**(1/2)
                coeff_x0 = ((1-abar_next) / (1-abar) * abar)**(1/2) - (abar_next)**(1/2)
                t_batch = self.model_ts[step_idx+1] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                log_snr = torch.log(abar_next) - torch.log(1-abar_next)
                eps = self.model.apply_model(xhat.reshape(xhat.shape[0], num_ch, img_size, img_size), t_batch, class_labels, log_snr)
                x0_pred = (xhat - (1-abar_next)**(1/2)*eps)/(abar_next)**(1/2)
                
                x = coeff_x * x - coeff_x0 * x0_pred
                for idx in range(self.order-1, 0, -1):
                    eps_hist[idx] = eps_hist[idx-1]
                eps_hist[0] = eps.clone()
            return x.reshape(x.shape[0], num_ch, img_size, img_size)

    def SDE_DPMSolver_plus(self, xt, class_labels, step_noise):
        device = xt.device
        step_noise = step_noise.reshape(step_noise.shape[0],step_noise.shape[1],-1)

        # load model and evaluate
        with torch.no_grad():
            img_size = xt.shape[2]
            num_ch = xt.shape[1]
            x = xt.reshape(xt.shape[0], -1).clone()
            
            x0_hist = torch.stack([torch.zeros_like(x) for _ in range(self.order)], dim=0)
            xhat = x.clone()
            
            for step_idx in range(self.nfe):
                hist_start = max(step_idx-self.order+1, 0)
                abar_hist = self.alphas_bar[hist_start:step_idx+1].flip(0)
                abar_next = self.alphas_bar[step_idx+1]
                
                t_batch = self.model_ts[step_idx] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                log_snr = torch.log(self.alphas_bar[step_idx]/(1-self.alphas_bar[step_idx]))
                eps = self.model.apply_model(x.reshape(x.shape[0], num_ch, img_size, img_size), t_batch, class_labels, log_snr)
                x0_hist[0] = (x - (1-abar_hist[0])**(1/2)*eps)/(abar_hist[0])**(1/2)
                xhat = self.SDESampler_onestep(x, p=min(step_idx+1, self.order), abar_next=abar_next, abar_hist=abar_hist, x0_hist=x0_hist)

                lambd_t = 0.5*torch.log(abar_hist[0]/(1-abar_hist[0]))
                lambd_next = 0.5*torch.log(abar_next/(1-abar_next))
                delta = lambd_next - lambd_t
                coeff_z = ((1-abar_next)*(1-torch.exp(-2*delta)))**(1/2)

                x = xhat.clone() + coeff_z*step_noise[step_idx]
                for idx in range(self.order-1, 0, -1):
                    x0_hist[idx] = x0_hist[idx-1]
            return x.reshape(x.shape[0], num_ch, img_size, img_size)
        
    def ODESolver(self, xt, class_labels):
        device = xt.device
        # load model and evaluate
        with torch.no_grad():
            img_size = xt.shape[2]
            num_ch = xt.shape[1]
            x = xt.reshape(xt.shape[0], -1).clone()
            eps_hist = torch.stack([x.clone() for _ in range(self.order)], dim=0)
            xhat = x.clone()
            
            for step_idx in range(self.nfe):
                hist_start = max(step_idx-self.order+1, 0)
                abar_hist = self.alphas_bar[hist_start:step_idx+1].flip(0)
                abar_next = self.alphas_bar[step_idx+1]
                
                t_batch = self.model_ts[step_idx] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                log_snr = torch.log(self.alphas_bar[step_idx]/(1-self.alphas_bar[step_idx]))
                eps_hist[0] = self.model.apply_model(x.reshape(x.shape[0], num_ch, img_size, img_size), t_batch, class_labels, log_snr)
                xhat = self.ODESampler_onestep(x, p=min(step_idx+1, self.order), abar_next=abar_next, abar_hist=abar_hist, eps_hist=eps_hist)
                x = xhat.clone()
                for idx in range(self.order-1, 0, -1):
                    eps_hist[idx] = eps_hist[idx-1]
            return x.reshape(x.shape[0], num_ch, img_size, img_size)
        
    def UniPC(self, xt, class_labels):
        device = torch.device(xt.device)
        with torch.no_grad():
            img_size = xt.shape[2]
            num_ch = xt.shape[1]
            x = xt.reshape(xt.shape[0],-1).clone()
            eps_hist = torch.stack([torch.zeros_like(x.clone()) for _ in range(self.order)], dim=0)
            eps_hist[0] = x.clone()
            xc = x.clone()
            
            for step_idx in range(self.nfe):
                hist_start = max(step_idx-self.order+1, 0)
                abar_hist = self.alphas_bar[hist_start:step_idx+1].flip(0)
                abar_next = self.alphas_bar[step_idx+1]
                t_batch = self.model_ts[step_idx] * torch.ones(x.shape[0], device=device, dtype=torch.long)
                log_snr = torch.log(self.alphas_bar[step_idx]/(1-self.alphas_bar[step_idx]))
                eps_hist[0] = self.model.apply_model(x.reshape(x.shape[0], num_ch, img_size, img_size), t_batch, class_labels, log_snr)
                
                # UniPC 3rd corrector
                if step_idx > 0:
                    xc = self.ODESampler_onestep(xc, p=min(step_idx+1, self.order), abar_next=self.alphas_bar[step_idx], abar_hist=abar_hist[[1, 0] + list(range(2, abar_hist.shape[0]))], eps_hist=eps_hist[[1, 0] + list(range(2, eps_hist.shape[0]))])
                    
                x = self.ODESampler_onestep(xc, p=min(self.nfe-step_idx,min(step_idx+1, self.order-1)), abar_next=abar_next, abar_hist=abar_hist[:self.order-1], eps_hist=eps_hist[:self.order-1])
                for idx in range(self.order-1, 0, -1):
                    eps_hist[idx] = eps_hist[idx-1]
            return x.reshape(x.shape[0], num_ch, img_size, img_size)
    
    def get_schedule(self, num_pts, device, use_ldm, model=None):
        if use_ldm:
            self.alphas_bar = model.alphas_cumprod
            num_steps = num_pts-1
            t_idx = torch.round(torch.linspace(0, len(self.alphas_bar)-1, num_steps+1)).to(device).to(torch.int)
            t_idx = torch.unique(t_idx)
            self.alphas_bar = self.alphas_bar[t_idx].flip(0)
            self.model_ts = t_idx.flip(0)
            lambd = 0.5*torch.log(self.alphas_bar/(1-self.alphas_bar))
            delta = lambd[1:]-lambd[:-1]
            lambd_prime = lambd[1:]+delta
            lambd_all = 0.5*torch.log(model.alphas_cumprod/(1-model.alphas_cumprod))
            self.look_ts = torch.argmin(torch.abs(lambd_prime[:, None] - lambd_all[None, :]), dim=1)
            self.look_ts = torch.minimum(self.look_ts, self.model_ts[1:]).to(t_idx.dtype)
            self.sigma_prime = ((1-model.alphas_cumprod[self.look_ts])/model.alphas_cumprod[self.look_ts]*self.alphas_bar[1:])**(1/2)

        else:
            self.sigma_max = 80
            self.sigma_min = 0.002
            rho = 7
            sched_idx = torch.linspace(0, num_pts-1, num_pts)
            self.model_ts = ((self.sigma_max**(1/rho) + sched_idx/(num_pts-1) * (self.sigma_min**(1/rho) - self.sigma_max**(1/rho)))**(rho)).to(device)
            self.model_ts = self.model_ts.to(torch.float64)
            self.alphas_bar = 1 / (self.model_ts**2+1)
            lambd = 0.5*torch.log(self.alphas_bar/(1-self.alphas_bar))
            delta = lambd[1:]-lambd[:-1]
            self.sigma_prime = (1-self.alphas_bar[1:])**(1/2)*torch.exp(-delta)
            self.look_ts = self.sigma_prime/(self.alphas_bar[1:])**(1/2)
            self.alphas_bar = self.alphas_bar.to(torch.float32)
            self.sigma_prime = self.sigma_prime.to(torch.float32)
            self.look_ts = self.look_ts.to(torch.float32)

    def ODESampler_onestep(self, x, p, abar_next=None, abar_hist=[], eps_hist=[]):
        device = x.device
        abar_hist = abar_hist[:p]
        eps_hist = eps_hist[:p]
        lambd = (1/2) * torch.log(abar_hist / (1 - abar_hist))
        diff = lambd - lambd[0] 
        exponents = torch.arange(len(abar_hist), device=device)
        A = (diff.unsqueeze(1) ** exponents.unsqueeze(0))
        B = eps_hist.reshape(A.shape[0], -1)
        AinvB = torch.linalg.solve(A, B)

        delta = (1/2) * torch.log( abar_next / (1 - abar_next)) - lambd[0]
        phi = torch.zeros(AinvB.shape[0], device=device)
        phi[0] = 1-torch.exp(-delta)
        for i in range(1,AinvB.shape[0]):
            phi[i] = i*phi[i-1] - (delta**i) * torch.exp(-delta)
        Delta = (phi@AinvB).reshape(eps_hist.shape[1], eps_hist.shape[2])

        x_coef = torch.sqrt(abar_next / abar_hist[0])
        eps_coef = torch.sqrt(abar_next)    
        x = x_coef * x - eps_coef * torch.exp(-lambd[0]) * Delta
        return x
    
    def SDESampler_onestep(self, x, p, abar_next=None, abar_hist=[], x0_hist=[]):
        device = x.device
        abar_hist = abar_hist[:p]
        x0_hist = x0_hist[:p]
        lambd = (1/2) * torch.log(abar_hist / (1 - abar_hist))
        lambd_next = (1/2) * torch.log(abar_next / (1 - abar_next))
        delta = lambd_next - lambd
        coeff_x = ((1-abar_next)/(1-abar_hist[0]))**(1/2)*torch.exp(-delta[0])
        eta_t = abar_next**(1/2)*(1-torch.exp(-2*delta[0]))
        if p == 1:
            x = coeff_x*x + eta_t*x0_hist[0]
        elif p == 2:
            Delta = (x0_hist[1] - x0_hist[0])/(lambd[1]-lambd[0])
            x = coeff_x*x + eta_t*x0_hist[0] + eta_t/2*delta[0]*Delta
        return x