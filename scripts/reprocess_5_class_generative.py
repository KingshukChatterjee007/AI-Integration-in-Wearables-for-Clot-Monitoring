import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "processed_data/integrated_features_enhanced_CLEAN.csv"
OUT_TRAIN = "processed_data/augmented_5class_train_gen.csv"
OUT_TEST = "processed_data/augmented_5class_test_gen.csv"

# ==========================================
# 1. GENERATIVE ARCHITECTURES
# ==========================================

class TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GeneratorWGANGP(nn.Module):
    def __init__(self, input_dim, noise_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, z):
        return self.net(z)

class CriticWGANGP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. GENERATION ROUTINES
# ==========================================

def train_and_generate_vae(real_data, num_samples, epochs=300):
    input_dim = real_data.shape[1]
    dataset = TensorDataset(torch.FloatTensor(real_data))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = TabularVAE(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            
            # Recon loss (MSE) + KLD
            recon_loss = nn.MSELoss()(recon_x, x)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + 0.1 * kld_loss # Beta-VAE scaling
            
            loss.backward()
            optimizer.step()
            
    # Generate new samples
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 16)
        synthetic_data = model.decode(z).numpy()
        
    return synthetic_data

def train_and_generate_wgan(real_data, num_samples, epochs=300):
    input_dim = real_data.shape[1]
    noise_dim = 32
    dataset = TensorDataset(torch.FloatTensor(real_data))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    gen = GeneratorWGANGP(input_dim, noise_dim)
    critic = CriticWGANGP(input_dim)
    
    opt_g = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_c = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    lambda_gp = 10
    
    for epoch in range(epochs):
        for batch in loader:
            real = batch[0]
            curr_batch_size = real.shape[0]
            
            # --- Train Critic ---
            for _ in range(5): # Critic updates more than generator
                noise = torch.randn(curr_batch_size, noise_dim)
                fake = gen(noise)
                
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                
                # Gradient Penalty
                epsilon = torch.rand(curr_batch_size, 1)
                interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
                critic_interp = critic(interpolated)
                gradients = torch.autograd.grad(
                    outputs=critic_interp,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(critic_interp),
                    create_graph=True,
                    retain_graph=True
                )[0]
                gradients = gradients.view(curr_batch_size, -1)
                gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                
                loss_c = critic_fake - critic_real + lambda_gp * gp
                
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                
            # --- Train Generator ---
            noise = torch.randn(curr_batch_size, noise_dim)
            fake = gen(noise)
            loss_g = -critic(fake).mean()
            
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    gen.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim)
        synthetic_data = gen(noise).numpy()
        
    return synthetic_data

# ==========================================
# 3. MAIN PREPROCESSING PIPELINE
# ==========================================

def main():
    print("Loading clean dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Encode targets to integers explicitly for 5-class
    cat_mapping = {'Low': 0, 'Low-Moderate': 1, 'Moderate': 2, 'High': 3, 'Critical': 4}
    df['target'] = df['risk_category'].map(cat_mapping)
    
    label_encoders = {}
    
    # Isolate categorical vs continuous
    categorical_cols = ['activity', 'gender', 'ppg_channel', 'quality_quality_category', 'data_source']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Exclude metadata and strings from training features
    meta_cols = ['subject_id', 'window_id', 'risk_category', 'target']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    print(f"Total features extracted: {len(feature_cols)}")
    
    # GroupShuffleSplit to enforce ZERO leakage using subject_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['subject_id']))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"Test Subjects: {test_df['subject_id'].unique()}")
    print("Original Train Distribution:")
    print(train_df['target'].value_counts())
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df['target'].values
    
    # Split into classes for processing
    X_low = X_train[y_train == 0]
    X_low_mod = X_train[y_train == 1]
    X_mod = X_train[y_train == 2]
    X_high = X_train[y_train == 3]
    X_crit = X_train[y_train == 4]
    
    TARGET_SAMPLES = len(X_low) # ~3000
    
    print(f"Targeting {TARGET_SAMPLES} samples per class.")
    
    # 1. Keep Low as Baseline
    aug_X = [X_low]
    aug_y = [np.zeros(len(X_low))]
    
    # 2. SMOTE for Low-Mod and Mod (They don't need complex topology generation)
    print("Applying SMOTE for Low-Moderate and Moderate tiers...")
    smote = SMOTE(random_state=42)
    # create a dummy set for smote
    X_dummy_lm = np.vstack([X_low, X_low_mod])
    y_dummy_lm = np.array([0]*len(X_low) + [1]*len(X_low_mod))
    X_res_lm, y_res_lm = smote.fit_resample(X_dummy_lm, y_dummy_lm)
    aug_X.append(X_res_lm[y_res_lm == 1])
    aug_y.append(np.ones(len(X_res_lm[y_res_lm == 1])))
    
    X_dummy_m = np.vstack([X_low, X_mod])
    y_dummy_m = np.array([0]*len(X_low) + [2]*len(X_mod))
    X_res_m, y_res_m = smote.fit_resample(X_dummy_m, y_dummy_m)
    aug_X.append(X_res_m[y_res_m == 2])
    aug_y.append(np.full(len(X_res_m[y_res_m == 2]), 2))
    
    # 3. High Tier: Pre-Jittering + TabularVAE (More stable for High Class)
    samples_needed_high = TARGET_SAMPLES - len(X_high)
    if samples_needed_high > 0:
        print(f"Pre-Jittering {len(X_high)} High samples into 600 variants...")
        noisy_high = []
        for _ in range(5): # 118 * 5 = 590 samples
            jitter = 1.0 + np.random.uniform(-0.1, 0.1, X_high.shape)
            noise = np.random.normal(0, 0.05, X_high.shape)
            noisy_high.append(X_high * jitter + noise)
        X_high_expanded = np.vstack([X_high] + noisy_high)
        
        remaining = TARGET_SAMPLES - len(X_high_expanded)
        if remaining > 0:
            print(f"Generating {remaining} High samples using TabularVAE...")
            synth_high = train_and_generate_vae(X_high_expanded, num_samples=remaining, epochs=250)
            aug_X.append(X_high_expanded)
            aug_X.append(synth_high)
            aug_y.append(np.full(len(X_high_expanded) + len(synth_high), 3))
        else:
            aug_X.append(X_high_expanded[:TARGET_SAMPLES])
            aug_y.append(np.full(TARGET_SAMPLES, 3))
    else:
        aug_X.append(X_high)
        aug_y.append(np.full(len(X_high), 3))
        
    # 4. Critical Tier: Pre-Jittering + WGAN-GP (Decoupled from High)
    samples_needed_crit = TARGET_SAMPLES - len(X_crit)
    if samples_needed_crit > 0:
        print(f"Pre-Jittering Critical ({len(X_crit)}) into broad variants (Reduced High-extrapolation)...")
        
        noisy_crit = []
        # Expand original critical samples
        for _ in range(25): # Increased expansion of pure critical
            jitter = 1.0 + np.random.uniform(-0.1, 0.1, X_crit.shape)
            noise = np.random.normal(0, 0.1, X_crit.shape)
            noisy_crit.append(X_crit * jitter + noise)
            
        # Reduced extrapolation to keep High/Critical distinct
        for _ in range(2): # Reduced from 5 to 2
            jitter = 1.08 + np.random.uniform(-0.05, 0.05, X_high.shape) # Pushed further out
            noise = np.random.normal(0, 0.05, X_high.shape)
            noisy_crit.append(X_high * jitter + noise)
            
        X_crit_expanded = np.vstack([X_crit] + noisy_crit)
        
        remaining = TARGET_SAMPLES - len(X_crit_expanded)
        if remaining > 0:
            print(f"Generating {remaining} Critical samples using WGAN-GP...")
            synth_crit = train_and_generate_wgan(X_crit_expanded, num_samples=remaining, epochs=250)
            aug_X.append(X_crit_expanded)
            aug_X.append(synth_crit)
            aug_y.append(np.full(len(X_crit_expanded) + len(synth_crit), 4))
        else:
            aug_X.append(X_crit_expanded[:TARGET_SAMPLES])
            aug_y.append(np.full(TARGET_SAMPLES, 4))
    else:
        aug_X.append(X_crit)
        aug_y.append(np.full(len(X_crit), 4))
        
    # Recombine Train Data
    X_train_final = np.vstack(aug_X)
    y_train_final = np.concatenate(aug_y).astype(int)
    
    print(f"Final Train Data Shape: {X_train_final.shape}")
    
    # Save Augmented Train
    train_out_df = pd.DataFrame(X_train_final, columns=feature_cols)
    train_out_df['target'] = y_train_final
    train_out_df['split'] = 'train'
    
    # Process Test Data similarly
    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df['target'].values
    test_out_df = pd.DataFrame(X_test, columns=feature_cols)
    test_out_df['target'] = y_test
    test_out_df['split'] = 'test'
    test_out_df['subject_id'] = test_df['subject_id'].values # Important for reporting
    
    print("Saving generalized continuous data...")
    train_out_df.to_csv(OUT_TRAIN, index=False)
    test_out_df.to_csv(OUT_TEST, index=False)
    
    print(f"Successfully wrote augmented subsets to {OUT_TRAIN} and {OUT_TEST}")
    
if __name__ == "__main__":
    os.makedirs("processed_data", exist_ok=True)
    main()
