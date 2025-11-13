from torch import nn

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, in_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.BatchNorm1d(in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, in_dim * 4),
            nn.BatchNorm1d(in_dim * 4),
            nn.SiLU(),
            nn.Linear(in_dim * 4, in_dim * 8),
            nn.BatchNorm1d(in_dim * 8),
            nn.SiLU(),
            nn.Linear(in_dim * 8, in_dim * 16)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.InstanceNorm1d(in_dim // 2),
            nn.SiLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.InstanceNorm1d(in_dim // 4),
            nn.SiLU(),
            nn.Linear(in_dim // 4, in_dim // 8),
            nn.InstanceNorm1d(in_dim // 8),
            nn.SiLU(),
            nn.Linear(in_dim // 8, in_dim // 16),
            nn.InstanceNorm1d(in_dim // 16),
            nn.SiLU(),
            nn.Linear(in_dim // 16, 1)
        )

    def forward(self, x):
        return self.model(x)
