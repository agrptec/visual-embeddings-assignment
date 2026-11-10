import torch
import torch.nn as nn

from torchvision.models import resnet18


class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        self.encoder = resnet18(weights=None)

        self.encoder.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1,
            padding=1, bias=False
        )

        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        return h, z


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SimCLR(projection_dim=128).to(device)
    model.train()

    x = torch.randn(8, 3, 32, 32, device=device)

    h, z = model(x)

    print("Device:", device)
    print("Input:", x.shape)
    print("Encoder output:", h.shape)
    print("Projection output:", z.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", parameters)

    loss = z.square().mean()
    loss.backward()

    print("Backward: OK")
